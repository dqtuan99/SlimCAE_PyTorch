import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import numpy as np
import math
import os
import glob
import argparse
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

from compressai.layers import GDN
from compressai.entropy_models import EntropyBottleneck
from pytorch_msssim import ms_ssim
from torch.utils.tensorboard import SummaryWriter

# ==============================================================================
# 1. CUSTOM DYNAMIC LAYERS
# ==============================================================================

class DynamicConv2d(nn.Module):
    """ A dynamic 2D convolution layer that can switch between different channel configurations. """
    def __init__(self, in_channels_list, out_channels_list, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
            for in_ch, out_ch in zip(in_channels_list, out_channels_list)
        ])
        self._active_level = 0

    def forward(self, x):
        return self.convs[self._active_level](x)

    def set_active_level(self, level):
        self._active_level = level

class DynamicConvTranspose2d(nn.Module):
    """ A dynamic 2D transposed convolution layer. """
    def __init__(self, in_channels_list, out_channels_list, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding, bias=bias)
            for in_ch, out_ch in zip(in_channels_list, out_channels_list)
        ])
        self._active_level = 0

    def forward(self, x):
        return self.convs[self._active_level](x)

    def set_active_level(self, level):
        self._active_level = level

# ==============================================================================
# 2. DATA HANDLING
# ==============================================================================

def get_padding(h, w, p=64):
    """ Calculates padding for a given height and width to make them divisible by p. """
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    return padding_left, padding_right, padding_top, padding_bottom

class ImageDataset(Dataset):
    """ A simple dataset to load images from a folder. """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        extensions = ["*.png", "*.jpg", "*.jpeg"]
        for ext in extensions:
            self.image_files.extend(glob.glob(os.path.join(root_dir, ext)))

        if not self.image_files:
            print(f"Warning: No images found in {root_dir} with extensions {extensions}")

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# ==============================================================================
# 3. MODEL DEFINITION
# ==============================================================================

class SlimCAE_TF_Matched(nn.Module):
    """
    Slimmable Compressive Autoencoder with an architecture that closely matches
    the provided TensorFlow implementation.
    """
    def __init__(self, total_filters=128, switch_list=(128, 96, 64)):
        super().__init__()
        self.total_filters = total_filters
        self.switch_list = switch_list
        self.num_levels = len(switch_list)

        # --- ENCODER ---
        self.enc_layer_0 = DynamicConv2d([3] * self.num_levels, switch_list, kernel_size=9, stride=4, padding=4, bias=True)
        self.gdn_an_0 = nn.ModuleList([GDN(ch) for ch in switch_list])
        
        self.enc_layer_1 = DynamicConv2d([self.total_filters] * self.num_levels, switch_list, kernel_size=5, stride=2, padding=2, bias=True)
        self.gdn_an_1 = nn.ModuleList([GDN(ch) for ch in switch_list])

        self.enc_layer_2 = DynamicConv2d([self.total_filters] * self.num_levels, switch_list, kernel_size=5, stride=2, padding=2, bias=False)
        self.gdn_an_2 = nn.ModuleList([GDN(ch) for ch in switch_list])

        # --- DECODER ---
        self.igdn_sy_0 = nn.ModuleList([GDN(ch, inverse=True) for ch in switch_list])
        self.dec_layer_0 = DynamicConvTranspose2d([self.total_filters] * self.num_levels, [self.total_filters] * self.num_levels, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True)

        self.igdn_sy_1 = nn.ModuleList([GDN(self.total_filters, inverse=True) for _ in switch_list])
        self.dec_layer_1 = DynamicConvTranspose2d([self.total_filters] * self.num_levels, [self.total_filters] * self.num_levels, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True)
        
        self.igdn_sy_2 = nn.ModuleList([GDN(self.total_filters, inverse=True) for _ in switch_list])
        self.dec_layer_2 = DynamicConvTranspose2d([self.total_filters] * self.num_levels, [3] * self.num_levels, kernel_size=9, stride=4, padding=4, output_padding=3, bias=True)

        self.entropy_bottlenecks = nn.ModuleList([
            EntropyBottleneck(channels) for channels in switch_list
        ])

    def forward(self, x):
        reconstructions, likelihoods_list = [], []

        for i in range(self.num_levels):
            _switch = self.switch_list[i]
            
            for module in self.modules():
                if hasattr(module, 'set_active_level'):
                    module.set_active_level(i)

            y = self.enc_layer_0(x)
            y = self.gdn_an_0[i](y)
            y = F.pad(y, (0, 0, 0, 0, 0, self.total_filters - _switch))

            y = self.enc_layer_1(y)
            y = self.gdn_an_1[i](y)
            y = F.pad(y, (0, 0, 0, 0, 0, self.total_filters - _switch))

            y = self.enc_layer_2(y)
            y = self.gdn_an_2[i](y)
            
            y_hat, likelihoods = self.entropy_bottlenecks[i](y)
            likelihoods_list.append(likelihoods)
            
            x_hat = self.igdn_sy_0[i](y_hat)
            x_hat = F.pad(x_hat, (0, 0, 0, 0, 0, self.total_filters - _switch))
            
            x_hat = self.dec_layer_0(x_hat)
            x_hat = self.igdn_sy_1[i](x_hat)
            
            x_hat = self.dec_layer_1(x_hat)
            x_hat = self.igdn_sy_2[i](x_hat)
            
            x_hat = self.dec_layer_2(x_hat)
            
            reconstructions.append(x_hat)

        return reconstructions, likelihoods_list
        
# ==============================================================================
# 4. CORE TRAINING AND EVALUATION LOGIC
# ==============================================================================

def rate_distortion_loss(reconstructions, original, likelihoods_list, lmbdas, num_pixels):
    total_loss = 0
    bpp_list, mse_list = [], []

    for i in range(len(reconstructions)):
        mse = F.mse_loss(reconstructions[i], original)
        bpp = torch.log(likelihoods_list[i]).sum() / (-math.log(2) * num_pixels)
        rd_loss = lmbdas[i] * (mse * 255**2) + bpp
        total_loss += rd_loss
        
        bpp_list.append(bpp)
        mse_list.append(mse)

    return total_loss, bpp_list, mse_list

def train_one_epoch(model, dataloader, optimizer, lmbdas, 
                    patch_size, device, start_step=0, writer=None, clip_max_norm=1.0):
    model.train()
    num_pixels = patch_size ** 2
    current_step = start_step
    
    for images in tqdm(dataloader, desc=f"Training (Step {start_step})"):
        images = images.to(device)
        optimizer.zero_grad()
        reconstructions, likelihoods_list = model(images)
        
        rd_loss, bpp_list, mse_list = rate_distortion_loss(
            reconstructions, images, likelihoods_list, lmbdas, num_pixels
        )
        
        aux_loss = sum(eb.loss() for eb in model.entropy_bottlenecks)
        total_loss_for_backward = rd_loss + aux_loss
        total_loss_for_backward.backward()
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        
        optimizer.step()

        if current_step % 10 == 0 and writer is not None:
            writer.add_scalar('Loss/rate_distortion', rd_loss.item(), current_step)
            writer.add_scalar('Loss/auxiliary', aux_loss.item(), current_step)
            writer.add_scalar('Loss/total', total_loss_for_backward.item(), current_step)
            for i in range(len(bpp_list)):
                writer.add_scalar(f'BPP/level_{i}', bpp_list[i].item(), current_step)
                writer.add_scalar(f'MSE/level_{i}', mse_list[i].item() * 255**2, current_step)
                
        current_step += 1
    return current_step

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    num_levels = model.num_levels
    
    bpp_avg = [0.0] * num_levels
    psnr_avg = [0.0] * num_levels
    mse_avg = [0.0] * num_levels
    msssim_avg = [0.0] * num_levels
    msssim_db_avg = [0.0] * num_levels
    count = 0

    for images in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        h, w = images.shape[2:]
        num_pixels = h * w
        
        pad_l, pad_r, pad_t, pad_b = get_padding(h, w)
        padded_images = F.pad(images, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
        reconstructions, likelihoods_list = model(padded_images)
        
        for i in range(num_levels):
            rec_i = reconstructions[i][..., pad_t:pad_t+h, pad_l:pad_l+w].clamp(0, 1)
            bpp = torch.log(likelihoods_list[i]).sum() / (-math.log(2) * num_pixels)
            mse = F.mse_loss(rec_i, images)
            psnr = 10 * torch.log10(1.0 / mse)
            msssim_val = ms_ssim(rec_i, images, data_range=1.0)
            msssim_db = -10 * torch.log10(1 - msssim_val)
            
            bpp_avg[i] += bpp.item()
            psnr_avg[i] += psnr.item()
            mse_avg[i] += mse.item()
            msssim_avg[i] += msssim_val.item()
            msssim_db_avg[i] += msssim_db.item()
        count += 1
        
    for i in range(num_levels):
        bpp_avg[i] /= count
        psnr_avg[i] /= count
        mse_avg[i] /= count
        msssim_avg[i] /= count
        msssim_db_avg[i] /= count
        
    return bpp_avg, psnr_avg, mse_avg, msssim_avg, msssim_db_avg

# ==============================================================================
# 5. MAIN WORKFLOW FUNCTIONS
# ==============================================================================

def train_model(args, device):
    print("Starting Stage 1: Initial model training.")
    
    train_transform = transforms.Compose([
        transforms.Resize(args.patchsize),
        transforms.RandomCrop(args.patchsize),
        transforms.ToTensor(),
    ])
    train_dataset = ImageDataset(root_dir=args.train_glob, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.preprocess_threads)
    
    model = SlimCAE_TF_Matched(total_filters=args.num_filters, switch_list=args.switch_list).to(device)
    
    main_params = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
    aux_params = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]
    optimizer = optim.Adam(
        [
            {"params": main_params, "lr": 1e-4},
            {"params": aux_params, "lr": 1e-3},
        ]
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    num_steps = 0
    if args.resume:
        try:
            checkpoint_files = glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_*.pth"))
            if not checkpoint_files: raise FileNotFoundError

            latest_ckpt_path = max(
                checkpoint_files, 
                key=lambda f: int("".join(filter(str.isdigit, f)))
            )
            
            print(f"Resuming training from checkpoint: {latest_ckpt_path}")
            checkpoint = torch.load(latest_ckpt_path, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            num_steps = checkpoint['step']
            args.lmbda = checkpoint['lmbda']

        except (FileNotFoundError, KeyError, IndexError):
            print("Could not find a valid checkpoint. Starting from scratch.")
            num_steps = 0

    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'tensorboard_logs'))
    max_steps = args.last_step
    
    while num_steps < max_steps:
        epoch = (num_steps // len(train_loader)) + 1
        print(f"\n--- Training Epoch {epoch} (Step ~{num_steps}/{max_steps}) ---")
        
        num_steps = train_one_epoch(
            model, train_loader, optimizer, args.lmbda, 
            args.patchsize, device, start_step=num_steps, writer=writer
        )

        if num_steps > 0 and num_steps % 10000 < len(train_loader):
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{num_steps}.pth")
            torch.save({
                'step': num_steps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lmbda': args.lmbda,
            }, checkpoint_path)
            print(f"✅ Checkpoint saved to {checkpoint_path}")

    writer.close()
    print("Stage 1 training finished.")

def train_lambda_schedule(args, device):
    print("Starting Stage 2: Lambda scheduling.")
    
    # Dataset for fine-tuning steps (uses training data)
    fine_tune_transform = transforms.Compose([
        transforms.Resize(args.patchsize),
        transforms.RandomCrop(args.patchsize),
        transforms.ToTensor()
    ])
    fine_tune_dataset = ImageDataset(root_dir=args.train_glob, transform=fine_tune_transform)
    
    # Dataset for evaluation steps (uses validation/test data)
    eval_transform = transforms.Compose([transforms.ToTensor()])
    eval_dataset = ImageDataset(root_dir=args.inputPath, transform=eval_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    
    model = SlimCAE_TF_Matched(total_filters=args.num_filters, switch_list=args.switch_list).to(device)
    
    main_params = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
    aux_params = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]
    optimizer = optim.Adam(
        [
            {"params": main_params, "lr": 1e-5}, # Lower LR for fine-tuning
            {"params": aux_params, "lr": 1e-3},
        ]
    )

    checkpoint_files = glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_*.pth"))
    if not checkpoint_files: raise FileNotFoundError("No checkpoints found. Please run Stage 1 training first.")
    
    latest_ckpt = max(checkpoint_files, key=lambda f: int("".join(filter(str.isdigit, f))))
    
    print(f"Loading checkpoint: {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Successfully loaded optimizer state.")
    else:
        print("Warning: Optimizer state not found or in old format. Re-initializing optimizer.")
    
    num_steps = checkpoint.get('step', 0)
    lmbdas = list(args.lmbda)
    
    bpp, psnr, _, _, _ = evaluate(model, eval_loader, device)
    
    print(f"Initial BPPs: {[f'{x:.4f}' for x in bpp]}, PSNRs: {[f'{x:.2f}' for x in psnr]}")

    for i in range(len(lmbdas) - 1):
        print(f"\n--- Adjusting lambdas for level {i+1} and beyond ---")
        
        if abs(bpp[i] - bpp[i+1]) < 1e-6: continue
        grad_flag = (psnr[i] - psnr[i+1]) / (bpp[i] - bpp[i+1])
        
        m = 1
        while m < 7:
            for j in range(i + 1, len(lmbdas)):
                lmbdas[j] *= 0.9
            print(f"Iteration {m}: New lambdas: {[f'{l:.4f}' for l in lmbdas]}")

            fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=args.batchsize, shuffle=True)
            for _ in range(5):
                train_one_epoch(
                    model, fine_tune_loader, optimizer, lmbdas, 
                    args.patchsize, device, start_step=0, clip_max_norm=0
                )
            
            bpp, psnr, _, _, _ = evaluate(model, eval_loader, device)
            print(f"Re-evaluated BPPs: {[f'{x:.4f}' for x in bpp]}, PSNRs: {[f'{x:.2f}' for x in psnr]}")

            if abs(bpp[i] - bpp[i+1]) < 1e-6: break
            grad_current = (psnr[i] - psnr[i+1]) / (bpp[i] - bpp[i+1])

            print(f"Slope check: Current grad={grad_current:.2f}, Target grad={grad_flag:.2f}")
            if grad_current > grad_flag:
                print("✅ Slope improved. Moving to next level.")
                break
            else:
                grad_flag = grad_current
                m += 1
                if m >= 7: print("⚠️ Max iterations reached. Moving on.")
    
    final_path = os.path.join(args.checkpoint_dir, "final_scheduled_model.pth")
    torch.save({
        'step': num_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lmbda': lmbdas,
    }, final_path)
    
    print(f"\n--- Lambda Scheduling Complete. Final model saved to {final_path} ---")
    print(f"Final optimized lambdas: {[f'{l:.4f}' for l in lmbdas]}")

def test_model(args, device):
    num_models = len(args.checkpoint_paths)
    if num_models not in [1, 2]:
        raise ValueError("This script can only evaluate 1 or 2 models at a time. Please provide 1 or 2 paths to --checkpoint_paths.")

    print("Setting up dataset and model...")
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = ImageDataset(root_dir=args.inputPath, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = SlimCAE_TF_Matched(total_filters=args.num_filters, switch_list=args.switch_list).to(device)

    if num_models == 1:
        print("\n--- Running Single-Model Evaluation ---")
        checkpoint_path = args.checkpoint_paths[0]
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        bpp, psnr, mse, msssim, msssim_db = evaluate(model, test_loader, device)
        
        print("\n--- Final Test Results ---")
        for i in range(model.num_levels):
            print(
                f"Level {i} (C={model.switch_list[i]}): "
                f"BPP: {bpp[i]:.4f} | PSNR: {psnr[i]:.2f} dB | MSE: {mse[i]*255**2:.2f} | "
                f"MS-SSIM: {msssim[i]:.4f} | MS-SSIM dB: {msssim_db[i]:.2f} dB"
            )

        if args.report_path:
            report_dir = os.path.dirname(args.report_path)
            if report_dir: os.makedirs(report_dir, exist_ok=True)
            report_file = args.report_path + ".csv"
            with open(report_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ["Level", "Channels", "BPP", "PSNR (dB)", "MSE", "MS-SSIM", "MS-SSIM (dB)"]
                writer.writerow(header)
                for i in range(model.num_levels):
                    writer.writerow([
                        i, model.switch_list[i], bpp[i], psnr[i], mse[i]*255**2, msssim[i], msssim_db[i]
                    ])
            print(f"\n✅ Report saved to {report_file}")

        if args.report_path:
            plt.style.use('seaborn-v0_8-whitegrid')
            
            print("Generating plots for metrics vs. model width...")
            channel_counts = sorted(model.switch_list)
            original_indices = {ch: i for i, ch in enumerate(model.switch_list)}
            sorted_indices = [original_indices[ch] for ch in channel_counts]
            
            plot_configs = [
                ([psnr[i] for i in sorted_indices], "Peak Signal-to-Noise Ratio (PSNR)", "PSNR (dB) (Higher is Better ↑)", "_psnr.png"),
                ([mse[i] * 255**2 for i in sorted_indices], "Mean Squared Error (MSE)", "MSE (Lower is Better ↓)", "_mse.png"),
                ([bpp[i] for i in sorted_indices], "Bits Per Pixel (bpp)", "bpp (Lower is Better ↓)", "_bpp.png"),
                ([msssim[i] for i in sorted_indices], "Multi-Scale SSIM", "MS-SSIM (Higher is Better ↑)", "_msssim.png"),
                ([msssim_db[i] for i in sorted_indices], "MS-SSIM (dB)", "MS-SSIM (dB) (Higher is Better ↑)", "_msssim_db.png"),
            ]
            
            for y_data, title, ylabel, suffix in plot_configs:
                plt.figure(figsize=(10, 6))
                plt.plot(channel_counts, y_data, 'o-')
                plt.title(title, fontsize=16)
                plt.xlabel("Model Width (# of Filters)", fontsize=12)
                plt.ylabel(ylabel, fontsize=12)
                plt.xticks(channel_counts)
                plt.savefig(args.report_path + suffix, dpi=300)
                plt.close()

            print("Generating Rate-Distortion curve plots...")
            plt.figure(figsize=(10, 7))
            plt.plot(bpp, psnr, 'o--', color='#c44e52', label='SlimCAE')
            for i, txt in enumerate(model.switch_list):
                plt.text(bpp[i] * 1.02, psnr[i], f"{txt} filters", fontsize=12)
            plt.title('Rate-Distortion Curve', fontsize=16)
            plt.xlabel('Rate (Bits Per Pixel)', fontsize=12)
            plt.ylabel('Quality (PSNR in dB)', fontsize=12)
            plt.grid(True, which='both', linestyle='--')
            plt.legend()
            plt.savefig(args.report_path + "_rd_psnr.png", dpi=300)
            plt.close()

            plt.figure(figsize=(10, 7))
            plt.plot(bpp, msssim_db, 'o--', color='#4c72b0', label='SlimCAE')
            for i, txt in enumerate(model.switch_list):
                plt.text(bpp[i] * 1.02, msssim_db[i], f"{txt} filters", fontsize=12)
            plt.title('Rate-Distortion Curve', fontsize=16)
            plt.xlabel('Rate (Bits Per Pixel)', fontsize=12)
            plt.ylabel('Quality (MS-SSIM in dB)', fontsize=12)
            plt.grid(True, which='both', linestyle='--')
            plt.legend()
            plt.savefig(args.report_path + "_rd_msssim_db.png", dpi=300)
            plt.close()
            
            print(f"✅ All plots saved to the directory of your report path.")

    elif num_models == 2:
        print("\n--- Running Two-Model Comparison ---")
        model_paths = args.checkpoint_paths
        model_labels = ["Before Fine-Tuning", "After Fine-Tuning"]
        results = {}

        for i, path in enumerate(model_paths):
            label = model_labels[i]
            print(f"\n--- Evaluating Model: '{label}' ---")
            print(f"Loading checkpoint from: {path}")
            
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            bpp, psnr, mse, msssim, msssim_db = evaluate(model, test_loader, device)
            results[label] = {
                "bpp": bpp, "psnr": psnr, "mse": [m * 255**2 for m in mse],
                "msssim": msssim, "msssim_db": msssim_db
            }

        if args.report_path:
            report_dir = os.path.dirname(args.report_path)
            if report_dir: os.makedirs(report_dir, exist_ok=True)
            report_file = args.report_path + ".csv"
            with open(report_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ["Level", "Channels", 
                          "BPP_Before", "PSNR_Before", "MSE_Before", "MSSSIM_Before", "MSSSIM_DB_Before",
                          "BPP_After", "PSNR_After", "MSE_After", "MSSSIM_After", "MSSSIM_DB_After"]
                writer.writerow(header)
                for i in range(model.num_levels):
                    row = [i, model.switch_list[i]]
                    row.extend([results[model_labels[0]][k][i] for k in ["bpp", "psnr", "mse", "msssim", "msssim_db"]])
                    row.extend([results[model_labels[1]][k][i] for k in ["bpp", "psnr", "mse", "msssim", "msssim_db"]])
                    writer.writerow(row)
            print(f"\n✅ Comparison report saved to {report_file}")

        if args.report_path:
            plt.style.use('seaborn-v0_8-whitegrid')
            
            print("Generating comparison plots for metrics vs. model width...")
            channel_counts = sorted(model.switch_list)
            plot_configs = [
                ("psnr", "Peak Signal-to-Noise Ratio (PSNR)", "PSNR (dB) (Higher is Better ↑)"),
                ("mse", "Mean Squared Error (MSE)", "MSE (Lower is Better ↓)"),
                ("bpp", "Bits Per Pixel (bpp)", "bpp (Lower is Better ↓)"),
                ("msssim", "Multi-Scale SSIM", "MS-SSIM (Higher is Better ↑)"),
                ("msssim_db", "MS-SSIM (dB)", "MS-SSIM (dB) (Higher is Better ↑)"),
            ]
            
            for metric, title, ylabel in plot_configs:
                plt.figure(figsize=(10, 6))
                original_indices = {ch: i for i, ch in enumerate(model.switch_list)}
                sorted_indices = [original_indices[ch] for ch in channel_counts]
                
                before_data = [results[model_labels[0]][metric][i] for i in sorted_indices]
                after_data = [results[model_labels[1]][metric][i] for i in sorted_indices]

                plt.plot(channel_counts, before_data, 'o--', label=model_labels[0])
                plt.plot(channel_counts, after_data, 's-', label=model_labels[1])
                plt.title(title, fontsize=16)
                plt.xlabel("Model Width (# of Filters)", fontsize=12)
                plt.ylabel(ylabel, fontsize=12)
                plt.xticks(channel_counts)
                plt.legend()
                plt.savefig(f"{args.report_path}_{metric}_vs_width.png", dpi=300)
                plt.close()

            print("Generating comparison Rate-Distortion curve plots...")
            
            plt.figure(figsize=(10, 7))
            plt.plot(results[model_labels[0]]['bpp'], results[model_labels[0]]['psnr'], 'o--', label=model_labels[0])
            plt.plot(results[model_labels[1]]['bpp'], results[model_labels[1]]['psnr'], 's-', label=model_labels[1])
            for i, txt in enumerate(model.switch_list):
                y_pos = max(results[model_labels[0]]['psnr'][i], results[model_labels[1]]['psnr'][i])
                x_pos = results[model_labels[1]]['bpp'][i]
                plt.text(x_pos, y_pos * 1.01, f"{txt} filters", fontsize=10, 
                         horizontalalignment='center', verticalalignment='bottom')
            plt.title('Rate-Distortion Curve Comparison (PSNR)', fontsize=16)
            plt.xlabel('Rate (Bits Per Pixel)', fontsize=12)
            plt.ylabel('Quality (PSNR in dB)', fontsize=12)
            plt.grid(True, which='both', linestyle='--')
            plt.legend()
            plt.savefig(f"{args.report_path}_rd_psnr.png", dpi=300)
            plt.close()

            plt.figure(figsize=(10, 7))
            plt.plot(results[model_labels[0]]['bpp'], results[model_labels[0]]['msssim_db'], 'o--', label=model_labels[0])
            plt.plot(results[model_labels[1]]['bpp'], results[model_labels[1]]['msssim_db'], 's-', label=model_labels[1])
            for i, txt in enumerate(model.switch_list):
                y_pos = max(results[model_labels[0]]['msssim_db'][i], results[model_labels[1]]['msssim_db'][i])
                x_pos = results[model_labels[1]]['bpp'][i]
                plt.text(x_pos, y_pos * 1.01, f"{txt} filters", fontsize=10, 
                         horizontalalignment='center', verticalalignment='bottom')
            plt.title('Rate-Distortion Curve Comparison (MS-SSIM dB)', fontsize=16)
            plt.xlabel('Rate (Bits Per Pixel)', fontsize=12)
            plt.ylabel('Quality (MS-SSIM in dB)', fontsize=12)
            plt.grid(True, which='both', linestyle='--')
            plt.legend()
            plt.savefig(f"{args.report_path}_rd_msssim_db.png", dpi=300)
            plt.close()

            print(f"✅ All comparison plots saved to the directory of your report path.")
            
# ==============================================================================
# 6. COMMAND-LINE INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train command ---
    train_parser = subparsers.add_parser("train", help="Train a new model from scratch (Stage 1).")
    train_parser.add_argument("--num_filters", type=int, default=192, help="Max number of filters.")
    train_parser.add_argument("--switch_list", nargs="+", type=int, required=True, help="List of channel counts.")
    train_parser.add_argument("--checkpoint_dir", default="train_torch", help="Directory for model checkpoints.")
    train_parser.add_argument("--train_glob", required=True, help="Path to training images folder.")
    train_parser.add_argument("--batchsize", type=int, default=8)
    train_parser.add_argument("--patchsize", type=int, default=128)
    train_parser.add_argument("--lmbda", nargs="+", type=float, required=True, help="Rate-distortion tradeoffs.")
    train_parser.add_argument("--last_step", type=int, default=1000000, help="Train up to this number of steps.")
    train_parser.add_argument("--preprocess_threads", type=int, default=4)
    train_parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")

    # --- Lambda schedule command ---
    schedule_parser = subparsers.add_parser("train_lambda_schedule", help="Fine-tune with lambda scheduling (Stage 2).")
    schedule_parser.add_argument("--num_filters", type=int, default=192, help="Max number of filters.")
    schedule_parser.add_argument("--switch_list", nargs="+", type=int, required=True, help="List of channel counts.")
    schedule_parser.add_argument("--checkpoint_dir", default="train_torch", help="Directory for model checkpoints.")
    schedule_parser.add_argument("--inputPath", type=str, required=True, help="Path to validation dataset folder.")
    schedule_parser.add_argument("--train_glob", required=True, help="Path to training images folder for fine-tuning steps.")
    schedule_parser.add_argument("--batchsize", type=int, default=8)
    schedule_parser.add_argument("--patchsize", type=int, default=128)
    schedule_parser.add_argument("--lmbda", nargs="+", type=float, required=True, help="Initial rate-distortion tradeoffs.")
    
    # --- Evaluate command ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a single model or compare two models.")
    eval_parser.add_argument("--num_filters", type=int, default=192, help="Max number of filters.")
    eval_parser.add_argument("--switch_list", nargs="+", type=int, required=True, help="List of channel counts.")
    eval_parser.add_argument("--inputPath", type=str, required=True, help="Path to evaluation dataset folder.")
    eval_parser.add_argument(
        "--checkpoint_paths", nargs='+', type=str, required=True, 
        metavar='PATH',
        help="Path to one model checkpoint for a single evaluation, or two paths to compare models."
    )
    eval_parser.add_argument(
        "--report_path", type=str, 
        help="Base path for saving the CSV report and plots (e.g., 'results/report')."
    )
    
    args = parser.parse_args()
    
    if hasattr(args, 'lmbda') and args.lmbda:
        if len(args.switch_list) != len(args.lmbda):
            raise ValueError("The number of switches (--switch_list) must match the number of lambdas (--lmbda).")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.command == "train":
        train_model(args, device)
    elif args.command == "train_lambda_schedule":
        train_lambda_schedule(args, device)
    elif args.command == "evaluate":
        test_model(args, device)

if __name__ == "__main__":
    main()
