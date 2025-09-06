python SlimCAE_PyTorch.py \
    --num_filters 192 \
    --switch_list 192 144 96 72 48 \
    --checkpoint_dir './checkpoints_torch/' \
    train \
    --train_glob './dataset/train/train/' \
    --patchsize 128 \
    --lmbda 2048 1024 512 256 128 \
    --last_step 1000000

python SlimCAE_PyTorch.py train_lambda_schedule \
    --num_filters 192 \
    --switch_list 192 144 96 72 48 \
    --lmbda 2048 1024 512 256 128 \
    --checkpoint_dir './checkpoints_torch/' \
    --train_glob './dataset/train/train/' \
    --inputPath './dataset/test/test/' \
    --patchsize 128 \
    --batchsize 8

python SlimCAE_PyTorch.py evaluate \
    --num_filters 192 \
    --switch_list 192 144 96 72 48 \
    --inputPath './dataset/test/test/' \
    --checkpoint_paths './checkpoints_torch/final_scheduled_model.pth' \
    --report_path './evaluation_torch/final_report_single'

python SlimCAE_PyTorch.py evaluate \
    --num_filters 192 \
    --switch_list 192 144 96 72 48 \
    --inputPath './dataset/test/test/' \
    --checkpoint_paths './checkpoints_torch/checkpoint_1000000.pth' './checkpoints_torch/final_scheduled_model.pth' \
    --report_path './evaluation_torch/comparison_report'

