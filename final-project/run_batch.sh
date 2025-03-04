#!/bin/bash
#SBATCH --job-name=simclr_train        # Job name
#SBATCH --output=logs/simclr_%j.out    # Output log file
#SBATCH --error=logs/simclr_%j.err      # Error log file
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --cpus-per-task=2               # Request 4 CPU cores
#SBATCH --partition=all                  # Use available partition

echo "Starting SimCLR Training on GTX 1080 Ti"

# Activate Conda environment
source activate cs236781-hw

# Define batch size (use 64 first, fallback to 32 if OOM)
BATCH_SIZE=128

# Try running with batch size 64
python main.py --self-supervised --contrastive --mnist --epochs 20 --batch-size $BATCH_SIZE --lr 1e-3 --device cuda
EXIT_CODE=$?

# If OOM error occurs, reduce batch size to 32 and retry
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "❌ Training failed. Reducing batch size to 32 and retrying..."
    BATCH_SIZE=64
    python main.py --self-supervised --contrastive --mnist --epochs 20 --batch-size $BATCH_SIZE --lr 1e-3 --device cuda
fi

echo "✅ Training Completed!"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    STATUS="Training completed successfully!"
else
    STATUS="Training encountered an error!"
fi

# Send email notification
MAIL_RECIPIENT="daniel.pe@campus.technion.ac.il"
echo "$STATUS" | mail -s "Training Job Notification" $MAIL_RECIPIENT