#!/bin/bash
#SBATCH --job-name=simclr_train
#SBATCH --output=logs/simclr_%j.out    # Logs output
#SBATCH --error=logs/simclr_%j.err      # Logs errors
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --cpus-per-task=2               # Request 2 CPU cores

# Load your environment (if needed)
source activate cs236781-hw   # Activate conda environment

# Run training
python main.py --self-supervised --contrastive --epochs 50 --batch-size 32 --lr 1e-3 --device cuda

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    STATUS="Training completed successfully!"
else
    STATUS="Training encountered an error!"
fi

# Send email notification
MAIL_RECIPIENT="daniel.pe@campus.technion.ac.il"
echo "$STATUS" | mail -s "Training Job Notification" $MAIL_RECIPIENT