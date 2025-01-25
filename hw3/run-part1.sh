#!/bin/bash

#SBATCH --mem=190G


# Setup environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236781-hw

# Make sure conda's libstdc++ is used first
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Print Python version
echo "hello from $(python --version) in $(which python)"

# Run the training script
TRAIN_SCRIPT="Part5_fine_tuning_llm.py"
python "$TRAIN_SCRIPT"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    STATUS="Training completed successfully!"
else
    STATUS="Training encountered an error!"
fi

# Send email notification
MAIL_RECIPIENT="daniel.pe@campus.technion.ac.il"
echo "$STATUS" | mail -s "Training Job Notification" $MAIL_RECIPIENT
