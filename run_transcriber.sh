#!/bin/bash
#SBATCH --job-name=xcy_fullexp
#SBATCH --account=def-arashmoh_gpu
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G

#SBATCH --output=/home/gkianfar/scratch/Amin/Voice/outputs/logs/%x_%j.out
#SBATCH --error=/home/gkianfar/scratch/Amin/Voice/outputs/logs/%x_%j.err

# ============================================================
# Voice2Text - Faster-Whisper transcription
# ============================================================

# Create output directories if they do not exist
mkdir -p /home/gkianfar/scratch/Amin/Voice/outputs/logs
mkdir -p /home/gkianfar/scratch/Amin/Voice/outputs

# Project paths
PROJECT_DIR="/home/gkianfar/scratch/Amin/Voice/Voice2Text"
SRC_DIR="${PROJECT_DIR}/src"
OUTPUT_DIR="/home/gkianfar/scratch/Amin/Voice/outputs"

# Activate virtual environment
source /home/gkianfar/scratch/Amin/Voice/Voicevenv/bin/activate

# Move to project source directory
cd "${SRC_DIR}" || exit 1

# Print job information
echo "============================================================"
echo "Voice2Text SLURM Job"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Job name:     ${SLURM_JOB_NAME}"
echo "Node:         ${SLURMD_NODENAME}"
echo "Project:      ${PROJECT_DIR}"
echo "Source:       ${SRC_DIR}"
echo "Output:       ${OUTPUT_DIR}"
echo "Python:       $(which python)"
echo "Python ver.:  $(python --version)"
echo "GPU:"
nvidia-smi
echo "============================================================"

# Run the main transcription program
python transcriber.py

EXIT_CODE=$?

echo "============================================================"
echo "Transcription finished"
echo "Exit code: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
