#!/bin/bash
#SBATCH --job-name=yolo_project_get_data
#SBATCH --output=slurm_logs/yolo_project_get_data-%j.out
#SBATCH --error=slurm_logs/yolo_project_get_data-%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

# --- Config ---

unset VIRTUAL_ENV
export PATH="$HOME/.local/bin:$PATH" 
UV_PROJECT=$HOME/slurm/yolo-project/get_data

# NOTEBOOK_PATH="$HOME/Yolo_project/test_CNN_pytorch-pretrained_backbone.ipynb"
# OUTPUT_NOTEBOOK="$HOME/Yolo_project/test_CNN_pytorch-pretrained_backbone_out-${SLURM_JOB_ID}.ipynb"

FORMAT_SCRIPT="$HOME/Yolo_project/Format_dataset.py"

echo "=== Informations SLURM ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Noeud : $SLURM_NODELIST"
echo "GPU(s) alloué(s) : $CUDA_VISIBLE_DEVICES"
echo "--------------------------"

# --- Initialisation uv ---
echo "=== Préparation de l'environnement uv ==="

# Crée un répertoire de travail isolé pour ce job
mkdir -p "$UV_PROJECT"
cd "$UV_PROJECT"

# Initialise le projet s'il n'existe pas encore
if [ ! -f "pyproject.toml" ]; then
  uv init --quiet
  uv venv
fi

uv pip install -r /home/lcoussy/Yolo_project/slurm/requirements.txt
uv pip install kaggle
# uv pip install torch papermill jupyter

export TRANSFORMERS_VERBOSITY=info

# uv run python -m ipykernel install --user --name=yolo_kernel
# uv run papermill "$NOTEBOOK_PATH" "$OUTPUT_NOTEBOOK" -k yolo_kernel

mkdir -p "$(dirname "$FORMAT_SCRIPT")"

uv run python "$FORMAT_SCRIPT"

echo "=== Script exécuté ==="
# echo "Résultat enregistré dans : $OUTPUT_NOTEBOOK"