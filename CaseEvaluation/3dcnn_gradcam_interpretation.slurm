#!/bin/bash

#SBATCH -J cnn_gradcam_interpretation
#SBATCH -e /home/alexhernandez/transmembranebindingAI/Notebooks/Cholesterol/3DCNN/JobErrors/myjob.%j.err
#SBATCH -o /home/alexhernandez/transmembranebindingAI/Notebooks/Cholesterol/3DCNN/JobOutput/myjob.%j.out
#SBATCH -t 240:00:00
#SBATCH -p extq
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

source /home/alexhernandez/.bashrc
conda activate cpp-ml
python -u /home/alexhernandez/transmembranebindingAI/Notebooks/Cholesterol/CaseEvaluation/3DInterpretation.py
