#!/bin/bash
#SBATCH --job-name=tr_data
##SBATCH --account=project_2003370
#SBATCH -o out_tr.txt
#SBATCH -e err_tr.txt
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100:1
##SBATCH --array=0-1
##SBATCH  --nodelist=nag10
##module load gcc/8.3.0 cuda/10.1.168
source activate l3_gpu
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wang9/.conda/envs/torch_env/lib/
python create_tr.py --input_path './evaluation_setup/train.csv' --dataset_path '/lustre/wang9/TAU-urban-audio-visual-scenes-2021-development/' --output_path './features_data/' 