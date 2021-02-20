#!/bin/bash
#SBATCH --job-name=av
##SBATCH --account=project_2003370
#SBATCH -o out_av.txt
#SBATCH -e err_av.txt
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100:1
##SBATCH --array=0-1
##SBATCH  --nodelist=r14g06
##module load gcc/8.3.0 cuda/10.1.168
source activate torch_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wang9/.conda/envs/torch_env/lib/
#python train.py --features_path '../create_data/features_data/' --model_type 'audio'
#python train.py --features_path '../create_data/features_data/' --model_type 'video'
python train.py --features_path '../create_data/features_data/' --model_type 'audio_video'
