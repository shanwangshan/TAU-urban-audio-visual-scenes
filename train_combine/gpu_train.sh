#!/bin/bash
#SBATCH --job-name=train
##SBATCH --account=project_2003370
#SBATCH -o out.txt
#SBATCH -e err.txt
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
python train.py --model_audio_path '../train/audio_model/model.pt' --model_video_path '../train/video_model/model.pt' --features_path '../create_data/features_data/' --output_dir './audio_video_model/' --n_epoch 100
