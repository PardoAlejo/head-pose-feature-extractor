#!/bin/bash
#SBATCH --job-name PoseFace
#SBATCH --array=1-50
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --cpus-per-task=9
#SBATCH --mem 96GB
##SBATCH --mail-type=ALL
##SBATCH --mail-user=alejandro.pardo@kaust.edu.sa
##SBATCH --constraint=[v100]
##SBATCH -A conf-gpu-2020.11.23

echo `hostname`
# conda activate refineloc
# module load anaconda3
source activate facenet

# DIR=$HOME/facenet_feature_extractor
# cd $DIR
echo `pwd`

# python code/inference.py --snapshot models/hopenet_alpha1.pkl --batch_size=256 --gpu_number=1 --path_video_csv data/video_paths_dummy.csv 
python code/inference.py --snapshot models/hopenet_alpha1.pkl --batch_size=256 --gpu_number=0 --path_video_csv data/video_paths.csv 
