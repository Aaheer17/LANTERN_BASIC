#!/bin/bash
##SBATCH -N 1
#SBATCH --cpus-per-task=8  
#SBATCH -n 1
#SBATCH --job-name=D2_Diff
##SBATCH --exclusive
#SBATCH -t 72:00:00
#SBATCH --mem=150000
#SBATCH -p bii-gpu
#SBATCH --gres=gpu:1
#SBATCH -A bii_nssac
module load miniforge
module load texlive
source activate torch_gpu_renew
# export RANK=0
# export WORLD_SIZE=4
# export MASTER_ADDR="localhost"
# export MASTER_PORT=12345

export PATH=~/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
which dvipng
python3 main.py ./configs/d2_shape_model_diffusion_p3_sigmoid.yaml --use_cuda
#srun python3 main_multiGPU.py /project/biocomplexity/fa7sa/calo_dreamer/configs/d2_shape_model_Autodiffusion.yaml --use_cuda
#python3 main.py /project/biocomplexity/fa7sa/calo_dreamer/configs/d2_energy_model_submission.yaml --use_cuda
#python3 cuda_mem_test.py
