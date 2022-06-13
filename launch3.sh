#!/bin/bash
#SBATCH --job-name=mitrabajo
#SBATCH --ntasks=1
#SBATCH --mem=512
#SBATCH --time=00:01:00

#SBATCH --partition=besteffort

#SBATCH --qos=besteffort_gpu

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sncalvo5@gmail.com
#SBATCH -o salida3.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

nvcc ./ej3.cu -o ej3_sol

./ej3_sol
# nvprof --metrics gld_efficiency,gst_efficiency ./ej3_sol
# nvprof ./ej3_sol
