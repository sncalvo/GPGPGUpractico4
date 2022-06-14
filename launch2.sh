#!/bin/bash
#SBATCH --job-name=mitrabajo
#SBATCH --ntasks=1
#SBATCH --mem=512
#SBATCH --time=00:08:00

#SBATCH --partition=besteffort

#SBATCH --qos=besteffort_gpu

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sncalvo5@gmail.com
#SBATCH -o salida2.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

nvcc ./ej2.cu -o ej2_sol

nvprof ./ej2_sol 4096 0
nvprof ./ej2_sol 8192 0
echo '========================='
echo '========================='
nvprof ./ej2_sol 4096 1
nvprof ./ej2_sol 8192 1
echo '========================='
echo '========================='
echo '========================='
echo '========================='
echo 'End time metrics'
nvprof --metrics gld_efficiency,gst_efficiency,shared_efficiency ./ej2_sol 4096 0
echo 'End eff metric 1'
nvprof --metrics gld_efficiency,gst_efficiency,shared_efficiency ./ej2_sol 8192 0
echo 'End eff metric 2'
echo '========================='
echo '========================='
echo 'End time metrics'
nvprof --metrics gld_efficiency,gst_efficiency,shared_efficiency ./ej2_sol 4096 1
echo 'End eff metric 1'
nvprof --metrics gld_efficiency,gst_efficiency,shared_efficiency ./ej2_sol 8192 1
echo 'End eff metric 2'
