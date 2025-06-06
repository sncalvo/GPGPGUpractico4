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
#SBATCH -o salida1.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

nvcc ./ej1.cu -o ej1_sol

./ej1_sol secreto.txt
# nvprof --metrics gld_efficiency,gst_efficiency ./ej1_sol
nvprof ./ej1_sol secreto.txt 64 1
echo 'Fin del programa 1'

nvprof ./ej1_sol secreto.txt 128 1
echo 'Fin del programa 2'

nvprof ./ej1_sol secreto.txt 256 1
echo 'Fin del programa 3'

nvprof ./ej1_sol secreto.txt 512 1
echo 'Fin del programa 4'

nvprof ./ej1_sol secreto.txt 1024 1
echo 'Fin del programa 5'

nvprof ./ej1_sol secreto.txt 2048 1
echo 'Fin del programa 6'


echo '=================================='
echo ''
echo ''
echo ''
nvprof --metrics atomic_transactions ./ej1_sol secreto.txt 64 1
echo 'Fin del programa 1'

nvprof --metrics atomic_transactions ./ej1_sol secreto.txt 128 1
echo 'Fin del programa 2'

nvprof --metrics atomic_transactions ./ej1_sol secreto.txt 256 1
echo 'Fin del programa 3'

nvprof --metrics atomic_transactions ./ej1_sol secreto.txt 512 1
echo 'Fin del programa 4'

nvprof --metrics atomic_transactions ./ej1_sol secreto.txt 1024 1
echo 'Fin del programa 5'

nvprof --metrics atomic_transactions ./ej1_sol secreto.txt 2048 1
echo 'Fin del programa 6'

echo '=================================='
echo ''
echo ''
echo ''
echo 'Fin de set 1'

nvprof ./ej1_sol secreto.txt 64 0
