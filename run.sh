#!/bin/bash
#SBATCH -p G1Part_sce
#SBATCH -N 1
#SBATCH -n 56
#SBATCH -c 56
#SBATCH -o log-stack

source /es01/paratera/parasoft/module.sh
source /es01/paratera/sce0228/.bashrc
module load anaconda/3
source activate sce0228

python -u fits_stack.py $1 $2 $3
