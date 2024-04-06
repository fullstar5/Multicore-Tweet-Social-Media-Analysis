#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-12:00:00
module load mpi4py/3.1.4
module load Python/3.10.4
time mpiexec -n 1 python3 A1_optimization3.py
my-job-stats -a -n -s