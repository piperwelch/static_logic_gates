#!/bin/bash
# Specify a partition 
#SBATCH --partition=bluemoon
# Request nodes 
#SBATCH --nodes=1
# Request some processor cores 
#SBATCH --ntasks=20
# Maximum runtime of 2 hours
#SBATCH --time=0:30:00
# Memory per CPU core (optional, if needed)
#SBATCH --mem-per-cpu=4G  # Adjust based on your memory requirements
python3 begin_evolving.py ${1} ${2} ${3} ${4} ${5}