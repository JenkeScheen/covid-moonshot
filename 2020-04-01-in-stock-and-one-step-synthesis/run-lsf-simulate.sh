#!/bin/bash

# Dock COVID Moonshot compounds in parallel

#BSUB -W 0:30
#BSUB -R "rusage[mem=2]"
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -q gpuqueue
#BSUB -gpu "num=1:mode=shared:mps=no:j_exclusive=yes"
#BSUB -m "lt-gpu ls-gpu lu-gpu lp-gpu ld-gpu"
#BSUB -o %J.moonshot-simulate.out
#BSUB -J "moonshot-dock[1-172]"

echo "Job $JOBID/$NJOBS"

echo "LSB_HOSTS: $LSB_HOSTS"

source ~/.bashrc

source activate perses

let JOBID=$LSB_JOBINDEX-1
python ../scripts/02-dock-and-prep.py --molecules merged-compounds-2020-04-01.csv --index $JOBID --output merged-compounds-2020-04-01.csv
