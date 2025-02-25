#!/bin/bash

# Dock COVID Moonshot compounds in parallel

#BSUB -W 12:00
#BSUB -R "rusage[mem=3]"
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -q gpuqueue
#BSUB -gpu "num=1:mode=shared:mps=no:j_exclusive=yes"
#BSUB -m "lt-gpu ls-gpu lu-gpu lp-gpu ld-gpu"
#BSUB -o %J.moonshot-simulate.out
#BSUB -J "moonshot-simulate[1-4685]"

echo "Job $JOBID/$NJOBS"

echo "LSB_HOSTS: $LSB_HOSTS"

source ~/.bashrc

conda activate perses

export PREFIX="covid_submissions_all_info"

let JOBID=$LSB_JOBINDEX-1
echo $JOBID

set -x
python ../scripts/02-dock-and-prep.py --receptors ../receptors/monomer --molecules $PREFIX.csv --index $JOBID --output $PREFIX-docked --simulate --fahprep --core22

# To transfer to Temple
# rsync -avz --include="*/" --include="RUN*/*.gro" --include="RUN*/*.top" --exclude="*" covid_submissions_all_info-docked tug27224@owlsnest.hpc.temple.edu:work
