#!/bin/bash
#SBATCH --qos=medium
#SBATCH --job-name=finalState
#SBATCH --output=constDelT_%j.out
#SBATCH --error=constDelT_%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16

module load hpc/2015 anaconda/2.3.0

export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

##################
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

srun --mpi=pmi2 -n $SLURM_NTASKS python finalValues_constantDeltaT.py 0

