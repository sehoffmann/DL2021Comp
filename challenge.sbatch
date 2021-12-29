#!/bin/bash

####
#a) Define slurm job parameters
####

# the #SBATCH are no comments. The #n is necessary.

#SBATCH --job-name=DNNChallenge

#resources:

#SBATCH --cpus-per-task=10
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=17:00:000
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=NONE
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=sh@uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder. We provided the challenge data already at 
#/common/datasets/DLChallenge2021/*
####

cp -r ~/DL2021Comp /scratch/$SLURM_JOB_ID/

####
#c) Execute your pytorch code in a specific singularity container
# TODO adapt the path to your train.py script.
# TODO adapt the path to your predicted_test_label_save_dir
####

cd /scratch/$SLURM_JOB_ID/DL2021Comp
singularity exec --nv /common/singularityImages/default.simg ./tcml-agent.sh $1

echo DONE!

