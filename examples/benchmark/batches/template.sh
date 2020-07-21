#! /bin/bash
#PBS -N {jobname}
#PBS -q default
#PBS -l walltime=03:00:00
#PBS -l {nodes}
#PBS -m n
#PBS -o {jobname}.out
#PBS -e {jobname}.err

basedir=$(pwd)

cd {path}
source env_benchmark.sh
source env_numpy.sh

{cmd} &> logs/{jobname}.log

cd ${{basedir}}

