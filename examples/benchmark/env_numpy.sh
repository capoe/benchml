#! /bin/bash
n_cores=4
export OMP_NUM_THREADS=${n_cores}
export MKL_NUM_THREADS=${n_cores}
export NUMEXPR_NUM_THREADS=${n_cores}
