#!/bin/bash
#PBS -N Searcher
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=06:00:00

#
####PBS -l select=1:ncpus=48:mem=100gb:scratch_local=220gb
####PBS -l walltime=24:00:00


module add python-3.6.2-gcc
export PYTHONPATH=/storage/brno2/home/xslavk01/.local/lib/python3.6/site-packages/

DATASET=wiki
REPO=/storage/brno2/home/xslavk01/workspace/dp/retrieval_bm25

cd $REPO

echo "$PBS_JOBID is running on node `hostname -f`" >> jobs_info.txt
echo "Dataset: $DATASET" >> jobs_info.txt

python server.py --dry-run >> jobs_info.txt
python server.py
