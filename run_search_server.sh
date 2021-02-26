#!/bin/bash
#PBS -N WikiIndex
#PBS -l select=1:ncpus=1:mem=32gb:scratch_local=80gb
#PBS -l walltime=16:00:00
#
####PBS -l select=1:ncpus=48:mem=100gb:scratch_local=220gb
####PBS -l walltime=24:00:00


module add python-3.6.2-gcc
export PYTHONPATH=/storage/brno2/home/xslavk01/.local/lib/python3.6/site-packages/

DATASET=wiki
REPO=/storage/brno2/home/xslavk01/workspace/dp/retrieval_bm25
DATA=$REPO/data/indexes

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> jobs_info.txt
# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

echo "Copying $DATA/wiki-en.index/ to $SCRATCHDIR" >> jobs_info.txt
cp -r $DATA/wiki-en.index/  $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }
echo "Copying $DATA/wiki-de.index/ to $SCRATCHDIR" >> jobs_info.txt
cp -r $DATA/wiki-de.index/  $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }
echo "Copying $DATA/wiki-es.index/ to $SCRATCHDIR" >> jobs_info.txt
cp -r $DATA/wiki-es.index/  $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }

cd $REPO

python server.py -n -i $SCRATCHDIR >> jobs_info.txt
python server.py -w 500 -i $SCRATCHDIR
clean_scratch
