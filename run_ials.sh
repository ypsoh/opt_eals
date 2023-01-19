#!/bin/bash
module load gcc/7.3

# Compact params
num_threads="56"
embedding_size="256"
epochs="1"
models="ials_main ialspp_main"
datasets="ml-20m"

# Extended params
# num_threads="1 4 8 14 28 56"
# embedding_size="64 128 256 512 1024 2048"
# epochs="1 4 8"
# models="ials_main ialspp_main icd_main icd_merged_main"
# datasets="ml-20m msd"

# We don't variate block size yet -- this only affects iALSpp
for dataset in $datasets; do
    for num_thread in $num_threads; do
        for emb in $embedding_size; do
            for model in $models; do
                for epoch in $epochs; do
                    ./run.sh -m $model -d $dataset -t $num_thread -f $emb -e $epoch -b 64 2>&1
                done
            done
        done
    done
done