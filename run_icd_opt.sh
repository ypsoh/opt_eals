#!/bin/bash
module load gcc/7.3
#export OMP_NUM_THREADS=4
export OMP_NUM_THREADS=56

if make icd_main_opt
then
    echo "Good build"
    ./bin/icd_main_opt   --train_data ml-20m/train.csv \
        --test_train_data ml-20m/test_tr.csv \
        --test_test_data ml-20m/test_te.csv \
        --embedding_dim 256 \
        --stddev 0.1 \
        --regularization 0.003 \
        --regularization_exp 1.0 \
        --unobserved_weight 0.1  \
        --epochs 10 --block_size 1 \
        --eval_during_training 0 \
        --print_train_stats 0
else
    echo "Bad build"
    exit 1
fi


numactl --cpunodebind=0 --membind=0 ./bin/icd_main_opt
--train_data ml-20m/train.csv
--test_train_data ml-20m/test_tr.csv
--test_test_data ml-20m/test_te.csv
--embedding_dim 256
--stddev 0.1
--regularization 0.003
--regularization_exp 1.0
--unobserved_weight 0.1 --epochs 10 --block_size 256 --eval_during_training 1
