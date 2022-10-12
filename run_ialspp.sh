#!/bin/bash

./bin/ialspp_main   --train_data ml-20m/train.csv \
                    --test_train_data ml-20m/test_tr.csv \
                    --test_test_data ml-20m/test_te.csv \
                    --embedding_dim 256 \
                    --stddev 0.1 \
                    --regularization 0.003 \
                    --regularization_exp 1.0 \
                    --unobserved_weight 0.1  \
                    --epochs 10 --block_size 64 --eval_during_training 0
