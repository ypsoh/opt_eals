#!/bin/bash
module load gcc/7.3

# ./run_eals.sh 
OMP_NUM_THREADS=56
export OMP_NUM_THREADS=$OMP_NUM_THREADS

function get_dataset() {
    declare -A path_to_dataset=(
        ["ml"]="ml-20m"
        ["ms"]="msd"
    )
    echo "${path_to_dataset[${1}]}"
}

function get_variation() {
    declare -A variation=(
        ["icd"]="icd_main"
        ["icd_opt"]="icd_main_opt"
        ["ials"]="ials_main"
        ["ialspp"]="ialspp_main"
    )
    echo "${variation[${1}]}"
}

dataset="$(get_dataset $2)"
train_data="--train_data ${dataset}/train.csv"
test_train_data="--test_train_data ${dataset}/test_tr.csv"
test_test_data="--test_test_data ${dataset}/test_te.csv"
embedding_dim="--embedding_dim 256"
stddev="--stddev 0.1"
regularization="--regularization 0.003"
regularization_exp="--regularization_exp 1.0"
unobserved_weight="--unobserved_weight 0.1"
epochs="--epochs 10"
block_size="--block_size 256"
eval_during_training="--eval_during_training 1"


options="${train_data} ${test_train_data} ${test_test_data} ${embedding_dim} ${stddev} ${regularization} ${regularization_exp} ${unobserved_weight} ${epochs} ${block_size} ${eval_during_training}"

if make $(get_variation $1)
then
    echo "Build successful..."
    cmd="./bin/$(get_variation $1)"
    echo "${cmd} ${options}"
    eval "${cmd} ${options}"
else
    echo "Build falied..."
    exit 1
fi
