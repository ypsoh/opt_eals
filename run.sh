#!/bin/bash
# Helper functions
echoerr() {
    echo -e "$@" 1>&2
}

print_help() {
    echo    "Usage: ./run.sh --model ials_main --dataset ml-20m --nthreads 56 --embedding 256 --epochs 1 --blocksize 16"
}

# write_log STRING
write_log() {
    INDENT=`printf '=%.0s' {1..20}`
    echo -n "$(date '+%Y-%m-%d_%H:%M:%S'):   " >> $LOGF
    echo -e $1 >> $LOGF
}

# Usage: write_logf FILEPATH
write_logf () {
    INDENT=`printf ' %.0s' {1..23}`
    echo -n "$(date '+%Y-%m-%d_%H:%M:%S'):   " >> $LOGF
    head -n 1 $1 >> $LOGF
    cat $1 | awk -v indent="$INDENT" 'NR > 1 {print indent $0}' >> $LOGF
}

BASE_DIR="$(dirname "$(realpath $0)")"

DATETAG="$(date '+%Y-%m-%d_%H-%M-%S')"
LOGF="${BASE_DIR}/results_${DATETAG}.log"

# Parse parameters
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -m|--model)
        MODEL="$2"
        shift
        shift
        ;;
    -d|--dataset)
        DATASET="$2"
        shift
        shift
        ;;
    -t|--nthreads)
        NUM_THREADS="$2"
        shift
        shift
        ;;
    -f|--embedding)
        EMB_DIMENSION="$2"
        shift
        shift
        ;;
    -e|--epochs)
        EPOCHS="$2"
        shift
        shift
        ;;
    -b|--blocksize)
        BLOCK_SIZE="$2"
        shift
        shift
        ;;
    *) # unknown option
        shift;;
esac
done

export OMP_NUM_THREADS=$NUM_THREADS

if [[ -z "$MODEL" || -z "$DATASET" || -z "$NUM_THREADS" || -z "$EMB_DIMENSION" || -z "$EPOCHS" ]]; then
    echoerr "Insufficient parameters"
    print_help
    exit 0
fi

TMPLOG=".tmp.${DATETAG}.log"

if make $MODEL
then
    echo "Good build"
    write_log "START==========================="
    write_log "Run $MODEL, store results in $LOGF"
    ./bin/$MODEL   --train_data $DATASET/train.csv \
        --test_train_data $DATASET/test_tr.csv \
        --test_test_data $DATASET/test_te.csv \
        --embedding_dim $EMB_DIMENSION \
        --stddev 0.1 \
        --regularization 0.003 \
        --regularization_exp 1.0 \
        --unobserved_weight 0.1  \
        --block_size $BLOCK_SIZE \
        --epochs $EPOCHS --eval_during_training 1 > $TMPLOG 2>&1
    write_logf $TMPLOG
    # cat $TMPLOG >> $LOGF
else
    echo "Bad build"
    exit 1
fi

rm $TMPLOG
write_log "FINISH==========================="