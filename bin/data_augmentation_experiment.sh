#!/usr/bin/env bash
if [ $# -eq 0 ]; then
    echo "Must specify num models"
    exit 1
fi

DATE="11-29"
LEARNING_RATE="1e-3"
WEIGHT_DECAY=0
WARMUP_STEPS=1000
BATCH_SIZE=32
NUM_EPOCHS=20
#SESSION_NAME="aug_exp_$1"

RUN_ID="subnet_$1models_lr${LEARNING_RATE}_bs${BATCH_SIZE}_warmup${WARMUP_STEPS}_wd${WEIGHT_DECAY}"
SAVE_DIR="logs/${DATE}/${RUN_ID}"
LOG_FILE="${SAVE_DIR}/log.txt"

mkdir -p "${SAVE_DIR}"
touch "${LOG_FILE}" && echo "Created ${LOG_FILE}"

CMD="python ensemble_basic.py --augmented --extract-subnetwork --num-models $1 --lr ${LEARNING_RATE} --wd ${WEIGHT_DECAY} --warmup-steps ${WARMUP_STEPS} --num-epochs ${NUM_EPOCHS} --save-dir ${SAVE_DIR}"
echo "tmux session '${SESSION_NAME}' running command: '$CMD'"

tmux new-session -d -s "${SESSION_NAME}" "${CMD}" | tee "${LOG_FILE}"
nohup "$CMD" > "$LOG_FILE" 2>&1 &
exit 0
