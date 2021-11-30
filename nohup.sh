if [ $# -eq 0 ]; then
	echo "Must specify num models"
	exit 1
fi
SAVE_DIR="logs/11-29/subnet_$1models_lr1e-3_bs8_warmup1000"
CMD="python ensemble_basic.py --extract-subnetwork --num-models $1 --batch-size 8 -wd 0. --lr 1e-3 --num-epochs 200 --warmup 1000 --save-dir $SAVE_DIR"
LOG_FILE="$SAVE_DIR/log.txt"
mkdir -p $SAVE_DIR
touch $LOG_FILE && echo "Created $LOG_FILE"
echo "Running in background with nohup: '$CMD'"
nohup $CMD > $LOG_FILE 2>&1 &
exit 0