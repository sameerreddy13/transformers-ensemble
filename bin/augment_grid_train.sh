if [ $# -eq 0 ]; then
	 echo "Must specify num models"
	 exit 1
fi
SAVE_DIR="logs/11-29/subnet_$1models_lr1e-3_bs32_warmup1000_wd1e-2"
CMD="python ensemble_basic.py --augmented True --extract-subnetwork --num-models $1 --save-dir $SAVE_DIR"
LOG_FILE="$SAVE_DIR/log.txt"
mkdir -p $SAVE_DIR
touch $LOG_FILE && echo "Created $LOG_FILE"
echo "Running in background with nohup: '$CMD'"
nohup $CMD > $LOG_FILE 2>&1 &
exit 0
