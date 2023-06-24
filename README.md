# Transformers, Ensemble
Class Project for CS-8803-SMR
Glenn Matlin (GT ID: X) , Piyush Patil (GT ID: Y), Sameer Reddy (GT ID: Z)

[Presentation](https://gtvault-my.sharepoint.com/:p:/g/personal/gmatlin3_gatech_edu/EaTUNYc6_dpJsZeyEE1q6wwBSzWMubU_9OyjctwuiNVadA?e=AHFhQv)

## Setup

```
- git clone repo
- install miniconda3
- install conda env using environment.yml
```

```
If using GPU:
	- pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```


## Script for training ensemble models
```
python ensemble_basic.py. Use `-h` to see help for running the script.
```

## Script for combining trained ensemble models, with a voting method
```
python ensemble_checkpoints.py
```

## Instructions for running examples
### First train subnetworks
- `examples/train_8_subnetworks.sh` to start training
- Provide any number of integers as GPU numbers to use e.g. `bash examples/train_8_subnetworks.sh 1 3 7` to train in parallel on GPUs 1, 3, 7  

### Second ensemble subnetworks with voting
- `examples/ensemble_subnetworks.sh` to train ensemble method and evaluate
- Run `bash examples/ensemble_subnetworks.sh [device]` where device is one of 'cpu' 'cuda' or 'cuda:n' 
