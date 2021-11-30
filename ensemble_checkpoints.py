import torch
import argparse
from pathlib import Path
import transformers
import utils
import datasets
import model_ensemble
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", help="Path to set of models for one experiment", required=True)
    ap.add_argument("--dataset", type=str, default="sst2")
    ap.add_argument("--val-batch-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=str, default=32) # TODO: Set to 1 for now, since we only support average voting which doesnt require training dataq
    ap.add_argument("--average-vote", action="store_true", default=True)
    return ap.parse_args()

def get_epoch_num(checkpoint_path):
    '''
    Helper to get epoch number as int from checkpoint path string
    '''
    return int(checkpoint_path.name.split('epoch')[-1][:-3])

def get_epoch_metrics(model_dir, key):
    '''
    Get metrics per epoch by given key
    '''
    p = (Path('.') / model_dir).glob('*.pt')
    return [(x, torch.load(x, map_location='cpu')[key]) for x in sorted(p, key=get_epoch_num)]

def get_last_epoch(model_dir):
    '''
    Get last epoch as proxy for best val accuracy, since we only save when val accuracy passes best so far
    '''
    p = (Path('.') / model_dir).glob('*.pt')
    last_epoch = max(p, key=get_epoch_num)
    return torch.load(last_epoch, map_location='cpu')

def main(args):
    p = Path(args.exp_dir)
    assert p.is_dir(), "Experiment directory not found"
    device = args.device
    model_dirs = [x for x in p.iterdir() if x.is_dir()]
    n_models = len(model_dirs)
    is_naive = 'naive' in str(p.resolve())
    models = []
    indiv_accs = []

    # Choose checkpoints for ensemble
    for i, dir in enumerate(model_dirs):
        print(f"Getting best checkpoint for model {i}", end='\r')
        checkpoint = get_last_epoch(dir)
        indiv_accs.append(checkpoint['val_acc'])
        models.append(utils.load_model_checkpoint(checkpoint, naive=is_naive))

    # Load validation data
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    ds = datasets.load_dataset("glue", args.dataset)
    train_dataloader = utils.create_dataloader(
        list(ds["train"])[:args.limit], tokenizer, args.batch_size, args.dataset
    )
    val_dataloader = utils.create_dataloader(
        ds["validation"], tokenizer, args.val_batch_size, args.dataset
    )
    print(
        "Train size =", args.limit,
        ", Val size =", ds["validation"].num_rows,
        ", Test size =", ds['test'].num_rows
    )
    print(f"Mean val accuracy (individual) = {sum(indiv_accs) /  len(indiv_accs)}")
    # Fit ensemble
    [model.eval() for model in models]
    if args.average_vote:
        ensemble = model_ensemble.AverageVote(models, device)
    else:
        raise ValueError("No ensemble strategy provided")
    ensemble.fit(train_dataloader)
    # Eval ensemble
    _, val_accs = ensemble.predict(val_dataloader)
    _, train_accs = ensemble.predict(train_dataloader)
    get_acc = lambda x: round((sum(x) / len(x)).item(), 4)
    print("Ensemble voting train accuracy:", get_acc(train_accs))
    print("Ensemble voting val accuracy:", get_acc(val_accs))

if __name__ == '__main__':
    main(parse_args())