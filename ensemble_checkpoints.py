import torch
import argparse
from pathlib import Path
import transformers
import utils
import datasets
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", help="Path to set of models for one experiment", required=True)
    ap.add_argument("--dataset", default="sst2")
    ap.add_argument("--val-batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()

def get_epoch_metrics(model_dir, key):
    p = (Path('.') / model_dir).glob('*.pt')
    get_epoch_num = lambda x: int(x.name.split('epoch')[-1][:-3])
    return [(x, torch.load(x, map_location='cpu')[key]) for x in sorted(p, key=get_epoch_num)]

def average_vote(models, input_ids, attention_mask, labels):
    '''
    Return ensmble predictions and accuracy
    '''
    all_preds = []
    for model in models:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        preds = outputs.logits.argmax(axis=-1)
        all_preds.append(preds)

    preds = torch.stack(all_preds).mode(dim=0).values
    return preds, (preds == labels).float().mean()


def main(args):
    p = Path(args.exp_dir)
    assert p.is_dir(), "Experiment directory not found"
    device = args.device
    # Get paths to ensemble models
    model_dirs = [x for x in p.iterdir() if x.is_dir()]
    n_models = len(model_dirs)
    is_naive = 'naive' in str(p.resolve())
    models = []
    accuracies = []
    # Choose checkpoints for ensemble
    for dir in model_dirs:
        val_accs = get_epoch_metrics(dir, 'val_acc')
        chk_path, val_acc = max(val_accs, key=lambda x: x[1])
        print(chk_path, val_acc)
        models.append(utils.load_model_checkpoint(chk_path, naive=is_naive))
        accuracies.append(val_acc)
    print(f"Mean accuracy = {sum(accuracies) /  len(accuracies)}")

    # Load validation data
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    ds = datasets.load_dataset("glue", args.dataset)
    val_dataloader = utils.create_dataloader(
        ds["validation"], tokenizer, args.val_batch_size, args.dataset
    )
    # Eval
    [model.eval() for model in models]
    models = [model.to(device) for model in models] # TODO: split up across gpus
    accs = []
    for example in val_dataloader:
        labels = example[2].to(device)
        _, acc = average_vote(
            models, 
            example[0].to(device), example[1].to(device), 
            labels
        )
        accs.append(acc)
    print("Ensemble voting validation accuracy:", sum(accs) / len(accs))

if __name__ == '__main__':
    main(parse_args())