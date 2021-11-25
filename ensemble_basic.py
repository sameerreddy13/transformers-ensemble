import argparse
import concurrent.futures
import os
import random

import datasets
import torch
import transformers

import utils

# Number of parameters in the original pretrained BERT architecture.
BERT_N_PARAMS = 109483778


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--save-dir", type=str, default="checkpoints")
    ap.add_argument("--gpus", nargs="+", default=list(range(8)))
    ap.add_argument("--num-models", type=int, default=8)
    ap.add_argument("--dataset", type=str, default="sst2")
    ap.add_argument("--num-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=-1)

    return ap.parse_args()


def train_wrapper(kwargs):
    """
    A useful wrapper to use when parallelizing the train() function.
    """
    return train(**kwargs)

def train(
    task_id,
    model,
    train_dataloader,
    val_dataloader,
    device,
    save_dir,
    lr=1e-5,
    num_epochs=100,
    print_freq=50,
):
    prefix = f"[Thread {task_id}]"
    os.makedirs(save_dir, exist_ok=True)
    print(f"{prefix} Created {save_dir}")

    model = model.to(device, non_blocking=True)
    print(f"{prefix} Moved model to device {device}")

    metrics = {}
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(num_epochs):
        for i, (input_ids, attention_mask, labels) in enumerate(train_dataloader):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_freq == 0:
                print(f"{prefix} [Epoch {epoch}] Step {i + 1} of {len(train_dataloader)}: "
                      f"loss = {loss.item()}")

        metrics["train_acc"] = utils.compute_acc(model, train_dataloader, device=device)
        metrics["val_acc"] = utils.compute_acc(model, val_dataloader, device=device)
        print(f"{prefix} Train accuracy: {metrics['train_acc']}")
        print(f"{prefix} Validation accuracy: {metrics['val_acc']}")

        save_path = os.path.join(save_dir, f"model_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
            "train_acc": metrics["train_acc"],
            "val_acc": metrics["val_acc"]
        }, save_path)
        print(f"{prefix} Saved model checkpoint to {save_path}")

    return metrics


def main(args):
    print(f"Save dir: {args.save_dir}")

    if args.gpus is None or len(args.gpus) == 0:
        print("WARNING: Using CPU")
        gpus = ["cpu"]
    else:
        gpus = [f"cuda:{i}" for i in args.gpus]
        print(f"Using GPUs: {', '.join(gpus)}")

    # Set up data loader.
    print(f"Building dataloaders for dataset: {args.dataset}")
    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v2')
    ds = datasets.load_dataset("glue", args.dataset)

    train_ds = list(ds["train"])[:args.limit]
    random.shuffle(train_ds)
    partition_size = len(train_ds) // args.num_models + 1
    train_dataloaders = [
        utils.create_dataloader(
            train_ds[i : i + partition_size], tokenizer, args.batch_size, args.dataset)
        for i in range(0, len(train_ds), partition_size)
    ]
    print(f"Partitioned {len(train_ds)} total training samples")
    val_dataloader = utils.create_dataloader(
        ds['validation'], tokenizer, args.val_batch_size, args.dataset)

    # Build models.
    print("Building models")
    config = transformers.AlbertConfig(
        embedding_size=128,
        # TODO(piyush) Don't hard-code (this is for 8 models).
        hidden_size=int(4096 * 7/32),
        intermediate_size=int(16384 * 3/16),
    )
    models = [
        transformers.AlbertForSequenceClassification(config)
        for _ in range(args.num_models)
    ]

    # Preserve the same total parameter count as original BERT, within a 10% margin.
    n_params = sum([param.numel() for param in models[0].parameters()])
    print(f"Created {args.num_models} models, each with {n_params / 1e6} million parameters")
    # assert 1 / 1.1 <= (args.num_models * n_params) / BERT_N_PARAMS <= 1.1 # TODO(piyush) uncomment

    # Train.
    pool = torch.multiprocessing.Pool(args.num_models)
    metrics = pool.map(train_wrapper, [
        {
            "task_id": i,
            "model": models[i],
            "train_dataloader": train_dataloaders[i],
            "val_dataloader": val_dataloader,
            "device": gpus[i % len(gpus)],
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "save_dir": os.path.join(args.save_dir, str(i)),
        }
        for i in range(args.num_models)
    ])

if __name__ == "__main__":
    main(parse_args())
