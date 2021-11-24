import argparse
import concurrent.futures
import os
import random
import time

import datasets
import torch
import transformers
from tqdm import tqdm

# Number of parameters in the original pretrained BERT architecture.
BERT_N_PARAMS = 109483778


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--save-dir", type=str, default="checkpoints")
    ap.add_argument("--gpus", nargs="+", default=list(range(8)))
    ap.add_argument("--num-models", type=int, default=10)
    ap.add_argument("--dataset", type=str, default="sst2")
    ap.add_argument("--num-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--limit", type=int, default=-1)

    return ap.parse_args()


@torch.no_grad()
def compute_acc(model, dataloader, device):
    accs = []
    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        accs.append((logits.argmax(axis=-1) == labels).float().mean())
    return sum(accs) / len(accs)


def create_dataloader(dataset, tokenizer, batch_size):
    # encodings = tokenizer(
    #     [example["premise"] for example in dataset], [example["hypothesis"] for example in dataset],
    #      max_length=128, add_special_tokens=True, padding=True, truncation=True,
    #     return_tensors='pt')
    encodings = tokenizer([example['sentence'] for example in dataset],
         max_length=128, add_special_tokens=True, padding=True, truncation=True,
        return_tensors='pt')
    labels = torch.tensor([example["label"] for example in dataset])
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels),
        batch_size=batch_size)


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)
    for epoch in range(num_epochs):
        tepoch = tqdm(train_dataloader, unit="batch")
        for i, (input_ids, attention_mask, labels) in enumerate(tepoch):
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

        metrics["train_acc"] = compute_acc(model, train_dataloader, device=device)
        metrics["val_acc"] = compute_acc(model, val_dataloader, device=device)
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
    save_dir = f"{args.save_dir}_{int(time.time())}"
    print(f"Save dir: {save_dir}")

    if args.gpus is None or len(args.gpus) == 0:
        print("WARNING: Using CPU")
        gpus = ["cpu"]
    else:
        gpus = [f"cuda:{i}" for i in args.gpus]
        print(f"Using GPUs: {', '.join(gpus)}")

    # Set up data loader.
    print("Building dataloaders")
    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v2')
    ds = datasets.load_dataset("glue", args.dataset) #TODO: Check if this needs to be set as "sst2"

    train_ds = list(ds["train"])[:args.limit]
    random.shuffle(train_ds)
    partition_size = len(train_ds) // args.num_models + 1
    train_dataloaders = [
        create_dataloader(train_ds[i : i + partition_size], tokenizer, args.batch_size)
        for i in range(0, len(train_ds), partition_size)
    ]
    print(f"Partitioned {len(train_ds)} total training samples")
    val_dataloader = create_dataloader(ds['validation'], tokenizer, args.val_batch_size)
    # Build models.
    print("Building models")
    config = transformers.AlbertConfig(
        embedding_size=128,          # 128
        # hidden_size=int(4096 * 3/16),
        hidden_size=int(4096 * 7/32), # TODO(piyush) remove (for 8 models)
        intermediate_size=int(16384 * 3/16),
        initializer_range=0.05
    )
    models = [
        transformers.AlbertForSequenceClassification(config)
        for _ in range(args.num_models)
    ]
    # Preserve the same total parameter count as original BERT, within a 10% margin.
    n_params = sum([param.numel() for param in models[0].parameters()])
    print(f"Created {args.num_models} models, each with {n_params / 1e6} million parameters")
    # assert 1 / 1.1 <= (args.num_models * n_params) / BERT_N_PARAMS <= 1.1
    model = transformers.AlbertForSequenceClassification.from_pretrained('albert-base-v2')
    results = train(
                    task_id=0, model=model, 
                    train_dataloader=train_dataloaders[0], val_dataloader=val_dataloader, 
                    device=gpus[0], lr=args.lr, num_epochs=args.num_epochs, save_dir=os.path.join(save_dir, str(0))
                )
    print(f"Results for model {i}:", results)
    # Train.
    # print("Launching training jobs")
    # for i, model in enumerate(models):
    #     results = train(
    #             task_id=i, model=model, 
    #             train_dataloader=train_dataloaders[i], val_dataloader=val_dataloader, 
    #             device=gpus[i], lr=args.lr, num_epochs=args.num_epochs, save_dir=os.path.join(save_dir, str(i))
    #         )
    #     print(f"Results for model {i}:", results)
    #     break
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [
    #         executor.submit(
    #             train,
    #             task_id=i,
    #             model=models[i],
    #             train_dataloader=train_dataloaders[i],
    #             valmatched_dataloader=valmatched_dataloader,
    #             valmismatched_dataloader=valmismatched_dataloader,
    #             device=gpus[i % len(gpus)],
    #             lr=args.lr,
    #             num_epochs=args.num_epochs,
    #             save_dir=os.path.join(save_dir, str(i)),
    #         )
    #         for i in range(args.num_models)
    #     ]
    #     metrics = [
    #         future.result()
    #         for future in concurrent.futures.as_completed(futures)
    #     ]

    print("ALL DONE")

if __name__ == "__main__":
    main(parse_args())
