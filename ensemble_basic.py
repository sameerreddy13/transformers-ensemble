import argparse
import concurrent.futures
import os
import pickle
import random
from pathlib import Path

import datasets
import torch
import transformers

import utils


def parse_args():
    ap = argparse.ArgumentParser()
    default_help = "(default: %(default)s)"
    ap.add_argument("--save-dir", type=str, default="checkpoints", help=default_help)
    ap.add_argument("--gpus", nargs="+", default=list(range(8)), help=default_help)
    ap.add_argument("--seq-per-gpu", action="store_true", default=False, help=default_help)
    ap.add_argument("--num-models", type=int, default=8, help=default_help)
    ap.add_argument("--dataset", type=str, default="sst2", help=default_help)
    ap.add_argument("--distillation-dataset", type=str, default=None, help=default_help)
    ap.add_argument("--augmented", action="store_true", default=False)
    ap.add_argument("--extract-subnetwork", action="store_true", default=False, help=default_help)
    ap.add_argument("--architecture-selection", type=str, default="fixed", help=default_help)
    ap.add_argument("--num-epochs", type=int, default=50, help=default_help)
    ap.add_argument("--batch-size", type=int, default=32, help=default_help)
    ap.add_argument("--val-batch-size", type=int, default=32, help=default_help)
    ap.add_argument("--lr", type=float, default=1e-3, help=default_help)
    ap.add_argument("--limit", type=int, default=-1, help=default_help)
    ap.add_argument("-wd", "--weight-decay", type=float, default=0.01, help=default_help)
    ap.add_argument("--warmup-steps", type=int, default=1000, help=default_help)

    return ap.parse_args()


def train_one_epoch(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    save_path,
    scheduler=None,
    distillation=False,
    print_freq=50,
    prefix="",
):
    model = model.train()
    metrics = {}
    train_losses = {}
    for i, example in enumerate(train_dataloader):
        input_ids = example[0].to(device)
        attention_mask = example[1].to(device)
        labels = example[2].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        loss = outputs.loss

        if distillation:
            bert_last_hidden_state = example[3].to(device)
            distill_loss = utils.distillation_loss(
                outputs["hidden_states"][-1],
                bert_last_hidden_state,
                mask=attention_mask,
            )
            loss = loss + distill_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        if i % print_freq == 0:
            print(f"{prefix} Step {i + 1} of {len(train_dataloader)}: " f"loss = {loss.item()}")
            train_losses[i] = loss.item()

    model = model.eval()
    metrics = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item(),
        "train_acc": utils.compute_acc(model, train_dataloader, device=device),
        "val_acc": utils.compute_acc(model, val_dataloader, device=device),
        "train_losses": train_losses,
    }
    print(f"{prefix} Train accuracy: {metrics['train_acc']}")
    print(f"{prefix} Validation accuracy: {metrics['val_acc']}")

    torch.save(metrics, save_path)
    print(f"{prefix} Saved model checkpoint to {save_path}")

    return metrics


# TODO(piyush) Turn this into a decorator and put in utils.py
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
    weight_decay=0.0,
    distillation=False,
    warmup_steps=0,
    num_epochs=100,
    print_freq=50,
):
    prefix = f"[Process {task_id}]"
    os.makedirs(save_dir, exist_ok=True)
    print(f"{prefix} Created {save_dir}")

    model = model.to(device)
    print(f"{prefix} Moved model to device {device}")

    metrics = {}
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = None
    if warmup_steps != 0:
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(train_dataloader) * num_epochs,
        )
    for epoch in range(num_epochs):
        epoch_metrics = train_one_epoch(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            device,
            scheduler=scheduler,
            distillation=distillation,
            save_path=os.path.join(save_dir, f"model_epoch{epoch}.pt"),
            print_freq=print_freq,
            prefix=f"{prefix} [Epoch {epoch}]",
        )
        metrics[epoch] = epoch_metrics

    return metrics


def train_share_gpu(jobs):
    prefix = f"[Process {jobs[0]['task_id']}]"

    for job in jobs:
        os.makedirs(job["save_dir"], exist_ok=True)
        print(f"{prefix} Created {job['save_dir']}")

    device = jobs[0]["device"]
    num_epochs = jobs[0]["num_epochs"]
    optimizers = [
        torch.optim.SGD(
            job["model"].parameters(),
            lr=job["lr"],
            momentum=0.9,
            weight_decay=job["weight_decay"],
        )
        for job in jobs
    ]
    # TODO - support scheduler
    metrics = [{} for _ in range(len(jobs))]
    for epoch in range(num_epochs):
        for i, job in enumerate(jobs):
            job_prefix = f"{prefix} [Model {i}]"
            print(f"{job_prefix} Starting training for epoch {epoch}")

            model = job["model"].to(device)
            print(f"{job_prefix} Moved model to device {device}")

            epoch_metrics = train_one_epoch(
                model,
                job["train_dataloader"],
                job["val_dataloader"],
                optimizers[i],
                device,
                save_path=os.path.join(job["save_dir"], f"model_epoch{epoch}.pt"),
                distillation=job["distillation"],
                prefix=f"{job_prefix} [Epoch {epoch}]",
            )
            metrics[i][epoch] = epoch_metrics

    return metrics


def main(args):
    print(f"Save dir: {args.save_dir}")
    # Determine devices
    if args.gpus is None or len(args.gpus) == 0:
        print("WARNING: Using CPU")
        gpus = ["cpu"]
    else:
        gpus = [f"cuda:{i}" for i in args.gpus]
        print(f"Using GPUs: {', '.join(gpus)}")
    # Set up data loaders.
    print(f"Building dataloaders for dataset: {args.dataset}")
    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    if args.augmented:
        aug_ds_path = Path(f"data/augmented_train_ds/{args.dataset}_augmented.pt")
        tensors_ds = list(torch.load(aug_ds_path))[: args.limit]
        partition_size = len(tensors_ds) // args.num_models + 1
        train_dataloaders = [
            torch.utils.data.DataLoader(
                tensors_ds[i : i + partition_size], batch_size=args.batch_size
            )
            for i in range(0, len(tensors_ds), partition_size)
        ]
        print(f"Partitioned {len(tensors_ds)} total training samples")
        ds = datasets.load_dataset("glue", args.dataset)
        val_dataloader = utils.create_dataloader(
            ds["validation"], tokenizer, args.val_batch_size, args.dataset
        )
    else:
        if args.distillation_dataset is not None:
            with open(args.distillation_dataset, "rb") as f:
                ds = pickle.load(f)
            print(f"Loaded distillation dataset from {args.distillation_dataset}")
        else:
            ds = datasets.load_dataset("glue", args.dataset)
        train_ds = list(ds["train"])[: args.limit]
        random.shuffle(train_ds)
        partition_size = len(train_ds) // args.num_models + 1
        train_dataloaders = [
            utils.create_dataloader(
                train_ds[i : i + partition_size],
                tokenizer,
                args.batch_size,
                args.dataset,
                distillation=args.distillation_dataset is not None,
            )
            for i in range(0, len(train_ds), partition_size)
        ]
        print(f"Partitioned {len(train_ds)} total training samples")
        val_dataloader = utils.create_dataloader(
            ds["validation"], tokenizer, args.val_batch_size, args.dataset
        )

    # Build models (and check param counts).
    models = utils.build_models(
        num_models=args.num_models,
        extract_subnetwork=args.extract_subnetwork,
        architecture_selection=args.architecture_selection,
    )
    utils.check_param_counts(models)
    # Setup jobs.
    jobs = [
        {
            "task_id": i,
            "model": models[i],
            "train_dataloader": train_dataloaders[i],
            "val_dataloader": val_dataloader,
            "device": gpus[i % len(gpus)],
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "save_dir": os.path.join(args.save_dir, str(i)),
            "distillation": args.distillation_dataset is not None,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
        }
        for i in range(args.num_models)
    ]
    # Fixes too many open files error. (See https://github.com/pytorch/pytorch/issues/11201)
    torch.multiprocessing.set_sharing_strategy("file_system")
    # Train.
    if args.num_models == 1:
        metrics = train(**jobs[0])
    elif args.seq_per_gpu:
        # Collect jobs by GPU.
        jobs_per_gpu = {gpu: [] for gpu in gpus}
        for job in jobs:
            jobs_per_gpu[job["device"]].append(job)
        with torch.multiprocessing.Pool(len(jobs_per_gpu)) as pool:
            metrics = pool.map(train_share_gpu, jobs_per_gpu.values())
    else:
        with torch.multiprocessing.Pool(len(jobs)) as pool:
            metrics = pool.map(train_wrapper, jobs)
    # Save final metrics.
    metrics_save_path = os.path.join(args.save_dir, "all_metrics.pkl")
    with open(metrics_save_path, "wb") as f:
        pickle.dump(metrics_save_path, f)
    print(f"Done. Saved all metrics to: {metrics_save_path}")


if __name__ == "__main__":
    main(parse_args())
