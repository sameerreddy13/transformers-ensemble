import argparse
import concurrent.futures
import os
import pickle
import random

import datasets
import nlpaug.augmenter.word as naw
# import torch
import transformers
from tqdm import tqdm

from utils import create_dataloader


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--save-dir", type=str, default="checkpoints")
    ap.add_argument("--gpus", nargs="+", default=list(range(8)))
    ap.add_argument("--seq-per-gpu", action="store_true", default=False)
    ap.add_argument("--num-models", type=int, default=8)
    ap.add_argument("--dataset", type=str, default="sst2")
    # ap.add_argument("--extract-subnetwork", action="store_true", default=False)
    # ap.add_argument("--num-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    # ap.add_argument("--val-batch-size", type=int, default=32)
    # ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--language", type=str, default="fr")
    # ['fr', 'de', 'es', 'it'] == [french, german, spanish, italian]
    return ap.parse_args()


def augment_sentences(ds, language):
    """
    Augment sentences with nlpaug
    """
    augmented_sentences = []
    idx = len(ds)
    aug = naw.BackTranslationAug(
        from_model_name=f"Helsinki-NLP/opus-mt-en-{language}",
        to_model_name=f"Helsinki-NLP/opus-mt-{language}-en",
        device="cuda",
        batch_size=1024,
    )
    for entry in tqdm(ds):
        sentence = aug.augment(entry["sentence"])
        augmented_sentences.append(
            {"idx": idx, "label": entry["label"], "sentence": sentence}
        )
        idx += 1
    return augmented_sentences


def main(args):
    print(f"Save dir: {args.save_dir}")

    if args.gpus is None or len(args.gpus) == 0:
        print("WARNING: Using CPU")
        gpus = ["cpu"]
    else:
        gpus = [f"cuda:{i}" for i in args.gpus]
        print(f"Using GPUs: {', '.join(gpus)}")

    # Set up data loader.
    print(
        f"Augmenting the training split from dataset: {args.dataset}"
        f"using back translation with Helsinki-NLP/opus-mt-en-{args.language}"
    )
    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    ds = datasets.load_dataset("glue", args.dataset)

    train_ds = list(ds["train"])
    aug_ds = augment_sentences(train_ds, args.language)

    random.shuffle(aug_ds[: args.limit])
    partition_size = len(aug_ds) // args.num_models + 1
    train_dataloaders = [
        create_dataloader(
            aug_ds[i : i + partition_size],
            tokenizer,
            args.batch_size,
            args.dataset,
            distillation=args.distillation_dataset is not None,
        )
        for i in range(0, len(aug_ds), partition_size)
    ]
    print(f"Partitioned {len(train_ds)} total training samples")


if __name__ == "__main__":
    main(parse_args())