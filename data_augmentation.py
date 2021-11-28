import argparse
import concurrent.futures
import os
import pickle
import random
from pathlib import Path

import datasets
import nlpaug.augmenter.word as naw
import torch
import transformers
from tqdm import tqdm

from utils import create_encodings, create_tensor_dataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="sst2")
    ap.add_argument("--gpu", type=str, default="cuda:0")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--language", type=str, default="fr")
    ap.add_argument("--save-dir", type=str, default="data/augmented_train_ds")
    # ['fr', 'de', 'es', 'it'] == [french, german, spanish, italian]
    return ap.parse_args()


def augment_sentences(ds, language, gpu="cuda:0"):
    """
    Augment sentences with nlpaug
    """
    augmented_sentences = []
    idx = len(ds)
    aug = naw.BackTranslationAug(
        from_model_name=f"Helsinki-NLP/opus-mt-en-{language}",
        to_model_name=f"Helsinki-NLP/opus-mt-{language}-en",
        device=gpu,
        batch_size=1024,
    )
    for entry in tqdm(ds):
        sentence = aug.augment(entry["sentence"])
        augmented_sentences.append(
            {"idx": idx, "label": entry["label"], "sentence": sentence}
        )
        idx += 1
    return augmented_sentences


def combine_datasets():
    """
    Combine the augmented datasets
    """
    # TODO: undo hard code if needed
    dataset = "sst2"
    save_dir = "data/augmented_train_ds"

    print(f"Combining augmented datasets")
    ds = datasets.load_dataset("glue", dataset)
    train_ds = list(ds["train"])
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    encodings = create_encodings(dataset=train_ds, tokenizer=tokenizer, name=dataset)
    tensors_ds = create_tensor_dataset(
        dataset=train_ds, encodings=encodings, distillation=False
    )
    print(f"Original dataset has length {len(tensors_ds)}")
    for language in ["fr", "de", "es", "it"]:
        print(f"Loading augmented dataset for {language}")
        augmented_ds = torch.load(f"data/augmented_train_ds/{dataset}_{language}.pt")
        print(f"{language} augmented dataset has length {len(augmented_ds)}")
        tensors_ds = torch.utils.data.ConcatDataset((tensors_ds, augmented_ds))
        print(f"Combined dataset now has length {len(tensors_ds)}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(f"{save_dir}/{dataset}_augmented.pt")

    if output_file.is_file():
        print(f"Existing file found at {output_file}, do you want to overwrite? [y/n]")
        answer = input("[y]es or [n]o: ")
        if answer == "yes" or answer == "y":
            output_file.unlink()
        elif answer == "no" or answer == "n":
            print("Exiting")
            exit(0)
        else:
            print("Please enter yes or no.")
    torch.save(obj=train_ds, f=output_file)
    print(f"Saved tensor dataset to {output_file} -- Testing the save")
    _combined_dataset = torch.load(f=output_file)
    print(f"Reloaded the combined dataset with length {len(_combined_dataset)}")


def main(args):
    print(f"Save dir: {args.save_dir}")

    if args.gpu is None or len(args.gpu) == 0:
        print("WARNING: Using CPU")
        gpu = ["cpu"]
    else:
        print(f"Using GPU: {args.gpu}")

    print(
        f"Augmenting the training split from dataset: {args.dataset}"
        f"using back translation with Helsinki-NLP/opus-mt-en-{args.language}"
    )
    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    ds = datasets.load_dataset("glue", args.dataset)

    train_ds = list(ds["train"])[: args.limit]
    print(f"Augmenting {len(train_ds)} sentences using {args.language}")
    aug_ds = augment_sentences(train_ds, args.language, args.gpu)
    print(f"Augmentation complete -- Saving tensor dataset to disk")
    encodings = create_encodings(
        dataset=train_ds, tokenizer=tokenizer, name=args.dataset
    )
    tensors_ds = create_tensor_dataset(
        dataset=train_ds, encodings=encodings, distillation=False
    )
    # Save the tensor dataset
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(f"{args.save_dir}/{args.dataset}_{args.language}.pt")
    torch.save(obj=tensors_ds, f=output_path)
    print(f"Saved tensor dataset to {output_path}")


if __name__ == "__main__":
    main(parse_args())
