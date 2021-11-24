import argparse
import pickle

import datasets
import tqdm
import transformers

import utils


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--output-path", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="sst2")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda:0")

    return ap.parse_args()


def main():
    ds = datasets.load_dataset("glue", ARGS.dataset)
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    bert = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')
    bert = bert.to(ARGS.device)

    new_ds = {key: [] for key in ds}
    for key in ds:
        print(f"Generating encodings for {key} dataset")
        if ARGS.dataset == "sst2":
            encodings = tokenizer(
                [example['sentence'] for example in ds[key]], max_length=128,
                add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')
        elif ARGS.dataset == "mnli":
            encodings = tokenizer(
                [example["premise"] for example in ds[key]],
                [example["hypothesis"] for example in ds[key]],
                max_length=128, add_special_tokens=True, padding=True, truncation=True,
                return_tensors='pt')
        else:
            raise ValueError(f"Unknown dataset {name}")

        print(f"Running model on {key} dataset")
        for i in tqdm.tqdm(range(0, len(ds[key]), ARGS.batch_size)):
            input_ids = encodings["input_ids"][i : i + ARGS.batch_size].to(ARGS.device)
            attention_mask = encodings["attention_mask"][i : i + ARGS.batch_size].to(ARGS.device)
            logits = bert(input_ids=input_ids, attention_mask=attention_mask).logits

            logits = logits.cpu().detach().numpy()
            for j in range(len(logits)):
                new_ds[key].append({**ds[key][i + j], "bert_logits": logits[j]})

    with open(ARGS.output_path, "wb") as f:
        pickle.dump(new_ds, f)
    print(f"Saved dataset to: {ARGS.output_path}")


if __name__ == "__main__":
    ARGS = parse_args()
    main()
