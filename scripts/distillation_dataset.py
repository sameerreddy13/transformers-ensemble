import argparse
import pickle

import datasets
import torch
import tqdm
import transformers
import utils


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--output-path", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="sst2")
    ap.add_argument("--store-logits", action="store_true", default=False)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda:0")

    # Fine-tuning arguments, if --store-logits is passed.
    ap.add_argument("--num-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.9)

    return ap.parse_args()


def train(model, train_dataloader, val_dataloader, device):
    model = model.train().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=ARGS.lr, momentum=ARGS.momentum)
    for epoch in range(ARGS.num_epochs):
        model = model.train()
        pbar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (input_ids, attention_mask, labels) in pbar:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            pbar.set_description(f"Loss = {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model = model.eval()
        print(
            f"Epoch {epoch} train accuracy:",
            utils.compute_acc(model, train_dataloader, device=device),
        )
        print(
            f"Epoch {epoch} val accuracy:",
            utils.compute_acc(model, val_dataloader, device=device),
        )

    return model


def main():
    ds = datasets.load_dataset("glue", ARGS.dataset)
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    encodings = {}
    for key in ds:
        print(f"Generating encodings for {key} dataset")
        if ARGS.dataset == "sst2":
            encodings[key] = tokenizer(
                [example["sentence"] for example in ds[key]],
                max_length=128,
                add_special_tokens=True,
                padding="max_length",
                return_tensors="pt",
            )
        elif ARGS.dataset == "mnli":
            encodings[key] = tokenizer(
                [example["premise"] for example in ds[key]],
                [example["hypothesis"] for example in ds[key]],
                max_length=128,
                add_special_tokens=True,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            raise ValueError(f"Unknown dataset {name}")

    model = transformers.BertForSequenceClassification.from_pretrained(
        "bert-base-uncased"
    )
    model = model.to(ARGS.device)

    if ARGS.store_logits:
        print(f"Fine-tuning model on train data")
        train_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                encodings["train"]["input_ids"],
                encodings["train"]["attention_mask"],
                torch.tensor([example["label"] for example in ds["train"]]),
            ),
            batch_size=ARGS.batch_size,
        )
        val_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                encodings["validation"]["input_ids"],
                encodings["validation"]["attention_mask"],
                torch.tensor([example["label"] for example in ds["validation"]]),
            ),
            batch_size=ARGS.batch_size,
        )
        model = train(model, train_dataloader, val_dataloader, ARGS.device)

    model = model.eval()
    new_ds = {key: [] for key in ds}
    for key in ds:
        print(f"Running model on {key} dataset")
        for i in tqdm.tqdm(range(0, len(ds[key]), ARGS.batch_size)):
            input_ids = encodings[key]["input_ids"][i : i + ARGS.batch_size].to(
                ARGS.device
            )
            attention_mask = encodings[key]["attention_mask"][
                i : i + ARGS.batch_size
            ].to(ARGS.device)
            if ARGS.store_logits:
                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                logits = logits.cpu().detach().numpy()
                for j in range(len(logits)):
                    new_ds[key].append({**ds[key][i + j], "bert_logits": logits[j]})
            else:
                features_dict = model.bert(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                last_hidden_state = (
                    features_dict["last_hidden_state"].cpu().detach().numpy()
                )
                pooler_output = features_dict["pooler_output"].cpu().detach().numpy()
                for j in range(len(last_hidden_state)):
                    new_ds[key].append(
                        {
                            **ds[key][i + j],
                            "bert_last_hidden_state": last_hidden_state[j],
                            "bert_pooler_output": pooler_output[j],
                        }
                    )

    with open(ARGS.output_path, "wb") as f:
        pickle.dump(new_ds, f)
    print(f"Saved dataset to: {ARGS.output_path}")


if __name__ == "__main__":
    ARGS = parse_args()
    main()
