import argparse

import datasets
import torch
import transformers
import tqdm


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cpu", action="store_true", default=False)
    ap.add_argument("--model-type", type=str, default="albert")
    ap.add_argument("--num-models", type=int, default=10)
    ap.add_argument("--dataset", type=str, default="mnli")
    ap.add_argument("--num-epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--val-batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)

    return ap.parse_args()


@torch.no_grad()
def compute_acc(model, dataloader, cpu=False):
    accs = []
    for input_ids, attention_mask, labels in tqdm.tqdm(dataloader):
        if not cpu:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        accs.append((logits.argmax(axis=-1) == labels).float().mean())
    return sum(accs) / len(accs)


def create_dataloader(dataset, tokenizer, batch_size):
    encodings = tokenizer(
        [example["premise"] for example in dataset], [example["hypothesis"] for example in dataset],
         max_length=128, add_special_tokens=True, padding=True, truncation=True,
        return_tensors='pt')
    labels = torch.tensor([example["label"] for example in dataset])
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels),
        batch_size=batch_size)


def main(args):
    # Set up data loader.
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    ds = datasets.load_dataset("glue", args.dataset)

    ds["train"] = [ds["train"][i] for i in range(120 * args.batch_size)] # TODO(piyush) remove

    train_dataloader = create_dataloader(ds["train"], tokenizer, args.batch_size)
    valmatched_dataloader = create_dataloader(
        ds["validation_matched"], tokenizer, args.val_batch_size)
    valmismatched_dataloader = create_dataloader(
        ds["validation_mismatched"], tokenizer, args.val_batch_size)

    # Build models.
    if args.model_type == "bert":
        config = transformers.BertConfig(
            hidden_size=540,         # 768
            num_hidden_layers=3,     # 12
            intermediate_size=1024,  # 3072
        )
        models = [
            transformers.BertForSequenceClassification(config)
            for _ in range(args.num_models)
        ]
    elif args.model_type == "albert":
        config = transformers.AlbertConfig(
            embedding_size=128,          # 128
            hidden_size=768,             # 4096
            num_hidden_layers=3,         # 12
            intermediate_size=3072,      # 16384
            initializer_range=0.2,       # 0.02
            num_labels=3,
        )
        models = [
            transformers.AlbertForSequenceClassification(config)
            for _ in range(args.num_models)
        ]
    else:
        raise ValueError(f"Unknown model type {model_type}")

    # Default: 109,483,778
    total_params = sum([param.numel() for param in models[0].parameters()])
    print(f"Number of parameters per model: {total_params / 1e6} million")

    # TODO(piyush) remove
    model = models[0]
    if not args.cpu: model = model.cuda()

    # Train.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}")
        pbar = tqdm.tqdm(train_dataloader)
        for input_ids, attention_mask, labels in pbar:
            if not args.cpu:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            pbar.set_description(f"Loss = {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = compute_acc(model, train_dataloader, cpu=args.cpu)
        matchedval_acc = compute_acc(model, valmatched_dataloader, cpu=args.cpu)
        mismatchedval_acc = compute_acc(model, valmismatched_dataloader, cpu=args.cpu)
        print(f"\tTrain accuracy: {train_acc}")
        print(f"\tMatched validation accuracy: {matchedval_acc}")
        print(f"\tMismatched validation accuracy: {mismatchedval_acc}")

if __name__ == "__main__":
    main(parse_args())
