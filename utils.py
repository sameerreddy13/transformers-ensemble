import torch


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


def create_dataloader(dataset, tokenizer, batch_size, name):
    if name == "sst2":
        encodings = tokenizer(
            [example['sentence'] for example in dataset], max_length=128, add_special_tokens=True,
            padding="max_length", return_tensors='pt')
    elif name == "mnli":
        encodings = tokenizer(
            [example["premise"] for example in dataset],
            [example["hypothesis"] for example in dataset],
            max_length=128, add_special_tokens=True, padding="max_length", return_tensors='pt')
    else:
        raise ValueError(f"Unknown dataset {name}")

    labels = torch.tensor([example["label"] for example in dataset])
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels),
        batch_size=batch_size)
