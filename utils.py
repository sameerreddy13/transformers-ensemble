import torch


@torch.no_grad()
def compute_acc(model, dataloader, device):
    accs = []
    for example in dataloader:
        input_ids = example[0].to(device)
        attention_mask = example[1].to(device)
        labels = example[2].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        accs.append((logits.argmax(axis=-1) == labels).float().mean())
    return sum(accs) / len(accs)


def create_dataloader(dataset, tokenizer, batch_size, name, distillation=False):
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
    tensors = [encodings["input_ids"], encodings["attention_mask"], labels]
    if distillation:
        tensors.append(torch.tensor([example["bert_last_hidden_state"] for example in dataset]))

    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*tensors), batch_size=batch_size)


# TODO(piyush) Incorporate difference of embedding vector magnitudes?
def distillation_loss(features, target_features, mask=None):
    if mask is not None:
        features = features * mask.unsqueeze(-1)
        target_features = target_features * mask.unsqueeze(-1)
    similarity = torch.nn.functional.cosine_similarity(features, target_features, dim=-1)
    loss = (1 - similarity.abs()) * mask
    # loss = (features - target_features).norm(dim=-1) # TODO(piyush) remove

    # Average over sequence and batch dimensions.
    loss = loss.sum(dim=-1) / (mask != 0).float().sum(dim=-1)
    return loss.mean()
