import random

import torch
import transformers


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


def extract_subnetwork_from_bert(
    # hidden_size=None, # TODO(piyush) remove
    num_hidden_layers=None,
    num_attention_heads=None,
    intermediate_size=None
):
    """
    For reference, the BERT module structure:
        bert
            embeddings
                word_embeddings: Embedding(vocab_size (30522), hidden_size)
                position_embeddings: Embedding(max_position_embeddings (512), hidden_size)
                token_type_embeddings: Embedding(type_vocab_size (2), hidden_size)
                LayerNorm
                dropout
            encoder
                layer
                    1, ..., num_hidden_layers
                        attention
                            self
                                query: Linear(hidden_size, hidden_size)
                                key: Linear(hidden_size, hidden_size)
                                value: Linear(hidden_size, hidden_size)
                                dropout
                            output
                                dense: Linear(hidden_size, hidden_size)
                                LayerNorm
                                dropout
                        intermediate
                            dense: Linear(hidden_size, intermediate_size)
                        output
                            dense: Linear(intermediate_size, hidden_size)
                            LayerNorm
                            dropout
            pooler
                dense: Linear(hidden_size, hidden_size)
                activation
        dropout
        classifier: Linear(hidden_size, num_labels)
    """
    model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
    bert = model.bert

    # Randomly select layers.
    if num_hidden_layers is not None:
        layers = sorted(random.sample(range(bert.config.num_hidden_layers), num_hidden_layers))
        bert.encoder.layer = torch.nn.ModuleList([bert.encoder.layer[i] for i in layers])
        bert.config.num_hidden_layers = num_hidden_layers

    # Randomly drop out neurons in fully connected layers.
    if intermediate_size is not None:
        output_neurons = sorted(random.sample(range(bert.config.intermediate_size), intermediate_size))
        for i in range(len(bert.encoder.layer)):
            layer = bert.encoder.layer[i].intermediate.dense
            layer.weight = torch.nn.Parameter(layer.weight[output_neurons])
            layer.bias = torch.nn.Parameter(layer.bias[output_neurons])
            layer.out_features = intermediate_size

            layer = bert.encoder.layer[i].output.dense
            layer.weight = torch.nn.Parameter(layer.weight[:, output_neurons])
            layer.in_features = intermediate_size
        bert.config.intermediate_size = intermediate_size

    # Randomly drop out attention heads.
    if num_attention_heads is not None:
        assert bert.config.hidden_size % num_attention_heads == 0
        heads = sorted(random.sample(range(bert.config.num_attention_heads), num_attention_heads))
        for i in range(len(bert.encoder.layer)):
            attention = bert.encoder.layer[i].attention

            layer = attention.self
            layer.num_attention_heads = num_attention_heads
            layer.all_head_size = num_attention_heads * layer.attention_head_size

            for matrix in (layer.query, layer.key, layer.value):
                matrix.weight = torch.nn.Parameter(torch.cat([
                    matrix.weight[
                        h * layer.attention_head_size : (h + 1) * layer.attention_head_size]
                    for h in heads
                ]))
                matrix.bias = torch.nn.Parameter(torch.cat([
                    matrix.bias[h * layer.attention_head_size : (h + 1) * layer.attention_head_size]
                    for h in heads
                ]))
                matrix.out_features = layer.all_head_size

            attention.output.dense.weight = torch.nn.Parameter(torch.cat([
                attention.output.dense.weight[
                    :, h * layer.attention_head_size : (h + 1) * layer.attention_head_size]
                for h in heads
            ], dim=1))
            attention.output.dense.in_features = layer.all_head_size
        bert.config.num_attention_heads = num_attention_heads
        bert.config.attention_head_size = bert.config.hidden_size // num_attention_heads

    model.bert = bert
    return model
