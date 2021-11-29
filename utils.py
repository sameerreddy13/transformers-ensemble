import random

import numpy as np
import torch
import transformers
from transformers import BertConfig, BertForSequenceClassification

# Number of parameters in the original pretrained BERT architecture.
BERT_N_PARAMS = 109483778
BERT_N_PARAMS_NO_EMB = 85648130
ENSEMBLE_COUNTS = [1, 2, 4, 8, 16, 32]


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


def create_encodings(dataset, tokenizer, name):
    if name == "sst2":
        encodings = tokenizer(
            [example["sentence"] for example in dataset],
            max_length=128,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )
    elif name == "mnli":
        encodings = tokenizer(
            [example["premise"] for example in dataset],
            [example["hypothesis"] for example in dataset],
            max_length=128,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )
    else:
        raise ValueError(f"Unknown dataset {name}")
    return encodings


def create_tensor_dataset(dataset, encodings, distillation=False):
    labels = torch.tensor([example["label"] for example in dataset])
    tensors = [encodings["input_ids"], encodings["attention_mask"], labels]
    tensors_ds = torch.utils.data.TensorDataset(*tensors)
    if distillation:
        tensors.append(torch.tensor([example["bert_last_hidden_state"] for example in dataset]))
    return tensors_ds


def create_dataloader(dataset, tokenizer, batch_size, name, distillation=False):
    encodings = create_encodings(dataset, tokenizer, name)
    tensors_ds = create_tensor_dataset(dataset, encodings, distillation)
    dataloader = torch.utils.data.DataLoader(tensors_ds, batch_size=batch_size)
    return dataloader


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


def get_subnet_configs_fixed(num_models, **kwargs):
    """
    Return list of model configs from num_models.

    TODO - support heterogenous model types
    Currently returns num_models copies of the same architecture

    Returns:
        configs (List[dict]): [
            {
            "num_hidden_layers": x,
            "num_attention_heads": y,
            "intermediate_size": z
            }
            ...
        ]
    """
    assert num_models in ENSEMBLE_COUNTS, f"Num models {num_models} not supported"
    base_config = {
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
    }
    if num_models == 2:
        base_config = {
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
        }
    elif num_models == 4:
        base_config = {
            "num_hidden_layers": 6,
            "num_attention_heads": 6,
            "intermediate_size": 3072 // 2,
        }
    elif num_models == 8:
        base_config = {
            "num_hidden_layers": 4,
            "num_attention_heads": 6,
            "intermediate_size": 3072 // 3,
        }
    elif num_models == 16:
        base_config = {
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 3072 // 3,
        }
    elif num_models == 32:
        base_config = {
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 3072 // 4,
        }
    return [base_config.copy() for _ in range(num_models)]


def get_subnet_configs_beta(num_models, beta=0.95, base_hidden_size=768):
    """
    Heuristic based around num_models for automatically
    getting subnet configs
    """
    import math

    m = pow(beta, num_models)
    if num_models == 1:
        m = 1

    base_config = {"num_hidden_layers": int(12 * m), "intermediate_size": int(3072 * m)}
    num_attention_heads = int(12 * m)
    valid_attention_heads = [a for a in range(1, 13) if base_hidden_size % a == 0]
    num_attention_heads = min(valid_attention_heads, key=lambda x: abs(x - num_attention_heads))
    base_config["num_attention_heads"] = num_attention_heads
    return [base_config.copy() for _ in range(num_models)]


def get_naive_model(**config):
    num_attention_heads = config.pop("num_attention_heads", 12)
    model = BertForSequenceClassification(
        BertConfig(
            **config
    ))
    # Use extract subnetwork method to adjust attention heads
    return extract_subnetwork_from_bert(
        pretrained=model,
        num_attention_heads=num_attention_heads
    )

def build_models(num_models, extract_subnetwork=False, architecture_selection="fixed"):
    """
    Build num_models models for ensemble
    """
    assert num_models > 0
    if architecture_selection.lower() == "fixed":
        configs = get_subnet_configs_fixed(num_models)
    elif architecture_selection.lower() == "beta":
        configs = get_subnet_configs_beta(num_models, base_hidden_size=768)
    else:
        raise ValueError("Subnet selection strategy {subnet_selection} not supported")
    print("Sample model config:", configs[0])
    if extract_subnetwork:
        print("Extracting subnetworks from pretrained BERT")
        if num_models == 1:
            print("Using pretrained BERT for single model")
        models = [
            extract_subnetwork_from_bert(**configs[i])
            for i in range(num_models)
        ]
    else:
        print("Creating models from scratch")
        models = [
            get_naive_model(**configs[i])
            for i in range(num_models)
        ]
    return models, configs

def load_model_checkpoint(checkpoint_path, naive=False):
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['arch']
    if naive:
        if '11-27' in str(checkpoint_path): 
            del config['num_attention_heads']
        model = get_naive_model(**config)
    else:
        model = extract_subnetwork_from_bert(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model    

def extract_subnetwork_from_bert(
    pretrained=None,
    num_hidden_layers=None,
    num_attention_heads=None,
    intermediate_size=None,
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
    if pretrained:
        model = pretrained
    else:
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    bert = model.bert

    # Randomly select layers.
    if num_hidden_layers is not None and num_hidden_layers != bert.config.num_hidden_layers:
        layers = sorted(random.sample(range(bert.config.num_hidden_layers), num_hidden_layers))
        # layers = range(num_hidden_layers)
        bert.encoder.layer = torch.nn.ModuleList([bert.encoder.layer[i] for i in layers])
        bert.config.num_hidden_layers = num_hidden_layers

    # Randomly drop out neurons in fully connected layers.
    if intermediate_size is not None and intermediate_size != bert.config.intermediate_size:
        output_neurons = sorted(
            random.sample(range(bert.config.intermediate_size), intermediate_size)
        )
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
    if num_attention_heads is not None and num_attention_heads != bert.config.num_attention_heads:
        assert bert.config.hidden_size % num_attention_heads == 0
        heads = sorted(random.sample(range(bert.config.num_attention_heads), num_attention_heads))
        for i in range(len(bert.encoder.layer)):
            attention = bert.encoder.layer[i].attention

            layer = attention.self
            layer.num_attention_heads = num_attention_heads
            layer.all_head_size = num_attention_heads * layer.attention_head_size

            for matrix in (layer.query, layer.key, layer.value):
                matrix.weight = torch.nn.Parameter(
                    torch.cat(
                        [
                            matrix.weight[
                                h * layer.attention_head_size : (h + 1) * layer.attention_head_size
                            ]
                            for h in heads
                        ]
                    )
                )
                matrix.bias = torch.nn.Parameter(
                    torch.cat(
                        [
                            matrix.bias[
                                h * layer.attention_head_size : (h + 1) * layer.attention_head_size
                            ]
                            for h in heads
                        ]
                    )
                )
                matrix.out_features = layer.all_head_size

            attention.output.dense.weight = torch.nn.Parameter(
                torch.cat(
                    [
                        attention.output.dense.weight[
                            :,
                            h * layer.attention_head_size : (h + 1) * layer.attention_head_size,
                        ]
                        for h in heads
                    ],
                    dim=1,
                )
            )
            attention.output.dense.in_features = layer.all_head_size
        bert.config.num_attention_heads = num_attention_heads
        bert.config.attention_head_size = bert.config.hidden_size // num_attention_heads

    model.bert = bert
    return model


def get_param_count(model):
    """
    Get param counts not including embedding layers
    """
    # (excluding embedding layers).
    n_params = sum(
        [
            param.numel()
            for name, param in model.named_parameters()
            if all(
                param_name not in name
                for param_name in (
                    "word_embeddings",
                    "position_embeddings",
                    "token_type_embeddings",
                )
            )
        ]
    )
    return n_params


def get_param_ratios(n_params, n_models):
    param_ratio = n_params / BERT_N_PARAMS_NO_EMB
    total_ratio = param_ratio * n_models
    return param_ratio, total_ratio


def check_param_counts(models):
    """
    Helper for printing param counts of ensemble models and comparing with BERT
    """
    # Preserve the same total parameter count as original BERT, within a 10% margin
    # (excluding embedding layers).
    n_params = get_param_count(models[0])
    param_ratio, total_ratio = get_param_ratios(n_params, len(models))
    print(
        f"Created {len(models)} models, each with {n_params / 1e6} million parameters "
        f"({round(param_ratio * 100, 2)}% per model "
        f"-> {round(total_ratio * 100, 2)}% total) "
        f"(not counting embedding layers)"
    )
    if not (1 / 1.1 <= total_ratio <= 1.1):
        print("WARNING: Total number of parameters isn't within 10% of BERT")
