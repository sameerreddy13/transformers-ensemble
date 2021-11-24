import pickle

import transformers


vals = {}
for n1 in range(1, 33):
    embedding_size = int(128 * n1 / 32)
    for n2 in range(1, 33):
        hidden_size = int(4096 * n2 / 32)
        for n3 in range(1, 33):
            intermediate_size = int(16384 * n3 / 32)
            config = transformers.AlbertConfig(
                embedding_size=embedding_size,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_labels=3,
            )
            model = transformers.AlbertForSequenceClassification(config)
            n_params = sum([param.numel() for param in model.parameters()])
            vals[(embedding_size, hidden_size, intermediate_size)] = n_params

            print(f"Emb size = {embedding_size}, h size = {hidden_size}, fc size = "
                  f"{intermediate_size}: {n_params / 1e6} million parameters")

filename = "albert_n32_paramcounts.pkl"
with open(filename, "wb") as f:
    pickle.dump(vals, f)
print(f"Done. Dumped outputs to {filename}")
