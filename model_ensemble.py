import torch
import torch.nn.functional as F


class Ensemble():
    def __init__(self, models, device):
        models = [model.to(device) for model in models] # TODO: split up across gpus    
        self.models = models
        self.device = device

    def fit(self, dataloader):
        pass

    def predict_batch(self, example):
        raise NotImplementedError()

    def predict(self, dataloader):
        '''
        Return predictions and accuracy for all batches in dataloader
        '''
        return [self.predict_batch(example)[1] for example in dataloader]

class AverageVote(Ensemble):
    '''
    Voting with all models equally weighted
    '''
    def __init__(self, models, device):
        super().__init__(models, device)

    @staticmethod
    def average_vote(models, num_epochs=None, input_ids, attention_mask, labels):
        '''
        Return ensemble predictions and accuracy
        '''
        if num_epochs is not None:
            print("WARNING: num_epochs argument is not used")

        all_preds = []
        for model in models:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            preds = outputs.logits.argmax(axis=-1)
            all_preds.append(preds)

        preds = torch.stack(all_preds).mode(dim=0).values
        return preds, (preds == labels).float().mean()

    def predict_batch(self, example):
        '''
        Return predictions and accuracy on batch
        '''
        example = [x.to(self.device) for x in example]
        return self.average_vote(self.models, *example)


class WeightedVote(Ensemble):
    '''
    Voting with learned weights per model
    '''
    def __init__(self, models, device):
        super().__init__(models, device)
        self.w = torch.nn.Parameter(torch.full(
            size=(1, len(self.models), 1), fill_value=1 / len(self.models),
            device=device
        ))

    def fit(self, dataloader, num_epochs=1, lr=1.0, print_freq=25):
        optimizer = torch.optim.SGD([self.w], lr=lr)
        accs = []
        for epoch in range(num_epochs):
            for i, example in enumerate(dataloader):
                example = [x.to(self.device) for x in example]
                probs, acc = self.predict_batch(example)
                loss = F.cross_entropy(input=probs, target=example[2])
                accs.append(acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % print_freq == 0:
                    acc, accs = sum(accs) / len(accs), []
                    _, voting_acc = AverageVote.average_vote(self.models, *example)
                    print(f"[Epoch {epoch}] [{i}/{len(dataloader)}] Average accuracy so far: "
                          f"{acc} (baseline voting accuracy: {voting_acc}, loss = {loss.item()})")

        print(f"Done. Final weights: {self.w.data.squeeze()}")

    def predict_batch(self, example):
        example = [x.to(self.device) for x in example]
        logits = torch.stack([  # Shape: (batch_size, num_models, num_labels)
            model(input_ids=example[0], attention_mask=example[1]).logits
            for model in self.models
        ]).permute(1, 0, 2)

        probs = (logits * self.w).sum(dim=1).softmax(dim=-1)  # Sum over num_models dim
        acc = (probs.argmax(dim=-1) == example[2]).float().mean().item()

        return probs, acc


class DynamicWeightedVote(Ensemble):
    '''
    Voting with learned weights per model, where the weights depend on the data point.
    '''
    def __init__(self, models, device, input_dim=128, hidden_dim=200, num_hidden_layers=2):
        super().__init__(models, device)

        layers = [
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
        ]
        for _ in range(num_hidden_layers):
            layers.extend([
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            ])
        layers.append(torch.nn.Linear(hidden_dim, len(models)))

        self.mlp = torch.nn.Sequential(*layers).to(device)

    def fit(self, dataloader, num_epochs=1, lr=1e-2, print_freq=25):
        optimizer = torch.optim.SGD(self.mlp.parameters(), lr=lr)
        accs = []
        for epoch in range(num_epochs):
            for i, example in enumerate(dataloader):
                example = [x.to(self.device) for x in example]
                probs, acc = self.predict_batch(example)
                loss = F.cross_entropy(input=probs, target=example[2])
                accs.append(acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % print_freq == 0:
                    acc, accs = sum(accs) / len(accs), []
                    _, voting_acc = AverageVote.average_vote(self.models, *example)
                    print(f"[Epoch {epoch}] [{i}/{len(dataloader)}] Average accuracy so far: "
                          f"{acc} (baseline voting accuracy: {voting_acc}, loss = {loss.item()})")

    def predict_batch(self, example):
        example = [x.to(self.device) for x in example]
        logits = torch.stack([  # Shape: (batch_size, num_models, num_labels)
            model(input_ids=example[0], attention_mask=example[1]).logits
            for model in self.models
        ]).permute(1, 0, 2)
        w = self.mlp(example[0].float())  # Shape: (batch_size, num_models)

        probs = (logits * w.unsqueeze(-1)).sum(dim=1).softmax(dim=-1)
        acc = (probs.argmax(dim=-1) == example[2]).float().mean().item()

        return probs, acc
