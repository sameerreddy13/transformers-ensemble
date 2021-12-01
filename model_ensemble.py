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
        results = [self.predict_batch(example) for example in dataloader]
        preds, accs = list(zip(*results))
        return torch.cat(preds), torch.tensor(accs)

class AverageVote(Ensemble):
    '''
    Voting with all models equally weighted
    '''
    def __init__(self, models, device):
        super().__init__(models, device)

    def average_vote(self, input_ids, attention_mask, labels):
        '''
        Return ensemble predictions and accuracy
        '''
        all_preds = []
        for model in self.models:
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
        return self.average_vote(*example)


class WeightedVote(Ensemble):
    '''
    Voting with learned weights per model
    '''
    def __init__(self, models, device):
        super().__init__(models, device)
        self.w = torch.nn.Parameter(
            torch.full(len(self))
            torch.ones(len(self.models), 1, device=device) / len(self.models)
        )

    def fit(self, dataloader, lr=1e-1):
        # optimizer = torch.optim.AdamW(lr=)
        optimizer = torch.optim.SGD([self.w], lr=lr)
        for example in dataloader:
            example = [x.to(self.device) for x in example]
            logits = torch.stack([
                model(input_ids=example[0], attention_mask=example[1]).logits
                for model in self.models
            ]).permute(1, 0, 2)
            import pdb; pdb.set_trace() 
            probs = self.w.dot(logits).softmax(-1)
            labels = example[2]
            loss = F.cross_entropy(input=probs, target=labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




        