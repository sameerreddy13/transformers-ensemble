import torch

class Ensemble():
    def __init__(self, models, device):
        models = [model.to(device) for model in models] # TODO: split up across gpus    
        self.models = models
        self.device = device

class AverageVote(Ensemble):
    def __init__(self, models, device):
        super().__init__(models, device)

    def average_vote(self, input_ids, attention_mask, labels):
        '''
        Return ensmble predictions and accuracy
        '''
        all_preds = []
        for model in self.models:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            preds = outputs.logits.argmax(axis=-1)
            all_preds.append(preds)

        preds = torch.stack(all_preds).mode(dim=0).values
        return preds, (preds == labels).float().mean()

    def fit(self, dataloader):
        pass

    def predict_batch(self, example):
        '''
        Return predictions and accuracy on batch
        '''
        example = [x.to(self.device) for x in example]
        return self.average_vote(*example)

    def predict(self, dataloader):
        '''
        Return predictions and accuracy for all batches in dataloader
        '''
        results = [self.predict_batch(example) for example in dataloader]
        preds, accs = list(zip(*results))
        return torch.cat(preds), torch.tensor(accs)
