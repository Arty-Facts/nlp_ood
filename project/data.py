from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from collections import defaultdict

class TextEncoder(nn.Module):
    def __init__(self, embed_model):
        super().__init__()
        self.name = embed_model
        self.model = AutoModel.from_pretrained(embed_model)

    def forward(self, Xbatch, Xmask):
        ber_out = self.model(input_ids=Xbatch, attention_mask=Xmask)
        return ber_out.last_hidden_state[:, 0, :]

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, encoder, tokenizer, token_len = None, device='cpu'):
        self.tokenizer = tokenizer
        self.encoder = encoder
        if token_len is None or token_len < 0 or token_len > self.tokenizer.model_max_length:
            self.token_len = self.tokenizer.model_max_length
        else:
            self.token_len = token_len
        self.token_half = self.token_len//2
        self.cls_token = self.tokenizer.cls_token_id
        self.pad_token = self.tokenizer.pad_token_id
        self.labels = data['label']
        # remove the cls_token from the encoded text
        self.tokens  = tokenizer(text=data['text'], add_special_tokens=False)['input_ids']
        self.index_info = [(line, item) for line, t in enumerate(self.tokens) for item in range(self.token_half-1, len(t)-self.token_half if len(t) > self.token_len else self.token_half, self.token_half)]
        self.device = device
        self.size = len(self.index_info)
        self.data_text = data['text']

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        return self

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        line, item = self.index_info[idx]
        # start self.token_half -1 tokens before item of at the beginning of the line, leave space for the cls token
        start = max(0, item-self.token_half+1)
        # end self.token_half tokens after item or at the end of the line
        end = min(len(self.tokens[line]), item+self.token_half)
        return [self.cls_token] + self.tokens[line][start:end], self.labels[line]
    
    def get_text(self, idx):
        line, item = self.index_info[idx]
        start = max(0, item-self.token_half-1)
        end = min(len(self.tokens[line]), item+self.token_half)
        return self.labels[line], self.tokenizer.decode(self.tokens[line][start:end])

    def collate_fn(self, instances):
        X, Y = zip(*instances)
        max_len = max(len(x) for x in X)
        Xpadded = torch.as_tensor([x + [self.pad_token]*(max_len-len(x)) for x in X]).to(self.device)
        with torch.no_grad():
            emb = self.encoder(Xpadded, (Xpadded != self.pad_token).long().to(self.device))
        return emb, Y
    from collections import defaultdict

def id_dataset(id_name):
    if id_name == 'IMDB':
        dataset = load_dataset("stanfordnlp/imdb")
        label = ['negative', 'positive']
        labels = [label[l] for l in dataset['train']['label']]
        texts = dataset['train']['text']
        train = {'label':labels, 'text':texts}

        labels = [label[l] for l in dataset['test']['label']]
        texts = dataset['test']['text']
        eval = {'label':labels, 'text':texts}
    
    elif id_name == 'Yelp':
        dataset = load_dataset("yelp_polarity")
        label = ['negative', 'positive']
        labels = [label[l] for l in dataset['train']['label']]
        texts = dataset['train']['text']
        train = {'label':labels, 'text':texts}

        labels = [label[l] for l in dataset['test']['label']]
        texts = dataset['test']['text']
        eval = {'label':labels, 'text':texts}
    
    elif id_name == 'SST2':
        dataset = load_dataset("stanfordnlp/sst2")
        label = ['negative', 'positive']

        labels = [label[l] for l in dataset['train']['label']]
        texts = dataset['train']['sentence']
        train = {'label':labels, 'text':texts}

        labels = [label[l] for l in dataset['test']['label']]
        texts = dataset['test']['sentence']
        eval = {'label':labels, 'text':texts}
    
    elif id_name == 'NC-Top-Dataset':
        split_topic = 7
        dataset = load_dataset("heegyu/news-category-dataset")
        new_dataset = defaultdict(list)
        for head, desc, cat in zip(dataset['train']['headline'], dataset['train']['short_description'], dataset['train']['category']):
            new_dataset[cat].append(head + ': ' + desc)
        new_dataset = sorted(new_dataset.items(), key=lambda x: len(x[1]), reverse=True)
        id = new_dataset[:split_topic]
        
        eval, train = {'label':[], 'text':[]}, {'label':[], 'text':[]}
        for label, data in id:
            t, e = train_test_split(data, test_size=0.2)
            eval['label'] += [label]*len(e)
            eval['text'] += e
            train['label'] += [label]*len(t)
            train['text'] += t
    elif id_name == 'NC-Bottom-Dataset':
        split_topic = 7
        dataset = load_dataset("heegyu/news-category-dataset")
        new_dataset = defaultdict(list)
        for head, desc, cat in zip(dataset['train']['headline'], dataset['train']['short_description'], dataset['train']['category']):
            new_dataset[cat].append(head + ': ' + desc)
        new_dataset = sorted(new_dataset.items(), key=lambda x: len(x[1]), reverse=True)
        id = new_dataset[split_topic:]
        
        eval, train = {'label':[], 'text':[]}, {'label':[], 'text':[]}
        for label, data in id:
            t, e = train_test_split(data, test_size=0.2)
            eval['label'] += [label]*len(e)
            eval['text'] += e
            train['label'] += [label]*len(t)
            train['text'] += t
    return train, eval

def ood_dataset(ood_name):
    train, eval = id_dataset(ood_name)
    ood_data = {key: train[key] + eval[key] for key in train}
    return ood_data

def load_datasets(id, ood, embed_model, device='cpu', token_len = None):
    encoder = TextEncoder(embed_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(embed_model)
    if isinstance(id, str):
        train_data, eval_data = id_dataset(id)
    else:
        trains, evals = zip(*[id_dataset(id_name) for id_name in id])
        train_data = {key : sum([t[key] for t in trains], []) for key in trains[0]}
        eval_data = {key : sum([e[key] for e in evals], []) for key in evals[0]}
    if isinstance(ood, str):
        ood_data = ood_dataset(ood)
    else:
        oods = [ood_dataset(ood_name) for ood_name in ood]
        ood_data = {key : sum([o[key] for o in oods], []) for key in oods[0]}
    train_data = TextDataset(train_data, encoder, tokenizer, token_len, device)
    eval_data = TextDataset(eval_data, encoder, tokenizer, token_len, device)
    ood_data = TextDataset(ood_data, encoder, tokenizer, token_len, device)
    return train_data, eval_data, ood_data