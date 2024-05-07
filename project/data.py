from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from collections import defaultdict

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative = [len(d) for d in datasets]
        self.size = sum(len(d) for d in datasets)

    def to(self, device):
        for d in self.datasets:
            d.to(device)
        return self
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        offset = 0
        for d, i in enumerate(self.cumulative):
            if idx < i:
                return self.datasets[d][idx-offset]
            offset += i
        return None
            
    def get_text(self, idx):
        offset = 0
        for d, i in enumerate(self.cumulative):
            if idx < i:
                return self.datasets[d].get_text(idx-offset)
            offset += i
        return None
            
    def collate_fn(self, instances):
        return self.datasets[0].collate_fn(instances)

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
        self.tokens = [list(filter(lambda t: t != self.cls_token, self.tokenizer.encode(d))) for d in data['text']]
        self.index_info = [(line, item) for line, t in enumerate(self.tokens) for item in range(self.token_half-1, len(t)-self.token_half if len(t) > self.token_len else self.token_half, self.token_half)]
        self.device = device
        self.size = len(self.index_info)

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        return self

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        line, item = self.index_info[idx]
        # start self.token_half -1 tokens before item of at the beginning of the line, leave space for the cls token
        start = max(0, item-self.token_half-1)
        # end self.token_half tokens after item or at the end of the line
        end = min(len(self.tokens[line]), item+self.token_half)
        return [self.cls_token] + self.tokens[line][start:end], self.labels[line]
    
    def get_text(self, idx):
        line, item = self.index_info[idx]
        start = max(0, item-self.token_half-1)
        end = min(len(self.tokens[line]), item+self.token_half)
        return self.tokenizer.decode(self.tokens[line][start:end])

    def collate_fn(self, instances):
        X, Y = zip(*instances)
        max_len = max(len(x) for x in X)
        Xpadded = torch.as_tensor([x + [self.pad_token]*(max_len-len(x)) for x in X]).to(self.device)
        with torch.no_grad():
            emb = self.encoder(Xpadded, (Xpadded != self.pad_token).long().to(self.device))
        return emb, Y
    from collections import defaultdict

def id_dataset(id_name, embed_model, tokenizer, token_len, device):
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
    
    elif id_name == 'NC-Dataset':
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
    train_dataset = TextDataset(train, embed_model, tokenizer, token_len, device)
    eval_dataset = TextDataset(eval, embed_model, tokenizer, token_len, device)
    return train_dataset, eval_dataset

def ood_dataset(ood_name, embed_model, tokenizer, token_len, device):
    if ood_name == 'NC-Dataset':
        dataset = load_dataset("heegyu/news-category-dataset")
        split_topic = 7
        new_dataset = defaultdict(list)
        for head, desc, cat in zip(dataset['train']['headline'], dataset['train']['short_description'], dataset['train']['category']):
            new_dataset[cat].append(head + ': ' + desc)
        new_dataset = sorted(new_dataset.items(), key=lambda x: len(x[1]), reverse=True)
        ood = new_dataset[split_topic:]
        ood_data = {'label':[], 'text':[]}
        for label, data in ood:
            ood_data['label'] += [label]*len(data)
            ood_data['text'] += data
        return TextDataset(ood_data, embed_model, tokenizer, token_len, device)
    else:
        return ConcatDataset(id_dataset(ood_name, embed_model, tokenizer, token_len, device))

def load_datasets(id, ood, embed_model, device='cpu', token_len = None):
    encoder = TextEncoder(embed_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(embed_model)
    if isinstance(id, str):
        train, eval = id_dataset(id, encoder, tokenizer, token_len, device)
    else:
        train, eval = zip(*[id_dataset(id_name, encoder, tokenizer, token_len, device) for id_name in id])
        train = ConcatDataset(train)
        eval = ConcatDataset(eval)
    if isinstance(ood, str):
        ood = ood_dataset(ood, encoder, tokenizer, token_len, device)
    else:
        ood = ConcatDataset([ood_dataset(ood_name, encoder, tokenizer, token_len, device) for ood_name in ood])
    return train, eval, ood