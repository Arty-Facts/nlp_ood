from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from collections import defaultdict
import requests
import zipfile
import pathlib
import pandas as pd

class TextEncoder(nn.Module):
    def __init__(self, embed_model):
        super().__init__()
        self.name = embed_model
        self.model = AutoModel.from_pretrained(embed_model)

    def forward(self, Xbatch, Xmask):
        if self.name == 'openai/clip-vit-base-patch32':
            out = self.model.text_model(input_ids=Xbatch, attention_mask=Xmask)
        else:
            out = self.model(input_ids=Xbatch, attention_mask=Xmask)
        return out.last_hidden_state[:, 0, :]

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, encoder, tokenizer, token_len = None, device='cpu', name='TextDataset', cache=True, batch_size=32):
        self.tokenizer = tokenizer
        self.encoder = encoder
        if token_len is None or token_len < 0 or token_len > self.tokenizer.model_max_length:
            self.token_len = self.tokenizer.model_max_length
        else:
            self.token_len = token_len
        self.token_half = self.token_len//2
        if encoder.name == 'openai/clip-vit-base-patch32': # this is a hack to get the feature dimension for the clip model
            self.feat_dim = 512
            self.cls_token = self.tokenizer.convert_tokens_to_ids('<|startoftext|>')
            self.pad_token = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')
        else:
            self.feat_dim = encoder.model.config.hidden_size
            self.cls_token = self.tokenizer.cls_token_id
            self.pad_token = self.tokenizer.pad_token_id
        self.labels = data['label']
        # remove the cls_token from the encoded text
        self.tokens  = tokenizer(text=data['text'], add_special_tokens=False)['input_ids']
        self.index_info = []
        for line, t in enumerate(self.tokens):
            for item in range(self.token_half-1, len(t), self.token_half):
                self.index_info.append((line, item))
            if len(t) <= self.token_half:
                self.index_info.append((line, len(t)-1))
        self.device = device
        self.size = len(self.index_info)
        self.text = data['text']
        self.data_cache = None
        self.name = name    
        if cache:
            data_cache = torch.zeros(self.size, self.feat_dim)
            loader = torch.utils.data.DataLoader(self, batch_size=batch_size, collate_fn=self.collate_fn)
            with torch.no_grad():
                for i, emb in enumerate(loader):
                    data_cache[i*batch_size:(i+1)*batch_size] = emb.cpu()
            self.data_cache = data_cache

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
        return [self.cls_token] + self.tokens[line][start:end], self.labels[line], idx
    
    def get_text(self, idx):
        line, label, _ = self[idx]
        return label, self.tokenizer.decode(line[1:])

    def collate_fn(self, instances):
        X, Y, idxs = zip(*instances)
        if self.data_cache is not None:
            emb = self.data_cache[list(idxs)]
            emb.to(self.device)
        else:
            max_len = max(len(x) for x in X)
            Xpadded = torch.as_tensor([x + [self.pad_token]*(max_len-len(x)) for x in X]).to(self.device)
            with torch.no_grad():
                emb = self.encoder(Xpadded, (Xpadded != self.pad_token).long().to(self.device))
        return emb
    from collections import defaultdict

def download_and_unzip(url, extract_to='.'):
    """
    Downloads a zip file from the given URL and unzips it to the specified directory.

    Args:
    url (str): The URL of the zip file to download.
    extract_to (str): Directory to extract the contents of the zip file.
    """
    # Get the name of the file from the URL
    file_name = url.split('/')[-1]

    # Download the file
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {file_name}")

        # Unzip the file
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted {file_name} to {extract_to}")
    else:
        print("Failed to download the file")

AVAILABLE_DATASETS = ['IMDB', 'Yelp', 'SST2', 'NC-Top-Dataset', 'NC-Bottom-Dataset', 'Amazon-Review', 'Amazon-Music', 'Amazon-Books', 'Amazon-DVD', 'Amazon-Camera', 'Amazon-Health', 'Amazon-Software']

def id_dataset(id_name):
    if id_name == 'IMDB':
        dataset = load_dataset("stanfordnlp/imdb")
        label = ['neg', 'pos']
        labels = [label[l] for l in dataset['train']['label']]
        texts = dataset['train']['text']
        train = {'label':labels, 'text':texts}

        labels = [label[l] for l in dataset['test']['label']]
        texts = dataset['test']['text']
        eval = {'label':labels, 'text':texts}
    
    elif id_name == 'Yelp':
        dataset = load_dataset("yelp_polarity")
        label = ['neg', 'pos']
        labels = [label[l] for l in dataset['train']['label']]
        texts = dataset['train']['text']
        train = {'label':labels, 'text':texts}

        labels = [label[l] for l in dataset['test']['label']]
        texts = dataset['test']['text']
        eval = {'label':labels, 'text':texts}
    
    elif id_name == 'SST2':
        dataset = load_dataset("stanfordnlp/sst2")
        label = ['neg', 'pos']

        labels = [label[l] for l in dataset['train']['label']]
        texts = dataset['train']['sentence']
        train = {'label':labels, 'text':texts}

        labels = [label[l] for l in dataset['test']['label']]
        texts = dataset['test']['sentence']
        eval = {'label':labels, 'text':texts}

    elif id_name == 'Rotten-Tomatoes':
        dataset = load_dataset("rotten_tomatoes")
        label = ['neg','pos']
        labels = [label[l] for l in dataset['train']['label']]
        texts = dataset['train']['text']
        train = {'label':labels, 'text':texts}

        labels = [label[l] for l in dataset['test']['label']]
        texts = dataset['test']['text']
        eval = {'label':labels, 'text':texts}

    elif id_name == 'Fake-News':
        dataset = load_dataset("mrm8488/fake-news")
        label = ['fake', 'real']
        labels = [label[l] for l in dataset['train']['label']]
        texts = dataset['train']['text']
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=0, stratify=labels)
        train = {'label':train_labels, 'text':train_texts}
        eval = {'label':val_labels, 'text':val_texts}

    elif id_name == 'Only-Fake-News':
        dataset = load_dataset("mrm8488/fake-news")
        label = ['fake', 'real']
        labels = [label[l] for l in dataset['train']['label']]
        texts = dataset['train']['text']
        fake_labels, fake_texts=[],[]
        for i in range(0,len(labels)):
            if labels[i]=='fake':
                fake_labels.append(labels[i])
                fake_texts.append(texts[i])

        train_texts, val_texts, train_labels, val_labels = train_test_split(fake_texts, fake_labels, test_size=0.2, random_state=0)
        train = {'label':train_labels, 'text':train_texts}
        eval = {'label':val_labels, 'text':val_texts}
    
    elif id_name == 'Only-Real-News':
        dataset = load_dataset("mrm8488/fake-news")
        label = ['fake', 'real']
        labels = [label[l] for l in dataset['train']['label']]
        texts = dataset['train']['text']
        real_labels, real_texts=[],[]
        for i in range(0,len(labels)):
            if labels[i]=='real':
                real_labels.append(labels[i])
                real_texts.append(texts[i])
        train_texts, val_texts, train_labels, val_labels = train_test_split(real_texts, real_labels, test_size=0.2, random_state=0)
        train = {'label':train_labels, 'text':train_texts}
        eval = {'label':val_labels, 'text':val_texts}

    elif id_name == 'MNLI':
        mnli_dataset = load_dataset("glue", "mnli")
        label = ['entailment','neutral','contradiction']
        labels = [label[l] for l in mnli_dataset['train']['label']]
        premise = [p for p in mnli_dataset['train']['premise']]
        hypothesis = [f' [SEP] {h}' for h in mnli_dataset['train']['hypothesis']]
        texts = [x + y for x, y in zip(premise, hypothesis)]
        train = {'label':labels, 'text':texts}

        labels = [label[l] for l in mnli_dataset['validation_matched']['label']]
        premise = [p for p in mnli_dataset['validation_matched']['premise']]
        hypothesis = [f' [SEP] {h}' for h in mnli_dataset['validation_matched']['hypothesis']]
        texts = [x + y for x, y in zip(premise, hypothesis)]
        eval = {'label':labels, 'text':texts}

    elif id_name == 'SNLI':
        snli_dataset = load_dataset("stanfordnlp/snli")
        label = ['entailment','neutral','contradiction']
        labels = [label[l] for l in snli_dataset['train']['label']]
        premise = [p for p in snli_dataset['train']['premise']]
        hypothesis = [f' [SEP] {h}' for h in snli_dataset['train']['hypothesis']]
        texts = [x + y for x, y in zip(premise, hypothesis)]
        train = {'label':labels, 'text':texts}

        labels = [label[l] for l in snli_dataset['validation']['label']]
        premise = [p for p in snli_dataset['validation']['premise']]
        hypothesis = [f' [SEP] {h}' for h in snli_dataset['validation']['hypothesis']]
        texts = [x + y for x, y in zip(premise, hypothesis)]
        eval = {'label':labels, 'text':texts}
    
    elif id_name == 'MNLI-SNLI-OOD':
        snli_dataset = load_dataset("stanfordnlp/snli")
        label = ['entailment','neutral','contradiction']
        labels = [label[l] for l in snli_dataset['test']['label']]
        premise = [p for p in snli_dataset['test']['premise']]
        hypothesis = [f' [SEP] {h}' for h in snli_dataset['test']['hypothesis']]
        texts = [x + y for x, y in zip(premise, hypothesis)]
        train = {'label':labels, 'text':texts}

        mnli_dataset = load_dataset("glue", "mnli")
        labels = [label[l] for l in mnli_dataset['validation_mismatched']['label']]
        premise = [p for p in mnli_dataset['validation_mismatched']['premise']]
        hypothesis = [f' [SEP] {h}' for h in mnli_dataset['validation_mismatched']['hypothesis']]
        texts = [x + y for x, y in zip(premise, hypothesis)]
        eval = {'label':labels, 'text':texts}

    elif id_name == 'STS':
        dataset = load_dataset("sentence-transformers/stsb")
        labels = dataset['train']['score']
        s1 = [p for p in dataset['train']['sentence1']]
        s2 = [f' [SEP] {h}' for h in dataset['train']['sentence2']]
        texts = [x + y for x, y in zip(s1, s2)]
        train = {'label':labels, 'text':texts}

        labels = dataset['validation']['score']
        s1 = [p for p in dataset['validation']['sentence1']]
        s2 = [f' [SEP] {h}' for h in dataset['validation']['sentence2']]
        texts = [x + y for x, y in zip(s1, s2)]
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
            t, e = train_test_split(data, test_size=0.2, random_state=0)
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
            t, e = train_test_split(data, test_size=0.2, random_state=0)
            eval['label'] += [label]*len(e)
            eval['text'] += e
            train['label'] += [label]*len(t)
            train['text'] += t
    elif id_name == 'Amazon-Review':
        dataset = load_dataset("amazon_polarity")
        label = ['neg', 'pos']
        labels = [label[l] for l in dataset['train']['label']]
        texts = dataset['train']['text']
        train = {'label':labels, 'text':texts}

        labels = [label[l] for l in dataset['test']['label']]
        texts = dataset['test']['text']
        eval = {'label':labels, 'text':texts}
    elif id_name.startswith('Amazon-'):
        product = id_name.split('-')[1].lower()
        if product not in ['music', 'books', 'dvd', 'camera', 'health', 'software']:
            raise ValueError(f'Unknown Amazon product {product}, choose from music, books, dvd, camera, health, software')
        if not pathlib.Path('dredze_amazon_reviews.zip').exists():
            url = 'http://www.cse.chalmers.se/~richajo/waspnlp2024/dredze_amazon_reviews.zip'
            download_and_unzip(url)
        amazon_corpus = pd.read_csv('dredze_amazon_reviews.tsv', sep='\t', header=None, names=['product', 'sentiment', 'text'])
        data = amazon_corpus[amazon_corpus['product'] == product]
        train, eval = train_test_split(data, test_size=0.2, random_state=0)
        train = {'label':train['sentiment'].tolist(), 'text':train['text'].tolist()}
        eval = {'label':eval['sentiment'].tolist(), 'text':eval['text'].tolist()}
    return train, eval

def ood_dataset(ood_name):
    train, eval = id_dataset(ood_name)
    ood_data = {key: train[key] + eval[key] for key in train}
    return ood_data

def load_datasets(id=None, ood=None, embed_model=None, device='cpu', token_len = None, **kvargs):
    if embed_model is None:
        raise ValueError('embed_model must be specified')
    encoder = TextEncoder(embed_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(embed_model)
    ret = []
    if id is not None:
        if isinstance(id, str):
            train_data, eval_data = id_dataset(id)
            id_name = id
        else:
            trains, evals = zip(*[id_dataset(id_name) for id_name in id])
            train_data = {key : sum([t[key] for t in trains], []) for key in trains[0]}
            eval_data = {key : sum([e[key] for e in evals], []) for key in evals[0]}
            id_name = '_'.join(id)
        ret.append(TextDataset(train_data, encoder, tokenizer, token_len, device, id_name, **kvargs))
        ret.append(TextDataset(eval_data, encoder, tokenizer, token_len, device, id_name+'_eval', **kvargs))
    if ood is not None:
        if isinstance(ood, str):
            oods = [TextDataset(ood_dataset(ood), encoder, tokenizer, token_len, device, ood, **kvargs)]
        else:
            oods = [TextDataset(ood_dataset(ood_name), encoder, tokenizer, token_len, device, ood_name, **kvargs) for ood_name in ood]
        ret.extend(oods)
    return ret