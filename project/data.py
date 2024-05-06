"""
List of Datasets:

1)arg: NC-Dataset (News Category Dataset) 
2)arg: SST2
3)arg: IMBD
4)arg: Yelp (Yelp Polarity) 

data.py downloads the above list of datasets depending on the argument and returns a dataframe.
"""
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPTextModel, AutoModel, CLIPModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch.nn.functional as F

class TextEncoder(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.name = model_name
        self.bert_model = AutoModel.from_pretrained(model_name)

    def forward(self, Xbatch):
        ber_out = self.bert_model(input_ids=Xbatch)
        return ber_out.last_hidden_state[:, 0, :]

class DocumentPreprocessor:
    def __init__(self, data, tokenizer, max_len=None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_len = max_len
        self.data = data
        self.tokens = self.tokenize_text(self.data)
        self.padded_tokens = self.padding(self.tokens)
        
    def padding(self, data):
        """
        Pad a given sentence based on the maximum length
        
        Argument:
        data: a list of instance (x, y, z, c), where x is the label, y is the sentence and z is the corresponding encoding

        Return:
        padded_text: a list of instance (x, y, z), where x is the label, y is the sentence, 
                     z is the corresponding encoding with padding
        """
        padded_text = [(x, y, pad_sequences(torch.tensor([z]), maxlen=self.max_len, padding='post', truncating='post')) 
                       for x,y,z in data]
        return padded_text
        
    def tokenize_text(self, data):
        """
        Tokenize the given sentence

        Argument:
        data: a list of x, [y], z, where x is the label, y is the list of all sentences in label x, 
              z is the no of sentences in label x.

        Return:
        tokenize_text: a list of instance (x, y, z), where x is the label, y is the sentence and z is the corresponding encoding
        """
    
        tokenize_text = [(data[label][0], data[label][1][i], self.tokenizer.encode(data[label][1][i])) for label in range(0,len(data))
                         for i in range(0,data[label][2])]
        return tokenize_text
        
    def token2text(self, tok):
        """
        Generate Text from tokens

        Argument: 
        tok: a list that contains token arrays 

        Return:
        text: list of text for the given tokens
        """
        text = [self.tokenizer.decode(token_array, skip_special_tokens=True) for token_array in tok]
        return text
    
def create_df(features, dataset):
    all_data={}
    for i in range(0,len(features)):
        data = dataset['train'][features[i]]
        all_data.update({features[i]: data})
    data = pd.DataFrame(all_data)
    return data

def prepare_data(data,dname):
    if dname == 'NC-Dataset':
        dataset = load_dataset("heegyu/news-category-dataset")
        features = list(dataset['train'].features.keys())
        data = create_df(features, dataset)
        data['text'] = data['headline'].astype(str)+'. '+ data['short_description']
        data = data.rename(columns={'category':'label'})
        data = data.drop(columns=data.columns.difference(['text', 'label']))

    elif dname == 'SST2':
        dataset = load_dataset("stanfordnlp/sst2")
        features = list(dataset['train'].features.keys())
        data = create_df(features, dataset)
        data = data.rename(columns={'sentence':'text'})
        data = data.drop(columns=data.columns.difference(['text', 'label']))
    
    elif dname == 'IMDB':
        dataset = load_dataset("stanfordnlp/imdb")
        features = list(dataset['train'].features.keys())
        data = create_df(features, dataset)
    
    elif dname == 'Yelp':
        dataset = load_dataset("yelp_polarity")
        features = list(dataset['train'].features.keys())
        data = create_df(features, dataset)

    else:
        print("Invalid Dataset Name!")
        
    data = data.sort_values(by='label', ascending=True)
    labels = data['label'].unique()
    text = [(label, data[data['label'] == label]['text'].values.tolist(), len(data[data['label'] == label]['text'].values.tolist())) 
            for label in labels]
    sorted_text = sorted(text, key=lambda x: x[2], reverse=True)
    
    if dname == 'NC-Dataset':
        id_data = sorted_text[0:7]
        ood_data = sorted_text[7:len(text)]

    else:
        id_data = [sorted_text[0]]
        ood_data = [sorted_text[1]]
    return id_data, ood_data


def dataset(dname, model_name, max_len):
    if model_name == 'Bert':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        name = 'bert-base-uncased'
    elif model_name == 'Roberta':
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        name = 'roberta-base'
    elif model_name == 'Xlm_Roberta':
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        name = 'xlm-roberta-base'
    
    elif model_name == 'CLIP':
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    else: 
        print("Invalid Encoder Name!")
   
    id_text, ood_text = prepare_data(dname)
    testing_preprocessor = DocumentPreprocessor(data = id_text, tokenizer=name, max_len=max_len)
    df = pd.DataFrame(testing_preprocessor.tokens, columns=['Label', 'Text', 'Tokens'])
    labels = list(set(map(lambda x: x[0], testing_preprocessor.tokens)))
    split_dfs = [group for _, group in df.groupby('Label')]
    batch={}
    for i in range(0,len(labels)):
        label = labels[i]
        data = split_dfs[i]['Tokens'].tolist()
        data = random.sample(data, len(data))
        data = np.concatenate(data)
        data = [torch.tensor(data[idk:idk+512]) for idk in range(0,len(data),256)]
        if len(data[-1]) == 256:
            data[-1] =  F.pad(data[-1], (0, 256), value=0)
        batch[label] = data
    label = labels[0]
    dataloader = DataLoader(batch[label], batch_size=32)
    model = TextEncoder(model_name='bert-base-uncased')
    emb = []
    with torch.no_grad():
        model.eval()
        model.to('cuda')
        for data in dataloader:
            mask = (data != 0).long().to('cuda')
            data = data.to('cuda')
            output = model(data,mask).cpu().detach()
            emb.append(output[0])
        model.to('cpu')
    return 0

