"""
List of Datasets:

1)arg: NC-Dataset (News Category Dataset) 
2)arg: SST2
3)arg: IMBD
4)arg: Yelp (Yelp Polarity) 
5)arg: Amazon (Amazon Review)

data.py downloads the above list of datasets depending on the argument and returns a dataframe.
"""

import pathlib
import argparse
import wget
import zipfile
import pandas as pd
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPTextModel, AutoModel, CLIPModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
        

def create_df(features, dataset):
    all_data={}
    for i in range(0,len(features)):
        data = dataset['train'][features[i]]
        all_data.update({features[i]: data})
    data = pd.DataFrame(all_data)
    return data



def prepare_data(data,dname):
    data = data.sort_values(by='label', ascending=True)
    labels = data['label'].unique()
    text = [(label, data[data['label'] == label]['text'].values.tolist(), len(data[data['label'] == label]['text'].values.tolist())) 
            for label in labels]
    sorted_text = sorted(text, key=lambda x: x[2], reverse=True)
    if dname == 'NC-Dataset':
        id_data = sorted_text[0:7]
        ood_data = sorted_text[7:len(text)]
    return id_data, ood_data

def get_dataset(name):
    if name == 'Amazon':
        if pathlib.Path('dredze_amazon_reviews.zip').exists():
            print("Already downloaded")
        else:
            print("Downloading Amazon Reviews Dataset")
            url = 'http://www.cse.chalmers.se/~richajo/waspnlp2024/dredze_amazon_reviews.zip'
            filename = wget.download(url)
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall()
        data = pd.read_csv('dredze_amazon_reviews.tsv', sep='\t', header=None, names=['product', 'sentiment', 'text'])
        data = data.rename(columns={'product':'label'})
        data = data.drop(columns=data.columns.difference(['text', 'label']))
        
    elif name == 'NC-Dataset':
        dataset = load_dataset("heegyu/news-category-dataset")
        features = list(dataset['train'].features.keys())
        data = create_df(features, dataset)
        data['text'] = data['headline'].astype(str)+'. '+ data['short_description']
        data = data.rename(columns={'category':'label'})
        data = data.drop(columns=data.columns.difference(['text', 'label']))

    elif name == 'SST2':
        dataset = load_dataset("stanfordnlp/sst2")
        features = list(dataset['train'].features.keys())
        data = create_df(features, dataset)
    
    elif name == 'IMDB':
        dataset = load_dataset("stanfordnlp/imdb")
        features = list(dataset['train'].features.keys())
        data = create_df(features, dataset)
    
    elif name == 'Yelp':
        dataset = load_dataset("yelp_polarity")
        features = list(dataset['train'].features.keys())
        data = create_df(features, dataset)

    else:
        print("Invalid Dataset Name!")
    return data

def get_datastream(data, model_name):

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
   
    id_text, ood_text = prepare_data(data,'NC-Dataset')
    testing_preprocessor = DocumentPreprocessor(data = id_text, tokenizer=name, max_len=512)
    return 0

def main():
    parser = argparse.ArgumentParser(description="Create a Dataloader for the given Dataset")
    parser.add_argument("--dname", type=str, default='Amazon')
    parser.add_argument("--encoder", type=str, default='Bert')
    args = parser.parse_args()
    data = get_dataset(args.dname)
    data_loader = get_datastream(data,args.encoder)
    return data

if __name__== "__main__":
    main()