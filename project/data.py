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

class TextEncoder(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.name = model_name
        self.bert_model = AutoModel.from_pretrained(model_name)

    def forward(self, Xbatch, Xmask):
        ber_out = self.bert_model(input_ids=Xbatch, attention_mask=Xmask)
        return ber_out.last_hidden_state[:, 0, :]
    

def create_df(features, dataset):
    all_data={}
    for i in range(0,len(features)):
        data = dataset['train'][features[i]]
        all_data.update({features[i]: data})
    data = pd.DataFrame(all_data)
    return data

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

def get_dataloader(data, model_name):

    if model_name == 'Bert':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    elif model_name == 'Roberta':
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    elif model_name == 'Xlm_Roberta':
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    
    elif model_name == 'CLIP':
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    else: 
        print("Invalid Encoder Name!")
    
    return 0 

def main():
    parser = argparse.ArgumentParser(description="Create a Dataloader for the given Dataset")
    parser.add_argument("--dname", type=str, default='Amazon')
    parser.add_argument("--encoder", type=str, default='Bert')
    args = parser.parse_args()
    data = get_dataset(args.dname)
    data_loader = get_dataloader(data,args.encoder)
    return data

if __name__== "__main__":
    main()