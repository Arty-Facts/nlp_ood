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
import torch.nn as nn
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

    elif name == 'NC-Dataset':
        dataset = load_dataset("heegyu/news-category-dataset")
        features = list(dataset['train'].features.keys())
        data = create_df(features, dataset)

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
        print("Invalid Dataset Name")
    return data

def main():
    parser = argparse.ArgumentParser(description="Create Dataloader for a given Dataset")
    parser.add_argument("--name", type=str, default='Amazon')
    args = parser.parse_args()
    name = args.name
    data = get_dataset(name)
    return data

if __name__== "__main__":
    main()