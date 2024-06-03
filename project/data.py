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
    def __init__(self, embed_model, num_classes=2, dropout=0.1):
        super().__init__()
        self.name = embed_model
        self.model = AutoModel.from_pretrained(embed_model)
        if embed_model == 'openai/clip-vit-base-patch32':
            self.feat_dim = 512
        else:
            self.feat_dim = self.model.config.hidden_size
        self.config = self.model.config 
        self.prepare_inputs_for_generation = self.model.prepare_inputs_for_generation
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(self.feat_dim, num_classes),
        )   

    def features(self, input_ids, attention_mask, **kwargs):
        if self.name == 'openai/clip-vit-base-patch32':
            out = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]
    
    def forward(self, input_ids, attention_mask, **kwargs):
        return self.head(self.features(input_ids, attention_mask, **kwargs))
    

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, encoder, tokenizer, token_len = None, device='cpu', name='TextEmbDataset', cache=True, batch_size=8, cache_dir='cache', fine_tune=False):
        self.tokenizer = tokenizer
        self.encoder = encoder
        if token_len is None or token_len < 0 or token_len > self.tokenizer.model_max_length:
            self.token_len = self.tokenizer.model_max_length
        else:
            self.token_len = token_len
        self.token_half = self.token_len//2
        if encoder.name == 'openai/clip-vit-base-patch32': # this is a hack to get the feature dimension for the clip model
            self.cls_token = self.tokenizer.convert_tokens_to_ids('<|startoftext|>')
            self.pad_token = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')
        else:
            self.cls_token = self.tokenizer.cls_token_id
            self.pad_token = self.tokenizer.pad_token_id
        self.feat_dim = encoder.feat_dim
        self.labels = data['label']
        self.unique_labels = sorted(list(set(self.labels)))
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
        self.cache_path = pathlib.Path(cache_dir)
        self.cache_path.mkdir(exist_ok=True, parents=True)
        self.tune_path = self.cache_path / f"{encoder.name.replace('/','_')}_lora"
        self.model_path = self.cache_path / f"{encoder.name.replace('/','_')}_lora.pt"

        if fine_tune:
            if self.model_path.exists():
                self.encoder.load_state_dict(torch.load(self.model_path))
                self.encoder = get_peft_model(self.encoder, lora_config)
                self.encoder.name = f"{self.encoder.name}_lora"
                
            else:
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
                from transformers import TrainingArguments, Trainer
                import bitsandbytes
                import evaluate

                accuracy = evaluate.load('accuracy')

                def compute_metrics(eval_pred):
                    logits, labels = eval_pred
                    predictions = logits.argmax(axis=-1)
                    print(logits.shape, predictions.shape, labels.shape)
                    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}
                            

                LORA_R = 8 # 512
                LORA_ALPHA = 8 # 1024
                LORA_DROPOUT = 0
                # Define LoRA Config
                lora_config = LoraConfig(
                                r = LORA_R, # the dimension of the low-rank matrices
                                lora_alpha = LORA_ALPHA, # scaling factor for the weight matrices
                                lora_dropout = LORA_DROPOUT, # dropout probability of the LoRA layers
                                bias="none",
                                task_type=TaskType.TOKEN_CLS,
                                # target_modules=["query_key_value"],
                )

                # Prepare int-8 model for training - utility function that prepares a PyTorch model for int8 quantization training. <https://huggingface.co/docs/peft/task_guides/int8-asr>
                self.encoder = prepare_model_for_kbit_training(self.encoder)
                # initialize the model with the LoRA framework
                self.encoder = get_peft_model(self.encoder, lora_config)
                print(f"Fine-tuning the model using LoRA regularization with R={LORA_R}, alpha={LORA_ALPHA}, and dropout={LORA_DROPOUT}")
                self.encoder.print_trainable_parameters()
                
                # define the training arguments first.
                EPOCHS = 3

                training_args = TrainingArguments(
                                    output_dir=self.tune_path,
                                    overwrite_output_dir=True,
                                    fp16=True, #converts to float precision 16 using bitsandbytes
                                    num_train_epochs=EPOCHS,
                                    load_best_model_at_end=True,
                                    logging_strategy="epoch",
                                    evaluation_strategy="epoch",
                                    save_strategy="epoch",
                                    save_total_limits=2
                                    # per_device_train_batch_size=batch_size,
                                    # per_device_eval_batch_size=batch_size,
                                    do_train=True,
                                    do_eval=True,
                                    
                )
                train_index, test_index = train_test_split(range(self.size), test_size=0.3, random_state=0)
                split_dataset = {
                    "train": torch.utils.data.Subset(self, train_index),
                    "test": torch.utils.data.Subset(self, test_index),
                }
                # training the model 
                trainer = Trainer(
                        model=self.encoder,
                        args=training_args,
                        train_dataset=split_dataset['train'],
                        eval_dataset=split_dataset["test"],
                        data_collator=self.tune_collate_fn,
                        compute_metrics=compute_metrics,
                )
                self.encoder.config.use_cache = False  # silence the warnings. Please re-enable for inference!
                trainer.train()
                # only saves the incremental 🤗 PEFT weights (adapter_model.bin) that were trained, meaning it is super efficient to store, transfer, and load.
                trainer.model.save_pretrained(self.tune_path)
                # save the full model and the training arguments
                trainer.save_model(self.tune_path)
                trainer.model.config.save_pretrained(self.tune_path)
                self.encoder.config.use_cache = True  # re-enable the cache for inference
                self.encoder.eval()
                self.encoder.name = f"{self.encoder.name}_lora"
                
                
                torch.save(self.encoder.state_dict(), self.model_path)
        self.cache_name = f'{name}_{encoder.name}_{self.token_len}_cache.pt'.replace('/','_')
        self.cache_path = self.cache_path / self.cache_name
        if cache:
            if self.cache_path.exists():
                self.data_cache = torch.load(self.cache_path)
            else:
                data_cache = torch.zeros(self.size, self.feat_dim)
                loader = torch.utils.data.DataLoader(self, batch_size=batch_size, collate_fn=self.collate_fn)
                with torch.no_grad():
                    for i, emb in enumerate(loader):
                        data_cache[i*batch_size:(i+1)*batch_size] = emb.cpu()
                self.data_cache = data_cache
                torch.save(data_cache, self.cache_path)
            

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
                emb = self.encoder.features(Xpadded, (Xpadded != self.pad_token).long().to(self.device))
        return emb
    
    def tune_collate_fn(self, instances):
        X, Y, idxs = zip(*instances)
        max_len = max(len(x) for x in X)
        Xpadded = torch.as_tensor([x + [self.pad_token]*(max_len-len(x)) for x in X])
        mask = (Xpadded != self.pad_token).long()
        return {"input_ids": Xpadded, "attention_mask": mask, "label": torch.tensor([self.unique_labels.index(y) for y in Y])}
        



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
    
    elif id_name == 'NC-Top-Dataset':
        split_topic = 5
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
        split_topic = 5
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
    tokenizer = AutoTokenizer.from_pretrained(embed_model, use_fast=True)
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