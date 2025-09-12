import os
import lmdb
import pickle
import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer, BertModel, BertConfig
#from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import ViTForImageClassification
import numpy as np
import torch.nn as nn
import gc

from tqdm import tqdm
    

def save_outputs(directory, outputs, prefix=''):
    os.makedirs(directory, exist_ok=True)
    for item_id, output in outputs.items():
        file_path = os.path.join(directory, f"{prefix}_{item_id}.pt")
        torch.save(output, file_path)

def load_output(directory, item_id, prefix=''):
    file_path = os.path.join(directory, f"{prefix}_{item_id}.pt")
    if os.path.exists(file_path):
        return torch.load(file_path)
    else:
        return None


def load_text_data(file_path):
    texts = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            item_id = line[0]
            print(item_id)
            if len(line)==1:
                item_title = ""
            else:
                item_title = line[1]
            texts[item_id] = item_title
    return texts


def process_items(texts, bert_model, tokenizer, batch_size=128):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bert_model.to(device).eval()
    
    item_ids_list = list(texts.keys())
    num_batches = len(item_ids_list) // batch_size + (len(item_ids_list) % batch_size != 0)
    
    for batch_num in tqdm(range(num_batches)):
        bert_outputs = {}
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        batch_item_ids = item_ids_list[start_idx:end_idx]
        
        # Process text batch
        text_batch = [texts[item_id] for item_id in batch_item_ids]
        encoded_input = tokenizer(text_batch, max_length=30, padding='max_length', truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            bert_output = bert_model(**encoded_input)
            
        for i, item_id in enumerate(batch_item_ids):   
            hidden_states_for_item = [hidden_state[i][0,:] for hidden_state in bert_output.hidden_states]
            hidden_states_for_item = torch.stack(hidden_states_for_item)
            bert_outputs[item_id] = hidden_states_for_item.cpu()

       

        # Release memory
        save_outputs('stored_vectors_sci/bert_large_outputs', bert_outputs, prefix='bert')
        torch.cuda.empty_cache()
        del encoded_input, text_batch, bert_output
        gc.collect()
        
    return bert_outputs

def load_and_save():

    print("loading bert...")
    bert_model_load = '../pretrained_models/bert/bert_large_uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_load)
    config = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
    bert_model = BertModel.from_pretrained(bert_model_load, config=config)
    
#     bert_model_load = "meta-llama/Meta-Llama-3-8B"
#     llama2_tokenizer = LlamaTokenizer.from_pretrained(bert_model_load)
#     llama2_model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf")


    # load data
    # Take Scientific as an example
    text_file_path = '../Dataset/Scientific/Industrial_and_Scientific_items.tsv'
    # lmdb_path = '../Dataset/Scientific/am_is.lmdb'
    # text_file_path = '../Dataset/Microlens/MicroLens-100k_title_en.tsv'
    print("loading text...")
    texts = load_text_data(text_file_path)


    print("processing model output...")

    bert_outputs = process_items(texts, bert_model, tokenizer)


def test():

    item_id_to_test = b'10'  
    item_id_to_test = item_id_to_test.decode('utf-8')
    bert_output_loaded = load_output('stored_vectors_sci/bert_large_outputs', item_id_to_test, prefix='bert')

    print(bert_output_loaded.shape)  # 输出加载的BERT输出
    
load_and_save()
test()