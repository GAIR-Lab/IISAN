import os
import lmdb
import pickle
import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import ViTForImageClassification
import numpy as np
import torch.nn as nn
import gc

from tqdm import tqdm

class LMDB_Image:
    def __init__(self, image, id):
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()
        self.id = id

    def get_image(self):
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)
    

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
            item_id, item_title = line.strip().split('\t')
            texts[item_id] = item_title
    return texts

def load_image_data(lmdb_path, transform, item_ids):
    images = {}
    env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                    readonly=True, lock=False,
                    readahead=False, meminit=False)
    
    with env.begin() as txn:
        count = 0
        for item_id in tqdm(item_ids):
            byte_image = pickle.loads(txn.get(item_id.encode()))
            image = Image.fromarray(byte_image.get_image()).convert('RGB')
            images[item_id] = transform(image)
            count+=1
            # For a quick check if both processing and loading are working properly
#             if count >=500:
#                  break
    return images


def process_items(texts, images, bert_model, cv_model, tokenizer, batch_size=128):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    bert_model.to(device).eval()
    cv_model.to(device).eval()
    
    item_ids_list = list(texts.keys())
    num_batches = len(item_ids_list) // batch_size + (len(item_ids_list) % batch_size != 0)
    
    for batch_num in tqdm(range(num_batches)):
        bert_outputs = {}
        vit_outputs = {}
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        batch_item_ids = item_ids_list[start_idx:end_idx]
        

        
        # Process image batch
        image_batch = [images[item_id] for item_id in batch_item_ids]
        image_batch = torch.stack(image_batch).to(device)
        with torch.no_grad():
            vit_output = cv_model(image_batch)
        for i, item_id in enumerate(batch_item_ids):             
            hidden_states_for_item = [hidden_state[i][0,:] for hidden_state in vit_output.hidden_states]
            hidden_states_for_item = torch.stack(hidden_states_for_item)
            vit_outputs[item_id] = hidden_states_for_item.cpu()

        # Release memory
        save_outputs('stored_vectors_sci/vit_huge_outputs', vit_outputs, prefix='vit')
        torch.cuda.empty_cache()
        gc.collect()
        
    return bert_outputs, vit_outputs

def load_and_save():

    print("loading bert...")
    bert_model_load = '../pretrained_models/bert/bert_base_uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_load)
    config = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
    bert_model = BertModel.from_pretrained(bert_model_load, config=config)

    print("loading vit...")
    cv_model_load = 'google/vit-huge-patch14-224-in21k'
    cv_model = ViTForImageClassification.from_pretrained(cv_model_load, output_hidden_states=True)
    cv_model.classifier = nn.Identity()

    # load data
    # Take Scientific as an example
    text_file_path = '../Dataset/Scientific/Industrial_and_Scientific_items.tsv'
    lmdb_path = '../Dataset/Scientific/am_is.lmdb'
    print("loading text...")
    texts = load_text_data(text_file_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    print("loading image...")
    images = load_image_data(lmdb_path, transform, texts.keys())

    print("processing model output...")

    bert_outputs, vit_outputs = process_items(texts, images, bert_model, cv_model, tokenizer)



def test():

    item_id_to_test = b'B00KLMWHTE'  
    item_id_to_test = item_id_to_test.decode('utf-8')
    vit_output_loaded = load_output('stored_vectors_sci/vit_huge_outputs', item_id_to_test, prefix='vit')
    print(vit_output_loaded.shape)  # 输出加载的ViT输出
    
load_and_save()
test()