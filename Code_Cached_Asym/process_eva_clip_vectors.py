import os
import lmdb
import pickle
import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AutoModel, AutoConfig  # 新增，用于加载 EVA-CLIP-18B
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

# 修改 load_image_data：若对应的 .pt 文件已存在，则跳过加载该图片
def load_image_data(lmdb_path, transform, item_ids, processed_dir, prefix='eva'):
    images = {}
    env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                    readonly=True, lock=False,
                    readahead=False, meminit=False)
    
    with env.begin() as txn:
        for item_id in tqdm(item_ids):
            output_file = os.path.join(processed_dir, f"{prefix}_{item_id}.pt")
            if os.path.exists(output_file):
                continue  # 如果 .pt 文件存在，则跳过加载该图片
            
            byte_image = pickle.loads(txn.get(item_id.encode()))
            image = Image.fromarray(byte_image.get_image()).convert('RGB')
            images[item_id] = transform(image)
        if len(images)==0:
            print("no images need to be processed.")
    return images

def process_items(texts, images, bert_model, cv_model, tokenizer, batch_size=128):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    bert_model.to(device).eval()
    cv_model.to(device).eval()
    
    # 这里仅处理 load_image_data 返回的图片，对应的 item_id 已排除了存在 .pt 文件的项
    item_ids_list = list(images.keys())
    num_batches = len(item_ids_list) // batch_size + (len(item_ids_list) % batch_size != 0)
    
    for batch_num in tqdm(range(num_batches)):
        eva_outputs = {}
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        batch_item_ids = item_ids_list[start_idx:end_idx]
        
        if not batch_item_ids:
            continue

        # 处理图像批次
        image_batch = [images[item_id] for item_id in batch_item_ids]
        image_batch = torch.stack(image_batch).to(device).half()
        with torch.no_grad():
            eva_output = cv_model.vision_model(pixel_values=image_batch, output_hidden_states=True)

        for i, item_id in enumerate(batch_item_ids):             
            # 提取每个隐藏层中第一个 token 的 hidden_state
            hidden_states_for_item = [hidden_state[i][0, :] for hidden_state in eva_output.hidden_states]
            hidden_states_for_item = torch.stack(hidden_states_for_item)
            eva_outputs[item_id] = hidden_states_for_item.cpu()

        # 保存当前批次的输出
        save_outputs('stored_vectors_sci/eva_clip_18b_outputs', eva_outputs, prefix='eva')
        torch.cuda.empty_cache()
        gc.collect()
        
    return eva_outputs

def load_and_save():
    print("loading bert...")
    bert_model_load = '../pretrained_models/bert/bert_base_uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_load)
    config = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
    bert_model = BertModel.from_pretrained(bert_model_load, config=config)

    print("loading eva clip...")
    cv_model_load = 'BAAI/EVA-CLIP-18B'
    cv_config = AutoConfig.from_pretrained(cv_model_load, output_hidden_states=True)
    cv_model = AutoModel.from_pretrained(cv_model_load, torch_dtype=torch.float16, trust_remote_code=True, output_hidden_states=True)

    # 注意：对于 EVA-CLIP-18B，不需要对分类器部分做修改

    # 加载数据
    text_file_path = '../Dataset/Scientific/Industrial_and_Scientific_items.tsv'
    lmdb_path = '../Dataset/Scientific/am_is.lmdb'
    print("loading text...")
    texts = load_text_data(text_file_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    processed_dir = 'stored_vectors_sci/eva_clip_18b_outputs'
    print("loading image...")
    # 只加载那些未生成 .pt 文件的图片
    images = load_image_data(lmdb_path, transform, texts.keys(), processed_dir, prefix='eva')

    print("processing model output...")
    eva_outputs = process_items(texts, images, bert_model, cv_model, tokenizer)

def test():
    item_id_to_test = b'B00KLMWHTE'
    item_id_to_test = item_id_to_test.decode('utf-8')
    eva_output_loaded = load_output('stored_vectors_sci/eva_clip_18b_outputs', item_id_to_test, prefix='eva')
    print(eva_output_loaded.shape)  # 输出加载的 EVA-CLIP-18B 的 hidden_states shape
    
#load_and_save()
test()
