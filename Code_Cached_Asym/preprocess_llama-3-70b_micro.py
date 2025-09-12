import os
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig
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
            try:
                item_id, item_title = line.strip().split('\t')
            except:
                
                item_id = line.strip().split('\t')[0]
                print(item_id)
                item_title=""
            texts[item_id] = item_title
    return texts

def manual_padding(encoded_texts, max_length, pad_token_id=0):
    padded_texts = []
    for tokens in encoded_texts:
        padded_length = max_length - len(tokens)
        if padded_length > 0:
            tokens.extend([pad_token_id] * padded_length)
        else:
            tokens = tokens[:max_length]
        padded_texts.append(tokens)
    return padded_texts

def process_items(texts, model, tokenizer, max_length=30, batch_size=128):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    model.to(device).eval()
    
    item_ids_list = list(texts.keys())
    num_batches = len(item_ids_list) // batch_size + (len(item_ids_list) % batch_size != 0)
    
    bert_outputs = {}
    for batch_num in tqdm(range(num_batches)):
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        batch_item_ids = item_ids_list[start_idx:end_idx]
        
        text_batch = [texts[item_id] for item_id in batch_item_ids]
        encoded_input = [tokenizer.encode(text, add_special_tokens=True) for text in text_batch]
        encoded_input = manual_padding(encoded_input, max_length)
        input_tensor = torch.tensor(encoded_input).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_tensor)
            hidden_states = outputs.hidden_states

            for i, item_id in enumerate(batch_item_ids):
                # Compute average of all layers' token embeddings for each item
                all_layers_embedding = torch.stack([torch.mean(layer[i], dim=0) for layer in hidden_states])
                bert_outputs[item_id] = all_layers_embedding.cpu()

        # Release memory
        save_outputs('stored_vectors_micro/llama70b_GPTQ_embeddings', bert_outputs, prefix='llama')
        torch.cuda.empty_cache()
        del input_tensor, encoded_input, text_batch, outputs
        torch.cuda.empty_cache()

    return bert_outputs

def load_and_save():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("loading LLaMA...")
    auth_token = "<your_token>"
    config = AutoConfig.from_pretrained("TechxGenus/Meta-Llama-3-70B-GPTQ",disable_exllama=True, output_hidden_states=True,token=auth_token)
    llama_tokenizer = AutoTokenizer.from_pretrained("TechxGenus/Meta-Llama-3-70B-GPTQ", token=auth_token)
    llama_model = LlamaForCausalLM.from_pretrained("TechxGenus/Meta-Llama-3-70B-GPTQ", config=config, token=auth_token)

    text_file_path = '../Dataset/Microlens/MicroLens-100k_title_en.tsv'
    print("loading text...")
    texts = load_text_data(text_file_path)

    print("processing model output...")
    llama_outputs = process_items(texts, llama_model, llama_tokenizer)


    print("read output...")

def test():
    item_id_to_test = 'B00KLMWHTE'
    llama_output_loaded = load_output('stored_vectors_micro/llama70b_GPTQ_embeddings', item_id_to_test, prefix='llama')
    print(llama_output_loaded.shape)  # 输出加载的LLaMA输出

load_and_save()

