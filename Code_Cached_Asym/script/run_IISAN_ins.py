import os
import torch

root_data_dir = '../'
dataset = '../Dataset/Instrument'
behaviors = 'am_Musical_Instruments_users_10K.tsv'
images = 'Musical_Instruments_items.tsv'
news = 'Musical_Instruments_items.tsv'
lmdb_data = 'am_mi.lmdb'
logging_num = 4
testing_num = 1

CV_resize = 224
CV_model_load = 'vit'
bert_model_load = 'bert_large_uncased'
freeze_paras_before = 0

stored_vector_path = "../stored_vectors_ins/"
cached_image_model = "vit_outputs"
cached_text_model = "llama70b_GPTQ_embeddings" #llama70b_GPTQ_embeddings llama2_embeddings llama2_embeddings llama_it_embeddings mistral_7b_embeddings mistral_8x7B_it_embeddings
mode = 'train'
item_tower = 'modal'

epoch = 100
load_ckpt_name = 'None'
pretrained_recsys_model = 'None'

adapter_type = "IISAN" 
drop_rate_list = [0.1]
batch_size_list = [64]
lr_list = [2e-4]
embedding_dim_list = [64]
adapter_cv_lr_list = [1e-4] #4e4
adapter_bert_lr_list = [1e-4] #4e4
adding_adapter_to_list = ['all']
#None or all
fine_tune_to_list = ['None']
# None or TRUE
finetune_layernorm = "None"
cv_adapter_down_size = 64
bert_adapter_down_size = 128
# None or True
fine_tune_lr_image_list = [1e-4]#1e-4
fine_tune_lr_text_list = [5e-5]#1e-4
seed_list = [12345]
em_dim_list = [64]
l2_weight_list=[0]
#0,1,2,3,4,5,6,7,8,9,10,11 
# "1,3,5,7,9,11,13,15,17,19,21,23","13,15,17,19,21,23","3,7,11,15,19,23"
# "1,7,13,19,25,31" -> for llama-7b and 8b
# "2,7,12,17,22,27" -> for gemma-7b
# "4,19,34,49,64,79" -> for llama-70b
# "5,15,25,35,45,55"
# 
side_adapter_bert_list_ls = ["4,19,34,49,64,79"]
side_adapter_cv_list_ls = ["1,3,5,7,9,11"]
# side_adapter_bert_list = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23"
# side_adapter_cv_list = "0,1,2,3,4,5,6,7,8,9,10,11"
side_adapter_mm_list = "1,3,5,7,9,11"
modality = "intra_inter"
text_layers = 80 # 32 for mistral_7b 80 for llama_70b 56 layers
text_embedding_dim = 8192 # 4096 for llama-7b 3072 for gemma-7b, 2048 for gemma-2b 8192 for llama-70b 4096 for mistral_7b_embeddings 6144 for mistral
cached_text_prefix = "llama"#llama gemma mistral


for side_adapter_bert_list in side_adapter_bert_list_ls:
    for side_adapter_cv_list in side_adapter_cv_list_ls:
        for em_dim in em_dim_list:
            cv_adapter_down_size = em_dim
            bert_adapter_down_size = em_dim
            for seed in seed_list:
                for adapter_bert_lr in adapter_bert_lr_list:
                    for adapter_cv_lr in adapter_cv_lr_list:
                        for adding_adapter_to in adding_adapter_to_list:
                            for fine_tune_to in fine_tune_to_list:
                                for l2_weight in l2_weight_list:
                                    for batch_size in batch_size_list:
                                        for drop_rate in drop_rate_list:
                                            for lr in lr_list:
                                                for embedding_dim in embedding_dim_list:
                                                    for fine_tune_lr_image in fine_tune_lr_image_list:
                                                        for fine_tune_lr_text in fine_tune_lr_text_list:
                                                            label_screen = '{}_bs{}_ed{}_lr{}_dp{}_L2{}_Flr{}_{}'.format(
                                                                item_tower, batch_size, embedding_dim, lr,
                                                                drop_rate, l2_weight, fine_tune_lr_image,fine_tune_lr_text)
                                                            run_py = "CUDA_VISIBLE_DEVICES='2' \
                                                                     torchrun --nproc_per_node 1 --master_port 1229\
                                                                     ../run.py --root_data_dir {}  --dataset {} --behaviors {} --images {} --news {}  --lmdb_data {}\
                                                                     --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                                                     --l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
                                                                     --CV_resize {} --CV_model_load {}  --epoch {} --freeze_paras_before {}  --fine_tune_lr_image {} --fine_tune_lr_text {} --pretrained_recsys_model {}\
                                                                     --adapter_cv_lr {} --adding_adapter_to {} --fine_tune_to {}\
                                                                     --finetune_layernorm {}  --cv_adapter_down_size {} --bert_adapter_down_size {} --adapter_bert_lr {} --bert_model_load {} --adapter_type {} --seed {} --side_adapter_vit_list {} --side_adapter_bert_list {} --side_adapter_mm_list {} --stored_vector_path {} --cached_image_model {} --cached_text_model {} --text_layers {} --text_embedding_dim {} --cached_text_prefix {}\
                                                                     ".format(
                                                                root_data_dir, dataset, behaviors, images,news, lmdb_data,
                                                                mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
                                                                l2_weight, drop_rate, batch_size, lr, embedding_dim,
                                                                CV_resize, CV_model_load, epoch, freeze_paras_before, fine_tune_lr_image,fine_tune_lr_text,pretrained_recsys_model,adapter_cv_lr,adding_adapter_to
                                                                             ,fine_tune_to,finetune_layernorm,cv_adapter_down_size,bert_adapter_down_size,adapter_bert_lr,bert_model_load,adapter_type,seed,side_adapter_cv_list,side_adapter_bert_list,side_adapter_mm_list,stored_vector_path,cached_image_model,cached_text_model,text_layers,text_embedding_dim,cached_text_prefix)
                                                            os.system(run_py)



