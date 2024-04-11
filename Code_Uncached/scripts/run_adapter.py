import os

import torch

root_data_dir = '../'
dataset = '../Dataset/Scientific'
behaviors = 'am_Industrial_and_Scientific_users.tsv'
images = 'Industrial_and_Scientific_items.tsv'
news = 'Industrial_and_Scientific_items.tsv'
lmdb_data = 'am_is.lmdb'
logging_num = 4
testing_num = 1

CV_resize = 224
CV_model_load = 'vit'
bert_model_load = 'bert_base_uncased'
freeze_paras_before = 0


mode = 'train'
item_tower = 'modal'
modality = "intra"
epoch = 100
load_ckpt_name = 'None'

adapter_type = "houslby" #kadapter
l2_weight_list = [0]
drop_rate_list = [0.1]
batch_size_list = [32]
lr_list = [2e-4] # learning rate for user encoder
embedding_dim_list = [64]
adapter_cv_lr_list = [4e-4] #4e4
adapter_bert_lr_list = [4e-4] #4e4
#None or all
fine_tune_to_list = ['None']
# None or TRUE
finetune_layernorm = "None"
cv_adapter_down_size = 64
bert_adapter_down_size = 64
# None or True
is_serial = "True"
fine_tune_lr_image_list = [1e-4]#1e-4
fine_tune_lr_text_list = [5e-5]#1e-4
adding_adapter_to = 'all'

for adapter_bert_lr in adapter_bert_lr_list:
    for adapter_cv_lr in adapter_cv_lr_list:
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
                                        run_py = "CUDA_VISIBLE_DEVICES='0' \
                                                 python  -m torch.distributed.launch --nproc_per_node 1 --master_port 1236\
                                                 ../run.py --root_data_dir {}  --dataset {} --behaviors {} --images {} --news {}  --lmdb_data {}\
                                                 --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                                 --l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
                                                 --CV_resize {} --CV_model_load {}  --epoch {} --freeze_paras_before {}  --fine_tune_lr_image {} --fine_tune_lr_text {}\
                                                 --adapter_cv_lr {} --fine_tune_to {}\
                                                 --finetune_layernorm {} --cv_adapter_down_size {} --bert_adapter_down_size {} --adapter_bert_lr {} --bert_model_load {} --modality {} --adapter_type {} --adding_adapter_to {}\
                                                 ".format(
                                            root_data_dir, dataset, behaviors, images,news, lmdb_data,
                                            mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
                                            l2_weight, drop_rate, batch_size, lr, embedding_dim,
                                            CV_resize, CV_model_load, epoch, freeze_paras_before, fine_tune_lr_image,fine_tune_lr_text,adapter_cv_lr
                                                         ,fine_tune_to,finetune_layernorm,cv_adapter_down_size,bert_adapter_down_size,adapter_bert_lr,bert_model_load, modality,adapter_type,adding_adapter_to)
                                        os.system(run_py)

