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
freeze_paras_before = 0


mode = 'train'
item_tower = 'modal'

epoch = 100
load_ckpt_name = 'None'
pretrained_recsys_model = 'None'
bert_model_load = 'bert_base_uncased'
modality = "intra"

l2_weight_list = [0]
drop_rate_list = [0.1]
batch_size_list = [32]
lr_list = [1e-4]
embedding_dim_list = [64]
fine_tune_lr_image_list = [1e-4]#1e-4
fine_tune_lr_text_list = [5e-5]#1e-4
seed_list = [12345]
for seed in seed_list:
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
                                         python  -m torch.distributed.launch --nproc_per_node 1 --master_port 1232\
                                         ../run.py --root_data_dir {}  --dataset {} --behaviors {} --images {} --news {} --lmdb_data {}\
                                         --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                         --l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
                                         --CV_resize {} --CV_model_load {}  --epoch {} --freeze_paras_before {}  --fine_tune_lr_image {} --fine_tune_lr_text {} --pretrained_recsys_model {} --bert_model_load {} --seed {} --modality {}".format(
                                    root_data_dir, dataset, behaviors, images,news, lmdb_data,
                                    mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
                                    l2_weight, drop_rate, batch_size, lr, embedding_dim,
                                    CV_resize, CV_model_load, epoch, freeze_paras_before, fine_tune_lr_image,fine_tune_lr_text,pretrained_recsys_model,bert_model_load,seed,modality)
                                os.system(run_py)

