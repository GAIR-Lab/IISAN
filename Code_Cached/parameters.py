from data_utils.utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    # ============== data_dir ==============
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--item_tower", type=str, default="modal")
    parser.add_argument("--root_data_dir", type=str, default="../",)
    parser.add_argument("--dataset", type=str, default='pinterest')
    parser.add_argument("--behaviors", type=str, default='users_log.tsv')
    parser.add_argument("--images", type=str, default='images_log.tsv')
    parser.add_argument("--lmdb_data", type=str, default='image.lmdb')
    parser.add_argument("--news", type=str, default='Industrial_and_Scientific_items.tsv')


    # ============== train parameters ==============
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fine_tune_lr_image", type=float, default=1e-4)
    parser.add_argument("--fine_tune_lr_text", type=float, default=5e-5)

    parser.add_argument("--l2_weight", type=float, default=0)
    parser.add_argument("--drop_rate", type=float, default=0.1)

    # ============== model parameters ==============
    parser.add_argument("--CV_model_load", type=str, default='resnet-50')
    parser.add_argument("--freeze_paras_before", type=int, default=0)
    parser.add_argument("--CV_resize", type=int, default=224)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--transformer_block", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=10)
    parser.add_argument("--min_seq_len", type=int, default=5)
    parser.add_argument("--arch", type=str, default='sasrec')
    parser.add_argument("--use_scale",type=str,default='half')
    parser.add_argument("--n_tokens", type=int, default=10)

    parser.add_argument("--bert_model_load", type=str, default='bert-base-uncased')
    parser.add_argument("--word_embedding_dim", type=int, default=768)
    parser.add_argument("--use_cls", type=bool, default=True)

    # ============== news information==============
    parser.add_argument("--num_words_title", type=int, default=30)
    parser.add_argument("--num_words_abstract", type=int, default=50)
    parser.add_argument("--num_words_body", type=int, default=50)
    
    parser.add_argument("--news_attributes", type=str, default='title')

    # ============== switch and logging setting ==============
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--load_ckpt_name", type=str, default='None')
    parser.add_argument("--label_screen", type=str, default='None')
    parser.add_argument("--logging_num", type=int, default=8)
    parser.add_argument("--testing_num", type=int, default=1)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--pretrained_recsys_model", default="None", type=str)

    # ============== For adding logs_adapter ==============
    parser.add_argument("--adapter_down_size",type=int,default=16)
    # It can be bert, sasrec_all, sasrec_first, sasrec_last, and all or None
    parser.add_argument("--adding_adapter_to", type=str, default="None")
    # It can be bert, sasrec_all, sasrec_first, sasrec_last, and all or None
    parser.add_argument("--fine_tune_to",type=str,default='all')
    parser.add_argument("--adapter_cv_lr",type=float,default=4e-4)
    parser.add_argument("--adapter_bert_lr",type=float,default=1e-4)
    parser.add_argument("--bert_adapter_down_size", type=int, default=64)
    parser.add_argument("--adapter_sasrec_lr", type=float, default=1e-4)
    parser.add_argument("--cv_adapter_down_size", type=int, default=64)
    parser.add_argument("--adapter_dropout_rate", type=float,default=0.1)
    parser.add_argument("--adapter_activation", type=str, default="RELU")
    parser.add_argument("--finetune_layernorm", type=str, default="None")
    parser.add_argument("--is_serial", type=str, default="True")
    parser.add_argument("--adapter_type",type=str,default='houslby')
    parser.add_argument("--k_adapter_bert_list", type=str, default='0,11')
    parser.add_argument("--k_adapter_bert_hidden_dim", type=int, default=384)
    parser.add_argument("--num_adapter_heads_sasrec", type=int, default=2)
    parser.add_argument("--num_adapter_heads_bert", type=int, default=12)
    parser.add_argument("--num_dnn", type=int, default=0)
    #==========================compacter==========================
    parser.add_argument("--hypercomplex_division", type=int, default=8)
    parser.add_argument("--phm_init_range", type=float, default=0.0001)
    #======================SideAdapter===========================
    #0,1,2,3,4,5,6,7,8,9,10,11 
    #1,3,5,7,9,11
    #6,7,8,9,10,11
    parser.add_argument("--side_adapter_vit_list",type=str,default='1,3,5,7,9,11')
    parser.add_argument("--side_adapter_bert_list",type=str,default='1,3,5,7,9,11')
    parser.add_argument("--side_adapter_mm_list",type=str,default='1,3,5,7,9,11')
    parser.add_argument("--fusion_method", type=str, default="gated")
    parser.add_argument("--remove_first", type=str, default="None")
    parser.add_argument("--fusion_inter",type=str,default="add")
    parser.add_argument("--stored_vector_path",type=str,default="")
    parser.add_argument("--modality",type=str,default="intra_inter") #intra_inter, inter, other
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    args.news_attributes = args.news_attributes.split(',')
    return args


if __name__ == "__main__":
    args = parse_args()
