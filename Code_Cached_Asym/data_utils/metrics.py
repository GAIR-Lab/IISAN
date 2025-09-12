import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .dataset import *
import torch.distributed as dist
import os
import math
from torch.utils.data.dataloader import default_collate
def my_collate(batch):
    imgs, texts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    texts = torch.stack(texts, dim=0)
    
    return imgs, texts


def item_collate_fn(arr):
    arr = torch.LongTensor(arr)
    return arr
class ItemsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]

def id_collate_fn(arr):
    arr = torch.LongTensor(arr)
    return arr


def print_metrics(x, Log_file, v_or_t):
    Log_file.info(v_or_t + "_results   {}".format('\t'.join(["{:0.5f}".format(i * 100) for i in x])))


def get_mean(arr):
    return [i.mean() for i in arr]


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]


def eval_concat(eval_list, test_sampler):
    eval_result = []
    for eval_m in eval_list:
        eval_m_cpu = distributed_concat(eval_m, len(test_sampler.dataset)) \
            .to(torch.device("cpu")).numpy()
        eval_result.append(eval_m_cpu.mean())
    return eval_result


def metrics_topK(y_score, y_true, item_rank, topK, local_rank):
    order = torch.argsort(y_score, descending=True)
    y_true = torch.take(y_true, order)
    rank = torch.sum(y_true * item_rank)
    eval_ra = torch.zeros(2).to(local_rank)
    if rank <= topK:
        eval_ra[0] = 1
        eval_ra[1] = 1 / math.log2(rank + 1)
    return eval_ra



def get_MM_item_embeddings(model, item_num, item_id_to_keys,item_content, test_batch_size, args, local_rank):
    model.eval()

    item_dataset = Build_MM_EMBED_Eval_Dataset_Cached(args=args,data=np.arange(item_num + 1), item_id_to_keys=item_id_to_keys,item_content=item_content,db_path=os.path.join(args.root_data_dir, args.dataset, args.lmdb_data),resize=args.CV_resize)

    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size,
                                 num_workers=args.num_workers, pin_memory=True,collate_fn=my_collate)
    item_embeddings_cv = []
    item_embeddings_text = []
    if "inter" in args.modality:
        item_embeddings_inter = []

    with torch.no_grad():
        for input_ids_cv, input_ids_text in item_dataloader:
            input_ids_cv, input_ids_text= input_ids_cv.to(local_rank),input_ids_text.to(local_rank)
            if "inter" == args.modality:
                item_emb_cv, item_emb_texts = model.module.mm_encoder(input_ids_cv,input_ids_text)
                item_emb_text = item_emb_texts[0]
                item_emb_inter = item_emb_texts[1].to(torch.device("cpu")).detach()

                item_embeddings_inter.extend(item_emb_inter)
            elif "inter" in args.modality and "intra" in args.modality:
                item_emb_cv, item_emb_texts = model.module.mm_encoder(input_ids_cv,input_ids_text)
                item_emb_cv = item_emb_cv.to(torch.device("cpu")).detach()
                item_emb_text = item_emb_texts[0].to(torch.device("cpu")).detach()
                item_emb_inter = item_emb_texts[1].to(torch.device("cpu")).detach()

                item_embeddings_cv.extend(item_emb_cv)
                item_embeddings_text.extend(item_emb_text)
                item_embeddings_inter.extend(item_emb_inter)
            else:
                item_emb_cv,item_emb_text = model.module.mm_encoder(input_ids_cv,input_ids_text)
                item_emb_cv,item_emb_text = item_emb_cv.to(torch.device("cpu")).detach(),item_emb_text.to(torch.device("cpu")).detach() 
                item_embeddings_cv.extend(item_emb_cv)
                item_embeddings_text.extend(item_emb_text)
    if "inter" == args.modality:
        return item_emb_cv,[item_emb_text,torch.stack(tensors=item_embeddings_inter, dim=0)] # cv and text are None
    elif "inter" in args.modality and "intra" in args.modality:
        return torch.stack(tensors=item_embeddings_cv, dim=0),[torch.stack(tensors=item_embeddings_text, dim=0),torch.stack(tensors=item_embeddings_inter, dim=0)]
            
    return torch.stack(tensors=item_embeddings_cv, dim=0),torch.stack(tensors=item_embeddings_text, dim=0)



def get_itemId_embeddings(model, item_num, test_batch_size, args, local_rank):
    model.eval()
    item_dataset = Build_Id_Eval_Dataset(data=np.arange(item_num + 1))
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size,
                                 num_workers=args.num_workers, pin_memory=True, collate_fn=id_collate_fn)
    item_embeddings = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            item_emb = model.module.id_embedding(input_ids).to(torch.device("cpu")).detach()
            item_embeddings.extend(item_emb)
    return torch.stack(tensors=item_embeddings, dim=0)


def get_itemLMDB_embeddings(model, item_num, item_id_to_keys, test_batch_size, args, local_rank):
    model.eval()
    item_dataset = Build_Lmdb_Eval_Dataset(data=np.arange(item_num + 1), item_id_to_keys=item_id_to_keys,
                                           db_path=os.path.join(args.root_data_dir, args.dataset, args.lmdb_data),
                                           resize=args.CV_resize)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size,
                                 num_workers=args.num_workers, pin_memory=True)
    item_embeddings = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            item_emb = model.module.mm_encoder.cv_encoder(input_ids)[0].to(torch.device("cpu")).detach()
            item_embeddings.extend(item_emb)
    return torch.stack(tensors=item_embeddings, dim=0)


def get_item_embeddings(model, item_content, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = ItemsDataset(item_content)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True, collate_fn=item_collate_fn)
    item_embeddings = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            if use_modal:
                item_emb = model.module.mm_encoder.bert_encoder(input_ids)[0]
            else:
                item_emb = model.module.id_embedding(input_ids)
            item_embeddings.extend(item_emb)
    return torch.stack(tensors=item_embeddings, dim=0).to(torch.device("cpu")).detach()


def eval_model(model, user_history, eval_seq, item_embeddings_image,item_embeddings_text, test_batch_size, args, item_num, Log_file, v_or_t,
               local_rank):
    # TODO needs to be changed to multimodality evaluation
    if "inter" in args.modality:
        item_embeddings_text,item_embeddings_inter = item_embeddings_text
        eval_dataset = BuildMMEvalDataset(args=args,u2seq=eval_seq, item_content_image=item_embeddings_image,
                                          item_content_text=item_embeddings_text,
                                    max_seq_len=args.max_seq_len,item_num=item_num,item_content_inter=item_embeddings_inter)
    else:
        eval_dataset = BuildMMEvalDataset(args=args,u2seq=eval_seq, item_content_image=item_embeddings_image,item_content_text=item_embeddings_text,
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()

    topK = 10
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    if "inter" == args.modality:
        item_embeddings_inter= item_embeddings_inter.to(local_rank)
        item_embeddings=model.module.com_dense(item_embeddings_inter)
        del item_embeddings_inter
    elif "inter" in args.modality:
        item_embeddings_image, item_embeddings_text,item_embeddings_inter= item_embeddings_image.to(local_rank), item_embeddings_text.to(local_rank),item_embeddings_inter.to(local_rank)
        item_embeddings=model.module.com_dense(torch.cat([item_embeddings_image, item_embeddings_text,item_embeddings_inter],dim=1))
        del item_embeddings_image,item_embeddings_text,item_embeddings_inter
    else:
        item_embeddings_image, item_embeddings_text= item_embeddings_image.to(local_rank), item_embeddings_text.to(local_rank)
        item_embeddings=model.module.com_dense(torch.cat([item_embeddings_image, item_embeddings_text],dim=1))
        del item_embeddings_image,item_embeddings_text
        
        
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        if "inter" == args.modality:
            for data in eval_dl:
                user_ids, input_embs_inter, log_mask, labels = data
                user_ids, input_embs_inter, log_mask, labels = \
                    user_ids.to(local_rank), input_embs_inter.to(local_rank), \
                    log_mask.to(local_rank), labels.to(local_rank).detach()
                score_embs = model.module.com_dense(input_embs_inter)
                del input_embs_inter
                prec_emb = model.module.user_encoder(score_embs, log_mask, local_rank)[:, -1].detach()
                scores = torch.matmul(prec_emb, item_embeddings.t()).squeeze(dim=-1).detach()
                for user_id, label, score in zip(user_ids, labels, scores):
                    user_id = user_id[0].item()
                    history = user_history[user_id].to(local_rank)
                    score[history] = -np.inf
                    score = score[1:]
                    eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        elif "inter" in args.modality:
            for data in eval_dl:
                user_ids, input_embs_image,input_embs_text,input_embs_inter, log_mask, labels = data
                user_ids, input_embs_image,input_embs_text,input_embs_inter, log_mask, labels = \
                    user_ids.to(local_rank), input_embs_image.to(local_rank),input_embs_text.to(local_rank),input_embs_inter.to(local_rank), \
                    log_mask.to(local_rank), labels.to(local_rank).detach()
                score_embs = model.module.com_dense(torch.cat([input_embs_image,input_embs_text,input_embs_inter],dim=2))
                del input_embs_image, input_embs_text,input_embs_inter
                prec_emb = model.module.user_encoder(score_embs, log_mask, local_rank)[:, -1].detach()
                scores = torch.matmul(prec_emb, item_embeddings.t()).squeeze(dim=-1).detach()
                for user_id, label, score in zip(user_ids, labels, scores):
                    user_id = user_id[0].item()
                    history = user_history[user_id].to(local_rank)
                    score[history] = -np.inf
                    score = score[1:]
                    eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        else:
            for data in eval_dl:
                user_ids, input_embs_image,input_embs_text, log_mask, labels = data
                user_ids, input_embs_image,input_embs_text, log_mask, labels = \
                    user_ids.to(local_rank), input_embs_image.to(local_rank),input_embs_text.to(local_rank), \
                    log_mask.to(local_rank), labels.to(local_rank).detach()
       
                score_embs = model.module.com_dense(torch.cat([input_embs_image,input_embs_text],dim=2))
                del input_embs_image, input_embs_text
                prec_emb = model.module.user_encoder(score_embs, log_mask, local_rank)[:, -1].detach()
                scores = torch.matmul(prec_emb, item_embeddings.t()).squeeze(dim=-1).detach()
                for user_id, label, score in zip(user_ids, labels, scores):
                    user_id = user_id[0].item()
                    history = user_history[user_id].to(local_rank)
                    score[history] = -np.inf
                    score = score[1:]
                    eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]
