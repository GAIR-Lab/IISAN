import torch.optim as optim
import re
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from transformers import BeitForImageClassification, CLIPVisionModel, ViTMAEModel, ViTForImageClassification,BertModel, BertTokenizer, BertConfig, RobertaTokenizer, RobertaModel, RobertaConfig, AutoModel, \
    AutoConfig, AutoTokenizer, CLIPModel, AutoTokenizer, AutoModelForMaskedLM,DebertaV2Model

from parameters import parse_args
from model import *
from data_utils import *
from data_utils.utils import *
import torchvision.models as models
from torch import nn
import random
from torch.cuda.amp import autocast
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.init import xavier_normal_, constant_

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def add_adapter_to_bert(self_output, args):
    return BertAdaptedSelfOutput(self_output, args)

def add_adapter_to_vit_selfoutput(self_output, args):
    return VITAdaptedSelfOutput(self_output, args)

def add_adapter_to_vit_output(self_output, args):
    return VITAdaptedOutput(self_output, args)

def add_interIISAN_adapter_to_model(mm_model, args):
    return IISANAdaptedMModel(mm_model, args)



def train(args, use_modal, local_rank):
    if use_modal:
        #--------------------------------- Image ---------------------------------
        if 'vit' in args.CV_model_load:
            Log_file.info('load vit model...')
            cv_model_load = '../../pretrained_models/vit-base-patch16-224'
            cv_model = ViTForImageClassification.from_pretrained(cv_model_load)
            num_fc_ftr = cv_model.classifier.in_features
            cv_model.classifier = nn.Linear(num_fc_ftr, args.embedding_dim)
            xavier_normal_(cv_model.classifier.weight.data)
            if cv_model.classifier.bias is not None:
                constant_(cv_model.classifier.bias.data, 0)
        else:
            cv_model = None

        for index, (name, param) in enumerate(cv_model.named_parameters()):
            if index < args.freeze_paras_before:
                param.requires_grad = False
        #--------------------------------- Text ---------------------------------

        Log_file.info('load bert model...')
        bert_model_load = '../../pretrained_models/bert/' + args.bert_model_load
        tokenizer = BertTokenizer.from_pretrained(bert_model_load)
        config = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
        bert_model = BertModel.from_pretrained(bert_model_load, config=config)

        if 'tiny' in args.bert_model_load:
            pooler_para = [37, 38]
            args.word_embedding_dim = 128
        if 'mini' in args.bert_model_load:
            pooler_para = [69, 70]
            args.word_embedding_dim = 256
        if 'medium' in args.bert_model_load:
            pooler_para = [133, 134]
            args.word_embedding_dim = 512
        if 'base' in args.bert_model_load:
            pooler_para = [197, 198]
            args.word_embedding_dim = 768
        if 'large' in args.bert_model_load:
            pooler_para = [389, 390]
            args.word_embedding_dim = 1024
        for index, (name, param) in enumerate(bert_model.named_parameters()):
            if index < args.freeze_paras_before or index in pooler_para:
                param.requires_grad = False

    else:
        cv_model = None


    #--------------------------------- Image ---------------------------------
    Log_file.info('read items...')
    before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name = read_images(
        os.path.join(args.root_data_dir, args.dataset, args.images))
    before_item_id_to_dic_text, before_item_name_to_id_text = read_news_bert(
        os.path.join(args.root_data_dir, args.dataset, args.news), args, tokenizer)


    Log_file.info('read behaviors...')
    item_num, item_id_to_keys, item_id_to_keys_text, users_train, users_valid, users_test, \
    users_history_for_valid, users_history_for_test, item_name_to_id, neg_sampling_list, pop_prob_list = \
        read_behaviors(os.path.join(args.root_data_dir, args.dataset, args.behaviors), before_item_id_to_dic, before_item_id_to_dic_text,
                       before_item_name_to_id, before_item_id_to_name, args.max_seq_len, args.min_seq_len, Log_file)
    

    Log_file.info('combine news information...')
    news_title, news_title_attmask, \
    news_abstract, news_abstract_attmask, \
    news_body, news_body_attmask = get_doc_input_bert(item_id_to_keys_text, args)

    item_content = np.concatenate([
        x for x in
        [news_title, news_title_attmask,
         news_abstract, news_abstract_attmask,
         news_body, news_body_attmask]
        if x is not None], axis=1)

    Log_file.info('build dataset...')
    if use_modal:
        train_dataset = Build_MM_Dataset(u2seq=users_train, item_content=item_content, item_num=item_num, max_seq_len=args.max_seq_len,
                                           db_path=os.path.join(args.root_data_dir, args.dataset, args.lmdb_data),
                                           item_id_to_keys=item_id_to_keys, resize=args.CV_resize,
                                           neg_sampling_list=neg_sampling_list,stored_vector_path=args.stored_vector_path)
    else:
        train_dataset = Build_Id_Dataset(u2seq=users_train, item_num=item_num, max_seq_len=args.max_seq_len)

    Log_file.info('build DDP sampler...')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    def worker_init_reset_seed(worker_id):
        initial_seed = torch.initial_seed() % 2 ** 31
        worker_seed = initial_seed + worker_id + dist.get_rank()
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    Log_file.info('build dataloader...')

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                          worker_init_fn=worker_init_reset_seed, pin_memory=True,sampler=train_sampler)

    Log_file.info('build model...')
    model = ModelMM(args, item_num, use_modal, cv_model,bert_model, pop_prob_list).to(local_rank)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    if 'None' not in args.pretrained_recsys_model:
        Log_file.info('load pretrained recsys model if not None...')
        ckpt_path = get_checkpoint("../pretrained_models/", args.pretrained_recsys_model)
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        Log_file.info('load checkpoint...')
        model.load_state_dict(checkpoint['model_state_dict'])
        Log_file.info(f"Model loaded from {ckpt_path}")
        torch.set_rng_state(checkpoint['rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])


    # adding adapters
    # make all model gradient becomes false
    if 'all' in args.fine_tune_to:
        pass
    elif 'None' in args.fine_tune_to:
        for index, (name, param) in enumerate(model.named_parameters()):
            param.requires_grad = False
    else:
        assert 1 == 0, "fine_tune_to should be defined properly"

    # adding adapters after this line
    if 'None' in args.adding_adapter_to:
        pass
    else:
        if "lora" in args.adapter_type:
            import loralib as lora
            for index, layer_module in enumerate(
                    model.mm_encoder.cv_encoder.image_net.vit.encoder.layer):
                layer_module.attention.attention.query = lora.Linear(768, 768, r=args.bert_adapter_down_size).to(local_rank)
                layer_module.attention.attention.value = lora.Linear(768, 768, r=args.bert_adapter_down_size).to(local_rank)

            for index, layer_module in enumerate(
                    model.mm_encoder.bert_encoder.text_encoders.title.bert_model.encoder.layer):
                layer_module.attention.self.query = lora.Linear(args.word_embedding_dim,args.word_embedding_dim, r=args.bert_adapter_down_size).to(local_rank)
                layer_module.attention.self.value = lora.Linear(args.word_embedding_dim,args.word_embedding_dim, r=args.bert_adapter_down_size).to(local_rank)

            for index, (name, param) in enumerate(model.named_parameters()):
                if any(["user" in name, "classifier" in name, "title.fc" in name]) or all(["user" not in name, "encoder" not in name]):
                    param.requires_grad = True
        elif "IISAN" in args.adapter_type:
            model.mm_encoder = add_interIISAN_adapter_to_model(
                model.mm_encoder, args).to(local_rank)
            # adding adapters to the SASRec model
            for index, (name, param) in enumerate(model.named_parameters()):
                if any(["user" in name, "classifier" in name, "title.fc" in name, "cv_pre_fc" in name, "bert_pre_fc" in name]) or all(["user" not in name, "encoder" not in name]):
                    param.requires_grad = True
        elif "houslby" in args.adapter_type:
            # using the classic houslby adapter
            if "None" not in args.is_serial:
                # adding adapters to the cv model

                if "mae" not in args.CV_model_load:
                    for index, layer_module in enumerate(
                            model.mm_encoder.cv_encoder.image_net.vit.encoder.layer):
                        # adding adapters to cv model
                        layer_module.attention.output = add_adapter_to_vit_selfoutput(layer_module.attention.output,
                                                                                      args).to(
                            local_rank)
                        layer_module.output = add_adapter_to_vit_output(layer_module.output, args).to(
                            local_rank)
                    # adding adapters to text model
                    for index, layer_module in enumerate(
                        model.mm_encoder.bert_encoder.text_encoders.title.bert_model.encoder.layer):
                        layer_module.attention.output = add_adapter_to_bert(layer_module.attention.output, args).to(
                            local_rank)
                        layer_module.output = add_adapter_to_bert(layer_module.output, args).to(local_rank)
                    for index, (name, param) in enumerate(model.named_parameters()):
                        if any(["user" in name, "classifier" in name, "title.fc" in name]) or all(["user" not in name, "encoder" not in name]):
                            param.requires_grad = True
                else:
                    for index, layer_module in enumerate(
                            model.mm_encoder.cv_encoder.image_net.encoder.layer):
                        layer_module.attention.output = add_adapter_to_vit_selfoutput(layer_module.attention.output,
                                                                                      args).to(
                            local_rank)
                        layer_module.output = add_adapter_to_vit_output(layer_module.output, args).to(
                            local_rank)
                    for index, (name, param) in enumerate(model.named_parameters()):
                        if any(["user" in name, "classifier" in name, "title.fc" in name]) or all(["user" not in name, "encoder" not in name]):
                            param.requires_grad = True
            else:
                # adding adapters to the cv model
                for index, layer_module in enumerate(
                        model.mm_encoder.cv_encoder.image_net.vit.encoder.layer):
                    layer_module.output = add_parallel_adapter_to_vit_output(layer_module.output, args).to(local_rank)
                # adding adapters to the SASRec model
                for index, transformer_block in enumerate(
                        model.user_encoder.transformer_encoder.transformer_blocks):
                    model.user_encoder.transformer_encoder.transformer_blocks[
                        index] = add_parallel_adapter_to_sasrec(
                        transformer_block, args).to(local_rank)

    if 'None' not in args.load_ckpt_name:
        Log_file.info('load ckpt if not None...')
        ckpt_path = get_checkpoint(model_dir, args.load_ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        Log_file.info('load checkpoint...')
        model.load_state_dict(checkpoint['model_state_dict'])
        Log_file.info(f"Model loaded from {ckpt_path}")
        start_epoch = int(re.split(r'[._-]', args.load_ckpt_name)[1])
        torch.set_rng_state(checkpoint['rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        is_early_stop = True
    else:
        checkpoint = None  # new
        ckpt_path = None  # new
        start_epoch = 0
        is_early_stop = True


    Log_file.info('model.cuda()...')
    if 'None' not in args.finetune_layernorm:
        for index, (name, param) in enumerate(model.named_parameters()):
            if "adapter" not in name:
                if "LayerNorm" in name or "layer_norm" in name or "layernorm" in name:
                    param.requires_grad = True
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=False)

    if use_modal:
        image_net_params = []
        text_encoder_params = []
        recsys_params = []
        adapter_cv_params = []
        adapter_text_params = []
        adapter_recsys_params = []
        for index, (name, param) in enumerate(model.module.named_parameters()):
            if param.requires_grad:
                if 'cv' in name:
                    if ('fc' in name and "fc_" not in name) or 'classifier' in name or 'decoder_pred' in name:
                        recsys_params.append(param)
                    else:
                        if "adapter" not in name and "lora" not in name:
                            image_net_params.append(param)
                        else:
                            adapter_cv_params.append(param)
                elif "bert" in name:
                    if 'fc' in name and "fc_" not in name:
                         recsys_params.append(param)

                    else:
                        if "adapter" not in name and "lora" not in name:
                            text_encoder_params.append(param)
                        else:
                            adapter_text_params.append(param)
                elif "mm_adapter" in name:
                    adapter_cv_params.append(param)

                elif "user" in name:
                    recsys_params.append(param)
                else:
                    recsys_params.append(param)

        if 'None' in args.adding_adapter_to:
            optimizer = optim.Adam([
                {'params': text_encoder_params, 'lr': args.fine_tune_lr_text},
                {'params': image_net_params, 'lr': args.fine_tune_lr_image},
                {'params': recsys_params, 'lr': args.lr}
            ])
        else:
            optimizer = optim.Adam([
                {'params': text_encoder_params, 'lr': args.fine_tune_lr_text},
                {'params': image_net_params, 'lr': args.fine_tune_lr_image},
                {'params': recsys_params, 'lr': args.lr},
                {'params': adapter_cv_params, 'lr': args.adapter_cv_lr},
                {'params': adapter_text_params, 'lr': args.adapter_bert_lr},
            ])
        
        if "None" not in args.adding_adapter_to:
            Log_file.info("***** {} parameters in images, {} parameters in model *****".format(
            len(list(model.module.mm_encoder.parameters())),
            len(list(model.module.parameters()))))
        else:
            Log_file.info("***** {} parameters in mm, {} parameters in model *****".format(
            #len(list(model.module.cv_encoder.image_net.parameters())),
            len(list(model.module.mm_encoder.parameters())),
            len(list(model.module.parameters()))))
            

        for children_model in optimizer.state_dict()['param_groups']:
            Log_file.info("***** {} parameters have learning rate {} *****".format(
                len(children_model['params']), children_model['lr']))

        model_params_require_grad = []
        model_params_freeze = []
        for param_name, param_tensor in model.module.named_parameters():
            if param_tensor.requires_grad:
                model_params_require_grad.append(param_name)
            else:
                model_params_freeze.append(param_name)
        Log_file.info("***** freeze parameters before {} in cv *****".format(args.freeze_paras_before))
        Log_file.info("***** model: {} parameters require grad, {} parameters freeze *****".format(
            len(model_params_require_grad), len(model_params_freeze)))
    else:
        optimizer = optim.Adam(model.module.parameters(), lr=args.lr)

    if 'None' not in args.load_ckpt_name:
        optimizer.load_state_dict(checkpoint["optimizer"])
        Log_file.info(f"optimizer loaded from {ckpt_path}")

    total_num = sum(p.numel() for p in model.module.parameters())
    trainable_num = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    Log_file.info("##### total_num {} #####".format(total_num))
    Log_file.info("##### trainable_num {} #####".format(trainable_num))
    Log_file.info("Trainable layers:")
    Log_file.info([name for name, p in model.module.named_parameters() if p.requires_grad])
    Log_file.info('\n')
    Log_file.info('Training...')
    next_set_start_time = time.time()
    max_epoch, early_stop_epoch = 0, args.epoch
    max_eval_value, early_stop_count = 0, 0

    steps_for_log, _ = para_and_log(model, len(users_train), args.batch_size, Log_file,
                                    logging_num=args.logging_num, testing_num=args.testing_num)
    if "half" in args.use_scale:
        scaler = torch.cuda.amp.GradScaler()
    Log_screen.info('{} train start'.format(args.label_screen))
    # Log_file.info(model)
    max_hit10 = 0
    for ep in range(args.epoch):
        now_epoch = start_epoch + ep + 1
        Log_file.info('\n')
        Log_file.info('epoch {} start'.format(now_epoch))
        Log_file.info('')
        loss, batch_index, need_break = 0.0, 1, False
        model.train()
        train_dl.sampler.set_epoch(now_epoch)
        for data in train_dl:
            sample_items_id, sample_items_image,sample_items_text, log_mask = data
            sample_items_id, sample_items_image,sample_items_text, log_mask = \
                sample_items_id.to(local_rank), sample_items_image.to(local_rank),sample_items_text.to(local_rank), log_mask.to(local_rank)
            if use_modal:
                sample_items_image = sample_items_image.view(-1, 11, 13, args.word_embedding_dim)
                sample_items_text = sample_items_text.view(-1, 11, 13, args.word_embedding_dim)
            else:
                sample_items = sample_items.view(-1)
            sample_items_id = sample_items_id.view(-1)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                bz_loss = model(sample_items_id, sample_items_image,sample_items_text, log_mask, local_rank)
                loss += bz_loss.data.float()
            scaler.scale(bz_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if torch.isnan(loss.data):
                need_break = True
                break
            if batch_index % steps_for_log == 0:
                Log_file.info('cnt: {}, Ed: {}, batch loss: {:.5f}, sum loss: {:.5f}'.format(
                    batch_index, batch_index * args.batch_size, loss.data.item() / batch_index, loss.data.item()))
            batch_index += 1

        if not need_break:
            Log_file.info('')
            max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break = \
                run_eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
                         model, item_id_to_keys,item_content, users_history_for_valid, users_valid, 256, item_num, use_modal,
                         args.mode, is_early_stop, local_rank)
            model.train()
            if max_eval_value > max_hit10 or max_hit10 == 0 or ep % 10 == 0:
                max_hit10 = max_eval_value
                if use_modal and dist.get_rank() == 0:
                    run_eval_test(model, item_id_to_keys,item_content, users_history_for_test, users_test, args.batch_size, item_num,use_modal,args.mode, local_rank)
                    #save_model(now_epoch, model, model_dir, optimizer, torch.get_rng_state(),
                               #torch.cuda.get_rng_state(), Log_file)

        Log_file.info('')
        next_set_start_time = report_time_train(batch_index, now_epoch, loss, next_set_start_time, start_time, Log_file)
        Log_screen.info('{} training: epoch {}/{}'.format(args.label_screen, now_epoch, args.epoch))
        if need_break:
            break
    if dist.get_rank() == 0:
        save_model(now_epoch, model, model_dir, optimizer, torch.get_rng_state(), torch.cuda.get_rng_state(), Log_file)
    Log_file.info('\n')
    Log_file.info('%' * 90)
    Log_file.info(' max eval Hit10 {:0.5f}  in epoch {}'.format(max_eval_value * 100, max_epoch))
    Log_file.info(' early stop in epoch {}'.format(early_stop_epoch))
    Log_file.info('the End')
    Log_screen.info('{} train end in epoch {}'.format(args.label_screen, early_stop_epoch))

def run_eval_test(model, item_id_to_keys,item_content, users_history_for_test, users_test, batch_size, item_num, use_modal,
                  mode, local_rank):
    eval_start_time = time.time()
    Log_file.info('Validating...')
    if use_modal:
        image_item_embeddings,text_item_embeddings = get_MM_item_embeddings(model, item_num, item_id_to_keys,item_content, batch_size, args, local_rank)

    else:
        item_embeddings = get_itemId_embeddings(model, item_num, batch_size, args, local_rank)
    valid_Hit10 = eval_model(model, users_history_for_test, users_test, image_item_embeddings,text_item_embeddings,batch_size, args,
                             item_num, Log_file, mode, local_rank)
    report_time_eval(eval_start_time, Log_file)
    

def run_eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
             model, item_id_to_keys,item_content, user_history, users_eval, batch_size, item_num, use_modal,
             mode, is_early_stop, local_rank):
    eval_start_time = time.time()
    Log_file.info('Validating...')
    if use_modal:
        image_item_embeddings,text_item_embeddings = get_MM_item_embeddings(model, item_num, item_id_to_keys,item_content, batch_size, args, local_rank)

    else:
        item_embeddings = get_itemId_embeddings(model, item_num, batch_size, args, local_rank)
    valid_Hit10 = eval_model(model, user_history, users_eval, image_item_embeddings,text_item_embeddings,batch_size, args,
                             item_num, Log_file, mode, local_rank)
    report_time_eval(eval_start_time, Log_file)
    Log_file.info('')
    need_break = False
    if valid_Hit10 > max_eval_value:
        max_eval_value = valid_Hit10
        max_epoch = now_epoch
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count > 10:
            if is_early_stop:
                need_break = True
            early_stop_epoch = now_epoch
    return max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = parse_args()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    setup_seed(args.seed)
    is_use_modal = True
    model_load = args.CV_model_load.replace('.pth', '')
    dir_label = str(args.seed)+str(args.arch)+ f'{model_load}_freeze_{args.freeze_paras_before}' + f"_add_adapter_to_{args.adding_adapter_to}" + f"_adapter_cv_lr_{args.adapter_cv_lr}" + f"_adapter_down_size_{args.adapter_down_size}" + f"_cv_adapter_down_size_{args.cv_adapter_down_size}_{args.adapter_type}"

    log_paras = f'{model_load}_bs_{args.batch_size}' \
                f'_ed_{args.embedding_dim}_lr_{args.lr}' \
                f'_L2_{args.l2_weight}_dp_{args.drop_rate}_Flr_{args.fine_tune_lr_image}_{args.fine_tune_lr_text}'
    model_dir = os.path.join('./checkpoint_'  + dir_label, 'cpt_' + log_paras)
    time_run = time.strftime('-%Y%m%d-%H%M%S', time.localtime())
    args.label_screen = args.label_screen + time_run

    Log_file, Log_screen = setuplogger(dir_label, log_paras, time_run, args.mode, dist.get_rank())

    Log_file.info(args)
    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if 'train' in args.mode:
        train(args, is_use_modal, local_rank)
    elif 'test' in args.mode:
        test(args, is_use_modal, local_rank)
    end_time = time.time()
    hour, minu, secon = get_time(start_time, end_time)
    Log_file.info("##### (time) all: {} hours {} minutes {} seconds #####".format(hour, minu, secon))
