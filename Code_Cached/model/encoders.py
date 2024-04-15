import torch
import torch.nn as nn
from .modules import TransformerEncoder
from torch.nn.init import xavier_normal_, constant_

class MM_Encoder(torch.nn.Module):
    def __init__(self,args,image_net,bert_model):
        super(MM_Encoder,self).__init__()
        self.cv_encoder = Vit_Encoder(image_net=image_net)
        self.bert_encoder = Bert_Encoder(args=args, bert_model=bert_model)
    def forward(self,sample_items_images, sample_items_text):
        score_embs_cv, _ = self.cv_encoder(sample_items_images)
        score_embs_text, _ = self.bert_encoder(sample_items_text)
        return score_embs_cv, score_embs_text


class Vit_Encoder(torch.nn.Module):
    def __init__(self, image_net):
        super(Vit_Encoder, self).__init__()
        self.image_net = image_net
        self.activate = nn.GELU()

    def forward(self, item_content):
        outputs = self.image_net(item_content, return_dict=None,output_hidden_states= True)
        return self.activate(outputs[0]),outputs[1]

class Vit_EncoderFFT(torch.nn.Module):
    def __init__(self, image_net):
        super(Vit_EncoderFFT, self).__init__()
        self.image_net = image_net
        self.activate = nn.GELU()

    def forward(self, item_content):
        return self.activate(self.image_net(item_content,return_dict=None)[0])


class User_Encoder(torch.nn.Module):
    def __init__(self, item_num, max_seq_len, item_dim, num_attention_heads, dropout, n_layers):
        super(User_Encoder, self).__init__()
        self.transformer_encoder = TransformerEncoder(n_vocab=item_num, n_position=max_seq_len,
                                                      d_model=item_dim, n_heads=num_attention_heads,
                                                      dropout=dropout, n_layers=n_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, input_embs, log_mask, local_rank):
        att_mask = (log_mask != 0)
        att_mask = att_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        att_mask = torch.tril(att_mask.expand((-1, -1, log_mask.size(-1), -1))).to(local_rank)
        att_mask = torch.where(att_mask, 0., -1e9)
        return self.transformer_encoder(input_embs, log_mask, att_mask)


class Text_Encoder(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 item_embedding_dim,
                 word_embedding_dim):
        super(Text_Encoder, self).__init__()
        self.bert_model = bert_model
        self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        self.activate = nn.GELU()


    def forward(self, text):

        batch_size, num_words = text.shape # 2688, 60
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)
        outputs = self.bert_model(input_ids=text_ids, attention_mask=text_attmask)
        hidden_states = outputs[0]
        all_layer_hidden_states = outputs[2]

        cls = self.fc(hidden_states[:, 0])
        return self.activate(cls),all_layer_hidden_states

class Text_EncoderFFT(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 item_embedding_dim,
                 word_embedding_dim):
        super(Text_EncoderFFT, self).__init__()
        self.bert_model = bert_model
        self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        self.activate = nn.GELU()


    def forward(self, text):

        batch_size, num_words = text.shape # 2688, 60
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)
        hidden_states = self.bert_model(input_ids=text_ids, attention_mask=text_attmask)[0]
        cls = self.fc(hidden_states[:, 0])
      
        return self.activate(cls)


class Bert_Encoder(torch.nn.Module):
    def __init__(self, args, bert_model):
        super(Bert_Encoder, self).__init__()
        self.args = args
        self.attributes2length = {
            'title': args.num_words_title * 2,
            'abstract': args.num_words_abstract * 2,
            'body': args.num_words_body * 2
        }

        for key in list(self.attributes2length.keys()):
            if key not in args.news_attributes:
                self.attributes2length[key] = 0

        self.attributes2start = {
            key: sum(
                list(self.attributes2length.values())
                [:list(self.attributes2length.keys()).index(key)]
            )
            for key in self.attributes2length.keys()
        }

        assert len(args.news_attributes) > 0
        text_encoders_candidates = ['title', 'abstract', 'body']
        self.text_encoders = nn.ModuleDict({
            'title': Text_Encoder(bert_model, args.embedding_dim, args.word_embedding_dim)
        })


        self.newsname = [name for name in set(args.news_attributes) & set(text_encoders_candidates)]


    def forward(self, news):
        text_vectors = [
            self.text_encoders['title'](
                torch.narrow(news, 1, self.attributes2start[name], self.attributes2length[name]))
            for name in self.newsname
        ]
        if len(text_vectors) == 1:
            final_news_vector = text_vectors[0][0]
            all_layer_hidden_states = text_vectors[0][1]
        else:
            final_news_vector = torch.mean(torch.stack(text_vectors, dim=1), dim=1)
        return final_news_vector,all_layer_hidden_states


class Bert_EncoderFFT(torch.nn.Module):
    def __init__(self, args, bert_model):
        super(Bert_EncoderFFT, self).__init__()
        self.args = args
        self.attributes2length = {
            'title': args.num_words_title * 2,
            'abstract': args.num_words_abstract * 2,
            'body': args.num_words_body * 2
        }

        for key in list(self.attributes2length.keys()):
            if key not in args.news_attributes:
                self.attributes2length[key] = 0

        self.attributes2start = {
            key: sum(
                list(self.attributes2length.values())
                [:list(self.attributes2length.keys()).index(key)]
            )
            for key in self.attributes2length.keys()
        }

        assert len(args.news_attributes) > 0
        text_encoders_candidates = ['title', 'abstract', 'body']
        self.text_encoders = nn.ModuleDict({
            'title': Text_EncoderFFT(bert_model, args.embedding_dim, args.word_embedding_dim)
        })


        self.newsname = [name for name in set(args.news_attributes) & set(text_encoders_candidates)]


    def forward(self, news):
        text_vectors = [
            self.text_encoders['title'](
                torch.narrow(news, 1, self.attributes2start[name], self.attributes2length[name]))
            for name in self.newsname
        ]
        if len(text_vectors) == 1:
            final_news_vector = text_vectors[0]
        else:
            final_news_vector = torch.mean(torch.stack(text_vectors, dim=1), dim=1)
        return final_news_vector
