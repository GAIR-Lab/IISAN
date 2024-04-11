import torch.nn as nn
import torch


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.activate = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.dropout(self.w_2(self.activate(self.w_1(x))))
        return self.layer_norm(residual + x)


class SelfAttention(nn.Module):
    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask):
        attn = torch.matmul(query, key.transpose(-2, -1)) / self.temperature
        attn = attn + mask
        p_attn = self.dropout(self.softmax(attn))
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_v = self.d_k

        self.w_Q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_K = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_V = nn.Linear(d_model, n_heads * self.d_v, bias=False)
        self.fc = nn.Linear(n_heads * self.d_v, d_model, bias=False)

        self.self_attention = SelfAttention(temperature=self.d_k ** 0.5, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query, key, value, mask):
        sz_b, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), value.size(1)
        residual = query

        q = self.w_Q(query).view(sz_b, len_q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_K(key).view(sz_b, len_k, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_V(value).view(sz_b, len_v, self.n_heads, self.d_v).transpose(1, 2)

        x, attn = self.self_attention(q, k, v, mask=mask)
        x = x.transpose(1, 2).contiguous().view(sz_b, len_q, self.d_model)
        x = self.dropout(self.fc(x))
        return self.layer_norm(residual + x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_inner, dropout):

        super().__init__()
        self.multi_head_attention = MultiHeadedAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, block_input, mask):
        output = self.multi_head_attention(block_input, block_input, block_input, mask)
        return self.feed_forward(output)


class TransformerEncoder(torch.nn.Module):
    def __init__(self, n_vocab, n_position, d_model, n_heads, dropout, n_layers):
        super(TransformerEncoder, self).__init__()
        # self.word_embedding = nn.Embedding(n_vocab + 1, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(n_position, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, d_inner=d_model * 4, dropout=dropout
                              ) for _ in range(n_layers)])

    def forward(self, input_embs, log_mask, att_mask):
        position_ids = torch.arange(log_mask.size(1), dtype=torch.long, device=log_mask.device)
        position_ids = position_ids.unsqueeze(0).expand_as(log_mask)
        output = self.layer_norm(input_embs + self.position_embedding(position_ids))
        output = self.dropout(output)
        if "SASRecKAdaptedTransformerBlocks" in str(type(self.transformer_blocks)):
            output = self.transformer_blocks(output, att_mask)
        else:
            for transformer in self.transformer_blocks:
                output = transformer.forward(output, att_mask)
        return output
class AdapterBlock(torch.nn.Module):
    def __init__(self,args, input_size,down_size,dropout=0.1):
        super(AdapterBlock, self).__init__()
        self.fc_down = nn.Linear(input_size,down_size)
        nn.init.normal_(self.fc_down.weight,std=1e-2)
        nn.init.zeros_(self.fc_down.bias)
        if args.adapter_activation =="GELU":
            self.activate = nn.GELU()
        else:
            self.activate = nn.ReLU()
        self.fc_up = nn.Linear(down_size,input_size)
        nn.init.normal_(self.fc_up.weight,std=1e-2)
        nn.init.zeros_(self.fc_up.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self,input_embs):
        x = self.fc_down(input_embs)
        x = self.activate(x)
        return self.fc_up(x)+input_embs



class AdapterPfeifferBlock(torch.nn.Module):
    # only has a bottle neck structure and without the residual
    def __init__(self,args, input_size,down_size,dropout=0.1):
        super(AdapterPfeifferBlock, self).__init__()
        self.fc_down = nn.Linear(input_size,down_size)
        nn.init.normal_(self.fc_down.weight,std=1e-2)
        nn.init.zeros_(self.fc_down.bias)
        if args.adapter_activation =="GELU":
            self.activate = nn.GELU()
        else:
            self.activate = nn.ReLU()
        self.fc_up = nn.Linear(down_size,input_size)
        nn.init.normal_(self.fc_up.weight,std=1e-2)
        nn.init.zeros_(self.fc_up.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self,input_embs):
        x = self.fc_down(input_embs)
        x = self.activate(x)
        return self.fc_up(x)

class KAdapterBlock(torch.nn.Module):
    def __init__(self,args,num_head,input_size,down_size,dropout=0.1):
        super(KAdapterBlock, self).__init__()
        self.args = args
        self.down_project = nn.Linear(
            input_size,
            down_size,
        )
        self.up_project = nn.Linear(down_size, input_size)
        self.init_weights()
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model=down_size, n_heads=num_head, d_inner=down_size * 4, dropout=dropout
                              ) for _ in range(2)])

    def forward(self,hidden_states):
        down_projected = self.down_project(hidden_states)
        input_shape = down_projected.size()[:-1]
        attention_mask = torch.ones(input_shape, device=down_projected.device)
        encoder_attention_mask = torch.ones(input_shape, device=down_projected.device)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        output =down_projected
        for transformer in self.transformer_blocks:
            output = transformer.forward(output, extended_attention_mask)

        up_projected = self.up_project(output)
        return hidden_states + up_projected

    def init_weights(self):
        self.down_project.weight.data.normal_(mean=0.0, std=2e-4)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=2e-4)
        self.up_project.bias.data.zero_()
