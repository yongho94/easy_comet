from torch import nn
import torch

class GPT_model(nn.Module):
    def __init__(self, hyp):
        super(GPT_model, self).__init__()
        self.embedding_layer = EmbeddingLayer(hyp)
        self.transformer_blocks = list()
        for _ in range(hyp['num_layers']):
            self.transformer_blocks.append(TransformerLayer(hyp))
        self.transformer_blocks = nn.ModuleList(self.transformer_blocks)
        self.predict_layer = PredictLayer(hyp)
        self.predict_softmax = nn.functional.softmax

    def forward(self, input, seq_mask):
        e_x = self.embedding_layer(input)  # [b, s, h]
        h_x = e_x
        for block in self.transformer_blocks:
            h_x = block(h_x, seq_mask)
        pred_score = self.predict_layer(h_x)  # for training
        pred_prob = self.predict_softmax(pred_score, -1)  # for PPL or something
        return pred_score, pred_prob

class EmbeddingLayer(nn.Module):
    def __init__(self, hyp):
        super(EmbeddingLayer, self).__init__()
        self.seq_len = hyp['seq_len']
        bpe_vocab_num = hyp['vocab_num']
        pos_vocab_num = hyp['pos_vocab_num']
        hidden = hyp['hidden']
        self.pos_emb = nn.Embedding(pos_vocab_num, hidden)
        self.bpe_emb = nn.Embedding(bpe_vocab_num, hidden)

    def forward(self, x):
        bpe = self.bpe_emb(x)
        pos_idx = torch.arange(self.seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand_as(x)
        pos = self.pos_emb(pos_idx)
        emb = bpe + pos
        return emb


class TransformerLayer(nn.Module):
    def __init__(self, hyp):
        super(TransformerLayer, self).__init__()
        self.local_attention_layer = LocalAttentionLayer(hyp)
        self.norm_layer_1 = NormalizeLayer(hyp['hidden'])
        self.feed_forward_layer = FeedForwardLayer(hyp)
        self.norm_layer_2 = NormalizeLayer(hyp['hidden'])

    def forward(self, x, seq_mask):
        a = self.local_attention_layer(x, seq_mask)
        x = self.norm_layer_1(x+a)
        m = self.feed_forward_layer(x)
        x = self.norm_layer_2(x+m)
        return x


class LocalAttentionLayer(nn.Module):
    def __init__(self, hyp):
        super(LocalAttentionLayer, self).__init__()
        self.hidden = hidden = hyp['hidden']
        self.num_heads = hyp['num_heads']
        self.scale_val = torch.tensor(self.hidden / self.num_heads)
        self.qkv_proj_layer = nn.Linear(hidden, hidden * 3)
        self.att_drop = nn.Dropout(hyp['att_dropout'])
        self.res_layer = nn.Linear(hidden, hidden)
        self.res_drop = nn.Dropout(hyp['res_dropout'])
        self.softmax = nn.Softmax(-1)

    def split_heads(self, x):
        new_shape = list(x.shape[:-1])+[self.num_heads, -1]
        return x.view(new_shape)  # [b, s, hn, hd]

    def make_mask_low_prob(self, mask):
        mask = (torch.ones_like(mask) - mask) * -1e9
        return mask

    def forward(self, x, seq_mask): # [ b, s ]
        qkv = self.qkv_proj_layer(x)  # [b, s, 3h]
        q, k, v = torch.split(qkv, self.hidden, 2)  # [b, s, h] * 3
        q = self.split_heads(q).permute([0, 2, 1, 3])  # [b, hn, s, hd]
        k = self.split_heads(k).permute([0, 2, 3, 1])  # [b, hn, hd, s]
        v = self.split_heads(v).permute([0, 2, 1, 3])  # [b, hn, s, hd]

        self.att_score = torch.matmul(q, k)  # [b, hn, s, s]
        self.att_score /= torch.sqrt(self.scale_val.to(x.device))

        att_mask = torch.tril(torch.ones_like(self.att_score), diagonal=0)
        seq_mask = seq_mask.unsqueeze(1).unsqueeze(2)
        mask = att_mask * seq_mask#torch.logical_and(seq_mask, att_mask)
        mask = self.make_mask_low_prob(mask)
        self.att_score += mask
        self.att_prob = self.softmax(self.att_score)
        self.att_prob = self.att_drop(self.att_prob)

        self.value = torch.matmul(self.att_prob, v).permute([0, 2, 1, 3])
        self.value = torch.flatten(self.value, start_dim=2, end_dim=-1)

        self.value = self.res_layer(self.value)
        self.value = self.res_drop(self.value)

        return self.value


class NormalizeLayer(nn.Module):
    "Construct a layernorm module in the OpenAI style \
    (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(NormalizeLayer, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class FeedForwardLayer(nn.Module):
    def __init__(self, hyp):
        super(FeedForwardLayer, self).__init__()
        hidden = hyp['hidden']
        self.linear_1 = nn.Linear(hidden, 4 * hidden)
        self.linear_2 = nn.Linear(4 * hidden, hidden)
        self.gelu = nn.GELU()

    def forward(self, x):
        a1 = self.linear_1(x)
        x2 = self.gelu(a1)
        a2 = self.linear_2(x2)
        return a2


class PredictLayer(nn.Module):
    def __init__(self, hyp):
        super(PredictLayer, self).__init__()
        self.linear = nn.Linear(hyp['hidden'], hyp['vocab_num'], bias=False)

    def forward(self, x):
        score = self.linear(x)
        return score