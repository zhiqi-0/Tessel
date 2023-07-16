
import torch
import cube

from examples.gpt.blocks import TransformerLayer
from dataclasses import dataclass

@dataclass
class Config:
    embed_dim: int = 1024
    layers: int = 8
    attention_heads: int = 16
    attn_hidden_dim: int = 1024
    ffn_hidden_dim: int = 4096
    num_embeddings: int = 51200
    seqlen: int = 1024
    dropout: float = 0.2
    attn_dropout: float = 0.2
    activation_dropout: float = 0.2


class GPT(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # self.embed = torch.nn.Embedding(cfg.num_embeddings, cfg.embed_dim)
        self.embedw = torch.nn.Parameter(torch.empty(cfg.num_embeddings, cfg.embed_dim))
        self.position = torch.nn.Embedding(cfg.seqlen, cfg.embed_dim)
        self.embed_dropout = torch.nn.Dropout()

        self.layers = torch.nn.ModuleList(
            [TransformerLayer(
                cfg.embed_dim, cfg.attention_heads,
                cfg.attn_hidden_dim, cfg.ffn_hidden_dim,
                cfg.dropout, cfg.attn_dropout, cfg.activation_dropout
            ) for _ in range(cfg.layers)]
        )
        self.final_layernorm = torch.nn.LayerNorm(cfg.embed_dim)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor):

        # embed = self.embed(input_ids)
        embed = torch.nn.functional.embedding(
            input_ids, self.embedw, padding_idx=None,
            max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False
        )
        pos_embed = self.position(position_ids)
        embed = embed + pos_embed
        embed = self.embed_dropout(embed)
        enc = embed.transpose(0, 1)

        for layer in self.layers:
            cube.runtime.function.anchor('transformer start')
            enc = layer(enc)
        enc = self.final_layernorm(enc)

        # ====> pretrain setting
        # logits = torch.nn.functional.linear(enc, self.embedw)
        # # simplify
        # loss = torch.sum(logits)

        # ===> finetune setting
        loss = torch.sum(enc)
        return loss


class GPTDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, bs: int, cfg: Config = None):
        self.cfg = Config() if cfg is None else cfg
        super().__init__(bs, [0, 0])
        self.sample = None
        self.set_batch_size(bs)

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample
    
    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        input_ids = torch.randint(
            0, self.cfg.num_embeddings,
            size=(self.batch_size, self.cfg.seqlen),
            dtype=torch.int64, device=torch.cuda.current_device()
        )
        position_ids = torch.arange(
            0, self.cfg.seqlen, dtype=torch.int64, device=torch.cuda.current_device()
        ).repeat(self.batch_size).view(self.batch_size, -1)
        self.sample = (input_ids, position_ids)
