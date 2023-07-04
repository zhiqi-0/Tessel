import torch
import cube

from dataclasses import dataclass
from examples.mt5.blocks import EncoderLayer, DecoderLayer


@dataclass
class Config:
    vocab_size: int = 250112
    d_model: int = 512  # size of the encoder layer
    d_kv: int = 64 # size of the key, query, value projections per attention head. d_kv == d_model // num_heads
    d_ff: int = 1024 # size of the intermediate feeadforward layer
    num_layers: int = 8  # number of encoder layers and decoder layers (total layers = 2 * num_layers)
    num_heads: int = 6
    seqlen: int = 1024  # sequence length
    relative_attention_num_buckets: int = 32  # The number of buckets to use for each attention layer.
    relative_attention_max_distance: int = 128  # The maximum distance of the longer sequences for the bucket separation.
    dropout_rate: float = 0.1 # dropout rate
    layer_norm_eps: float = 1e-6 # layer norm epsilon
    feed_forward_proj: str = 'gated-gelu' #  Type of feed forward layer to be used. Should be one of "relu" or "gated-gelu".


class mT5(torch.nn.Module):

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.embedw = torch.nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_model))
        
        self.encoder_position = torch.nn.Embedding(cfg.seqlen, cfg.d_model)
        self.embed_dropout = torch.nn.Dropout(p=0.0)

        self.encoders = torch.nn.ModuleList(
            [EncoderLayer(
                cfg.d_model, cfg.num_heads, cfg.d_ff,
                cfg.dropout_rate, cfg.dropout_rate, cfg.dropout_rate,
                cfg.layer_norm_eps
            ) for _ in range(cfg.num_layers)]
        )


        self.decoder_position = torch.nn.Embedding(cfg.seqlen, cfg.d_model)
        self.decoders = torch.nn.ModuleList(
            [DecoderLayer(
                cfg.d_model, cfg.num_heads, cfg.d_ff,
                cfg.dropout_rate, cfg.dropout_rate, cfg.dropout_rate,
                cfg.layer_norm_eps
            ) for _ in range(cfg.num_layers)]
        )

        self.final_layernorm = torch.nn.LayerNorm(cfg.d_model)

    def forward(self, 
                input_ids: torch.LongTensor,
                input_position_ids: torch.LongTensor,
                decoder_input_ids: torch.LongTensor,
                decoder_position_ids: torch.LongTensor):

        # encoder input ids
        embed = torch.nn.functional.embedding(
            input_ids, self.embedw, padding_idx=None,
            max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False
        )
        pos_embed = self.encoder_position(input_position_ids)
        embed = embed + pos_embed
        embed = self.embed_dropout(embed)

        # encoder
        enc = embed
        for encoder in self.encoders:
            cube.runtime.function.anchor('encoder start')
            enc = encoder(enc)

        # decoder input ids
        embed = torch.nn.functional.embedding(
            decoder_input_ids, self.embedw, padding_idx=None,
            max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False
        )
        pos_embed = self.decoder_position(decoder_position_ids)
        embed = embed + pos_embed
        embed = self.embed_dropout(embed)

        # decoder
        dec = embed
        for decoder in self.decoders:
            cube.runtime.function.anchor('decoder start')
            dec = decoder(dec, enc)
        
        # simplify for finetuning
        loss = torch.sum(dec)
        return loss
        

class mT5DataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, bs: int, cfg: Config):
        self.cfg = cfg
        super().__init__(bs, [0] * 4)
        self.sample = None
        self.set_batch_size(bs)

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample
    
    def set_batch_size(self, bs: int):
        self.batch_size = bs
        encoder_input_ids = torch.randint(
            0, self.cfg.vocab_size,
            size=(self.batch_size, self.cfg.seqlen),
            dtype=torch.int64, device=torch.cuda.current_device()
        )
        encoder_position_ids = torch.arange(
            0, self.cfg.seqlen, dtype=torch.int64, device=torch.cuda.current_device()
        ).repeat(self.batch_size).view(self.batch_size, -1)

        decoder_input_ids = torch.randint(
            0, self.cfg.vocab_size,
            size=(self.batch_size, self.cfg.seqlen),
            dtype=torch.int64, device=torch.cuda.current_device()
        )

        decoder_position_ids = torch.arange(
            0, self.cfg.seqlen, dtype=torch.int64, device=torch.cuda.current_device()
        ).repeat(self.batch_size).view(self.batch_size, -1)

        self.sample = (
            encoder_input_ids,
            encoder_position_ids,
            decoder_input_ids,
            decoder_position_ids
        )
