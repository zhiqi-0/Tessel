"""
https://github.com/facebookresearch/multimodal/blob/96970a9a3f92847146f1a3ed16c0b697f4dad784/torchmultimodal/models/flava/model.py#L418

"""

from typing import Callable, Any, Optional

from examples.flava.blocks import flava_image_encoder, flava_text_encoder, flava_multimodal_encoder

import torch.nn as nn
from torch import Tensor
import torch

import cube
from cube.profiler.timer import CudaTimer

from dataclasses import dataclass


@dataclass
class Config:

    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12

    image_size: int = 480 # 512 # 224
    patch_size: int = 16
    num_channels: int = 3

    vocab_size: int = 30522
    seq_len: int = 1024 # 512

    dropout: float = 0.2


class FLAVAModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.image_encoder = flava_image_encoder(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_heads,
            num_hidden_layers=cfg.num_layers,
            use_image_masking=False,
            dropout=cfg.dropout,
            intermediate_size=cfg.hidden_size * 4,
            intermediate_activation=nn.GELU,
            layer_norm_eps=1e-12,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            num_channels=cfg.num_channels,
        )
        self.text_encoder = flava_text_encoder(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_heads,
            num_hidden_layers=cfg.num_layers,
            dropout=cfg.dropout,
            intermediate_size=cfg.hidden_size * 4,
            intermediate_activation=nn.GELU,
            layer_norm_eps=1e-12,
            vocab_size=cfg.vocab_size,
            pad_token_id=0,
            type_vocab_size=2,
            max_position_embeddings=cfg.seq_len,
        )
        self.mm_encoder = flava_multimodal_encoder(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_heads,
            num_hidden_layers=cfg.num_layers // 2,
            dropout=cfg.dropout,
            intermediate_size=cfg.hidden_size * 4,
            intermediate_activation=nn.GELU,
            layer_norm_eps=1e-12,
        )
        self.image_to_mm_projection = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.text_to_mm_projection = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        # self.text_projection = text_projection
        # self.image_projection = image_projection

    def forward(self, image: Tensor, text: Tensor) -> Tensor:
        """
        image: N C H W
        text: N L
        """
        # N L1 E
        # CudaTimer().start('image')
        cube.runtime.function.anchor('image')
        image_hidden = self.image_encoder(image)
        image_hidden = self.image_to_mm_projection(image_hidden)
        # CudaTimer().stop('image')

        # N L2 E
        # CudaTimer().start('text')
        cube.runtime.function.anchor('text')
        text_hidden = self.text_encoder(text)
        text_hidden = self.text_to_mm_projection(text_hidden)
        # CudaTimer().stop('text')

        # N (L1+L2) E
        # CudaTimer().start('mm')
        cube.runtime.function.anchor('mm')
        fused_state = torch.cat([image_hidden, text_hidden], dim=1)
        multimodal_logits = self.mm_encoder(fused_state)
        # CudaTimer().stop('mm')
        return multimodal_logits


class ImageTextDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int, dtype: torch.dtype, cfg: Config):
        super().__init__(batch_size, [0, 0])
        self.img_size = cfg.image_size
        self.seqlen = cfg.seq_len
        self.dtype = dtype
        self.sample = self.random_sample()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.sample
    
    def random_sample(self) -> torch.Tensor:
        sample = (
            torch.randn((self.batch_size, 3, self.img_size, self.img_size),
                        dtype=self.dtype, device=torch.cuda.current_device()),
            torch.randint(0, 1000, (self.batch_size, self.seqlen), dtype=torch.long, 
                          device=torch.cuda.current_device())
        )
        return sample
        
    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        self._sample = self.random_sample()