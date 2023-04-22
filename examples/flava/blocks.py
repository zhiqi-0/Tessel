import torch
import cube
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Any
import math


@cube.graph.parser.register('L^ N E^, (h+ d^ 3) E^, (h+ d^ 3), E^ (h+ d^) -> L^ N E^', name='self_attention')
def self_attention(query: torch.Tensor, 
                   qkv_proj: torch.Tensor, qkv_bias: torch.Tensor,
                   out_proj: torch.Tensor,
                   h: int, scale: float, dropout_p: float, mask: bool = False):
    num_head = h
    L, N = query.size(0), query.size(1)
    dim_head = qkv_proj.size(0) // num_head // 3

    qkv = torch.nn.functional.linear(query, qkv_proj, qkv_bias) # L N E, (h d 3) E -> L N (h d 3)
    qkv = qkv.view(L, N, num_head * dim_head, 3) # L N (h d 3) -> L N (h d) 3
    q, k, v = qkv.chunk(3, dim=-1)  # L N (3 h d) -> L N (h d), L N (h d), L N (h d)
    q = q.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    k = k.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    v = v.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d

    # preallocating input tensor: (N h) L L
    matmul_input_buffer = torch.empty([N * h, L, L], dtype=query.dtype, device=query.device)
    # L (N h) d, L (N h) d -> (N h) L L
    attn = torch.baddbmm(
        matmul_input_buffer,
        q.transpose(0, 1),  # (N h) L d
        k.transpose(0, 1).transpose(1, 2), # (N h) d L
        beta=0.0, alpha=scale
    )
    # ======== replace the semantic into more efficient implementation ============

    # attention mask
    if mask: # (N h) L L -> (N h) L L
        attn = attn.view(N, num_head, L, L)
        ones = torch.ones((N, L, L), device=attn.device)
        amask = torch.tril(ones)
        amask = amask.view(N, 1, L, L)
        amask = (amask < 0.5)
        attn = attn.masked_fill_(amask, -10000.0)
        attn = attn.view((N * num_head), L, L)

    attn = torch.nn.functional.softmax(attn, dim=-1) # (N h) L L -> (N h) L L
    attn = torch.nn.functional.dropout(attn, dropout_p, True, False) # (N h) L L -> (N h) L L
    v = v.transpose(0, 1)  # L (N h) d -> (N h) L d
    output = torch.bmm(attn, v) # (N h) L L, (N h) L d -> (N h) L d
    output = output.transpose(0, 1).contiguous()     # (N h) L d -> L (N h) d
    output = output.view(L, N, num_head * dim_head)  # (N h) L d -> L N (h d)
    output = torch.nn.functional.linear(output, out_proj) # L N (h d), E E  -> L N E
    return output


class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, inner_dim: int, dropout: float = 0.0):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout
        # QKV [(h d 3), E]
        self.qkv_proj = torch.nn.Parameter(torch.empty(3 * inner_dim, embed_dim))
        self.qkv_bias = torch.nn.Parameter(torch.empty(3 * inner_dim))
        # Out
        self.out_proj = torch.nn.Parameter(torch.empty(embed_dim, inner_dim))
        self.out_bias = torch.nn.Parameter(torch.empty(embed_dim))

    def forward(self, query):
        attn = self_attention(
            query, self.qkv_proj, self.qkv_bias,
            self.out_proj,
            self.num_heads, self.scaling, self.dropout_p, mask=False
        )
        attn = attn + self.out_bias
        return attn


@cube.graph.parser.register('L^ N E^, H+ E^, H+, E^ H+ -> L^ N E^', name='feedforward')
def feedforward(x: torch.Tensor,
                proj1: torch.Tensor, proj1_bias: torch.Tensor,
                proj2: torch.Tensor,
                dropout: float,
                is_training: bool = True) -> torch.Tensor:
    x = torch.nn.functional.linear(x, proj1, proj1_bias)
    x = torch.nn.functional.gelu(x)
    x = torch.nn.functional.dropout(x, dropout, is_training, False)
    x = torch.nn.functional.linear(x, proj2, None)
    return x


class MLP(torch.nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.proj1 = torch.nn.Parameter(torch.empty((hidden_dim, embed_dim)))
        self.proj1_bias = torch.nn.Parameter(torch.empty((hidden_dim,)))
        self.proj2 = torch.nn.Parameter(torch.empty((embed_dim, hidden_dim)))
        self.proj2_bias = torch.nn.Parameter(torch.empty((embed_dim,)))
        self.dropout = dropout

    def forward(self, x: torch.Tensor):
        x = feedforward(x, self.proj1, self.proj1_bias,
                        self.proj2, self.dropout, self.training)
        x = x + self.proj2_bias
        return x


class TransformerLayer(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int,
                 attn_hidden_dim: int, ffn_hidden_dim: int,
                 dropout: float = 0.0, atten_dropout: float = 0.0, activation_dropout: float = 0.0):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            embed_dim, num_heads, attn_hidden_dim, atten_dropout
        )
        self.self_attn_layer_norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.mlp = MLP(embed_dim, ffn_hidden_dim, activation_dropout)
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: N L E
        x = x.transpose(0, 1)  # N L E -> L N E
        
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual
        
        x = x.transpose(0, 1)  # L N E -> L N E
        return x


class TransformerEncoder(torch.nn.Module):

    def __init__(
        self, 
        n_layer: int, 
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(n_layer):
            self.layers.append(TransformerLayer(
                d_model, n_head, d_model, dim_feedforward,
                dropout, dropout, dropout
            ))
    
    def forward(self, x: torch.Tensor):
        # x: N L E
        for layer in self.layers:
            cube.runtime.function.anchor('encoder starts')
            x = layer(x)
        return x


@cube.graph.parser.register('N hidden ps^ ps^, 1 1 hidden, 1 np^ hidden -> N np^ hidden', name='cls_embedding')
def cls_embedding(embeddings: torch.Tensor, cls_token: torch.Tensor, position_embeddings: torch.Tensor):
    # embeddings: N hidden ps ps
    # cls_token: 1 1 hidden
    # position_embeddings: 1 np hidden
    embeddings = embeddings.flatten(2).transpose(1, 2)
    batch_size = embeddings.shape[0]
    cls_tokens = cls_token.expand(batch_size, -1, -1)
    embeddings = torch.cat((cls_tokens, embeddings), dim=1)
    return embeddings


class ImageEmbeddings(torch.nn.Module):

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, hidden_size))

        # patch embedding
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embeddings = torch.nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)

        self.position_embeddings = torch.nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )
        # self.position_embeddings = torch.nn.Parameter(
        #     torch.zeros(1, self.num_patches + 1, hidden_size)
        # )

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        # N C H W -> N hidden ps ps
        embeddings = self.patch_embeddings(pixel_values)
        # N hidden ps ps -> N hidden np -> N np hidden
        # embeddings = embeddings.flatten(2).transpose(1, 2)
        # add the [CLS] token to the embedded patch tokens
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # N np hidden -> N 1+np hidden
        # embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # embeddings = embeddings + self.position_embeddings
        embeddings = cls_embedding(embeddings, self.cls_token, self.position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Pooler(nn.Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        # N L E -> N E  get CLS token hidden state
        first_token_tensor = hidden_states[:, 0]
        # N E -> N E
        pooled_output = self.dense(first_token_tensor)
        # N E -> N E
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ImageTransformer(nn.Module):

    def __init__(
        self,
        embeddings: nn.Module,
        encoder: nn.Module,
        layernorm: nn.Module,
        pooler: nn.Module,
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        self.layernorm = layernorm
        self.pooler = pooler


    def forward(self, pixel_values: Tensor) -> Tensor:
        # N C H W -> N L C
        embedding_output = self.embeddings(pixel_values)
        output = self.encoder(embedding_output)
        output = self.layernorm(output)
        return output


def flava_image_encoder(
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    num_hidden_layers: int = 12,
    use_image_masking: bool = False,
    dropout: float = 0.0,
    intermediate_size: int = 3072,
    intermediate_activation: Callable[..., nn.Module] = nn.GELU,
    layer_norm_eps: float = 1e-12,
    image_size: int = 224,
    patch_size: int = 16,
    num_channels: int = 3,
) -> ImageTransformer:

    embeddings = ImageEmbeddings(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        hidden_size=hidden_size,
        hidden_dropout_prob=dropout,
        # use_image_masking=use_image_masking,
    )
    encoder = TransformerEncoder(
        n_layer=num_hidden_layers,
        d_model=hidden_size,
        n_head=num_attention_heads,
        dim_feedforward=intermediate_size,
        dropout=dropout,
    )

    layernorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
    pooler = Pooler(hidden_size=hidden_size)

    return ImageTransformer(
        embeddings=embeddings,
        encoder=encoder,
        layernorm=layernorm,
        pooler=pooler,
    )

@cube.graph.parser.register('N L^ -> L^', name='get_position_ids')
def get_position_ids(input_ids: torch.Tensor):
    return torch.arange(
        input_ids.size(1), dtype=torch.long, device=input_ids.device)

@cube.graph.parser.register('N L -> N L')
def zero_like(input_ids: torch.Tensor):
    return torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)


class BERTTextEmbeddings(nn.Module):
    """Construct word, position, and token type embeddings following BERT, similar to HuggingFace BertEmbeddings
    Attributes:
        hidden_size (int): size of embedding space. Default is 768.
        vocab_size (int): size of vocabulary. Default is 30522.
        pad_token_id (int): id used for padding token. Default is 0.
        max_position_embeddings (int): the highest position id number, or max sequence length. Default is 512.
        type_vocab_size (int): the highest token type id number. Default is 2.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        dropout (float): dropout probability after all embeddings and layernorm
        offset_pos_ids (bool): if True, shift position ids by one for the padding token. Used in RoBERTa.
            Default is False.
    Args:
        input_ids (Tensor, optional): Tensor of input vocab token ids of shape [batch, seq_len].
        token_type_ids (Tensor, optional): Tensor of input token type ids of shape [batch, seq_len]. In BERT,
            used to indicate whether a word is in sentence A or B for next sentence prediction
        position_ids (Tensor, optional): Tensor of input position ids of shape [batch, seq_len]
        inputs_embeds (Tensor, optional): Tensor of input embeddings of shape [batch, hidden_size],
            if embeddings are calculated elsewhere
    """

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        pad_token_id: int = 0,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.0,
        offset_pos_ids: bool = False,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.pad_token_id = pad_token_id
        self.offset_pos_ids = offset_pos_ids

    def create_position_ids_from_input_ids(self, input_ids: Tensor) -> Tensor:
        """
        Replace non-padding symbols with their position numbers.
        Position numbers begin at pad_token_id+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.
        Inputs: input_ids (Tensor): Tensor from which to create position IDs.
                pad_token_id (int): Padding index
                    (determines starting point of position IDs).
        """
        mask = input_ids.ne(self.pad_token_id).int()
        incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
        return incremental_indices.long() + self.pad_token_id

    def forward(self, input_ids: Tensor) -> Tensor:
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        inputs_embeds = self.word_embeddings(input_ids)
        
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = get_position_ids(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        token_type_ids = zero_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # N L E
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTTextEncoder(nn.Module):
    """
    General text transformer encoder with embeddings, following BERT.
    Can be constructed with any user-provided embeddings and encoder.
    Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L870
    Attributes:
        embeddings (nn.Module): Module that projects text token ids into embeddings.
            See :py:class: `torchmultimodal.modules.layers.text_embedding.BERTTextEmbeddings` for interface.
        encoder (nn.Module): Module for transformer encoder. See :py:class:
            `torchmultimodal.modules.layers.transformer.TransformerEncoder` for interface.
        layernorm (nn.Module, optional): Module for layernorm to be applied after encoder. Defaults to ``None``.
        pooler (nn.Module, optional): Module for pooler to be applied after layernorm. Defaults to ``None``.
        weight_init_fn (Callable, optional): function for custom weight initialization of both the transformer
            encoder and embeddings. See :py:func: `torchmultimodal.models.flava.transformer.init_transformer_weights`
            as an example. Defaults to ``None``.
    Args:
        input_ids (Tensor, optional): Tensor of input vocab token ids of shape [batch, seq_len].
        attention_mask (Tensor, optional): Tensor indicating which tokens to attend to, shape [batch, seq_len]
        token_type_ids (Tensor, optional): Tensor of input token type ids of shape [batch, seq_len]. In BERT,
            used to indicate whether a word is in sentence A or B for next sentence prediction
        position_ids (Tensor, optional): Tensor of input position ids of shape [batch, seq_len]
        inputs_embeds (Tensor, optional): Tensor of input embeddings of shape [batch, hidden_size],
            if embeddings are calculated elsewhere
    Raises:
        ValueError: if input_ids and inputs_embeds are both ``None``.
    """

    def __init__(
        self,
        embeddings: nn.Module,
        encoder: nn.Module,
        layernorm: Optional[nn.Module] = None,
        pooler: Optional[nn.Module] = None,
        weight_init_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.embeddings = embeddings
        self.encoder = encoder
        self.layernorm = layernorm
        self.pooler = pooler

        if weight_init_fn:
            self.apply(weight_init_fn)

    def forward(self, input_ids: Tensor,
        # attention_mask: Optional[Tensor] = None,
        # token_type_ids: Optional[Tensor] = None,
        # position_ids: Optional[Tensor] = None,
        # inputs_embeds: Optional[Tensor] = None,
        # return_attn_weights: bool = False,
        # return_hidden_states: bool = False,
    ):
        embedding_output = self.embeddings(input_ids)
        last_hidden_state = self.encoder(embedding_output)
        if self.layernorm is not None:
            last_hidden_state = self.layernorm(last_hidden_state)
        # if self.pooler is not None:
        #     last_hidden_state = self.pooler(last_hidden_state)
        return last_hidden_state


def flava_text_encoder(
    # TransformerEncoder params
    num_hidden_layers: int = 12,
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
    intermediate_activation: Callable[..., nn.Module] = nn.GELU,
    layer_norm_eps: float = 1e-12,
    dropout: float = 0.0,
    # TextEmbeddings params
    vocab_size: int = 30522,
    pad_token_id: int = 0,
    type_vocab_size: int = 2,
    max_position_embeddings: int = 512,
    # TextEncoder params
    initializer_range: float = 0.02,
) -> BERTTextEncoder:

    embeddings = BERTTextEmbeddings(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        type_vocab_size=type_vocab_size,
        max_position_embeddings=max_position_embeddings,
        layer_norm_eps=layer_norm_eps,
        dropout=dropout,
    )

    encoder = TransformerEncoder(
        n_layer=num_hidden_layers,
        d_model=hidden_size,
        n_head=num_attention_heads,
        dim_feedforward=intermediate_size,
        # activation=intermediate_activation,
        # layer_norm_eps=layer_norm_eps,
        dropout=dropout,
        # norm_first=True,
    )

    layernorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
    pooler = Pooler(hidden_size=hidden_size)

    return BERTTextEncoder(
        embeddings=embeddings,
        encoder=encoder,
        layernorm=layernorm,
        pooler=pooler,
        # weight_init_fn=weight_init_fn,
    )


class FLAVATransformerWithoutEmbeddings(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        layernorm: nn.Module,
        pooler: nn.Module,
        hidden_size: int = 768,
        weight_init_fn: Optional[Callable] = None,
        initializer_range: float = 0.02,
        use_cls_token: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.encoder = encoder
        self.layernorm = layernorm
        self.pooler = pooler
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.cls_token = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        # hidden_states: N (L1+L2) E
        sequence_output = self.encoder(hidden_states)
        sequence_output = self.layernorm(sequence_output)
        return sequence_output


def flava_multimodal_encoder(
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    num_hidden_layers: int = 12,
    dropout: float = 0.0,
    intermediate_size: int = 3072,
    intermediate_activation: Callable[..., nn.Module] = nn.GELU,
    layer_norm_eps: float = 1e-12,
) -> FLAVATransformerWithoutEmbeddings:
    encoder = TransformerEncoder(
        n_layer=num_hidden_layers,
        d_model=hidden_size,
        n_head=num_attention_heads,
        dim_feedforward=intermediate_size,
        dropout=dropout,
    )
    layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
    pooler = Pooler(hidden_size=hidden_size)

    return FLAVATransformerWithoutEmbeddings(
        encoder=encoder, layernorm=layernorm, pooler=pooler, hidden_size=hidden_size
    )