"""Text tower used by the alignment model (moved unchanged from the research code's model file; checkpoints store
state dicts, so the module path does not matter for loading)."""
import torch
from torch import nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """Qwen3-Embedding text tower: last valid token -> L2 normalise -> trainable 512-d projection.

    Left and right padding are both supported. Texts up to the model's position limit are encoded directly; longer texts must
    be split by the caller (chunk_owner / chunk_weights average the chunks); nothing is truncated silently. Class queries use
    the Qwen Instruct/Query format; paired captions are encoded as documents without an instruction.
    """

    def __init__(self, config, backbone=None):
        super().__init__()
        self.text_batch = config.text_batch
        if backbone is None:
            from transformers import AutoModel
            backbone, info = AutoModel.from_pretrained(config.text_path, trust_remote_code=False,
                local_files_only=config.local_files_only, output_loading_info=True)
            if info['missing_keys'] or info.get('mismatched_keys') or info.get('error_msgs'):
                raise ValueError(f'Text backbone was not fully loaded: {info}')
        self.backbone = backbone
        if backbone.config.hidden_size != config.text_feature_dim:
            raise ValueError('Text feature dimension differs from config')
        if config.text_gradient_checkpointing and hasattr(backbone, 'gradient_checkpointing_enable'):
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.projection = nn.Linear(config.text_feature_dim, config.dim, bias=False)

    @staticmethod
    def format_query(description, instruction):
        """Qwen Instruct/Query format for class queries; paired captions are documents without an instruction."""
        return f'Instruct: {instruction}\nQuery:{description}'

    def forward(self, input_ids, attention_mask, chunk_owner=None, chunk_weights=None, num_texts=None):
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or not len(input_ids):
            raise ValueError('Text ids/mask must be nonempty [chunks,length]')
        if input_ids.dtype != torch.long or not attention_mask.bool().any(1).all():
            raise ValueError('Text requires integer token ids and nonempty chunks')
        if input_ids.shape[1] > self.backbone.config.max_position_embeddings:
            raise ValueError('Text exceeds Qwen positions; split it into chunks before forward')
        pooled = []
        for ids, mask in zip(input_ids.split(self.text_batch), attention_mask.split(self.text_batch)):
            # Right-padded training documents can be trimmed independently per microbatch.
            # Valid token positions stay unchanged; left-padding callers retain their old layout.
            if torch.equal(mask.bool(),mask.bool().cummin(1).values):
                length=int(mask.sum(1).max())
                ids=ids[:,:length];mask=mask[:,:length]
            hidden = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            positions = torch.arange(ids.shape[1], device=ids.device).expand_as(mask)
            last = positions.masked_fill(~mask.bool(), -1).max(1).values
            pooled.append(F.normalize(hidden[torch.arange(len(ids), device=ids.device), last].float(), dim=-1))
        cls = torch.cat(pooled)
        if chunk_owner is not None:
            if (chunk_owner.dtype != torch.long or chunk_owner.shape != (len(cls),)
                    or num_texts is None or num_texts < 1):
                raise ValueError('Chunk owner must index every chunk into num_texts documents')
            if (chunk_owner < 0).any() or (chunk_owner >= num_texts).any():
                raise ValueError('Text chunk owner is outside document range')
            if chunk_weights is None or chunk_weights.shape != chunk_owner.shape:
                raise ValueError('Chunked text requires effective nonduplicate token weights')
            weights = chunk_weights.to(cls.dtype)
            if not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError('Text chunk weights must be finite and positive')
            total = cls.new_zeros((num_texts,)).index_add(0, chunk_owner, weights)
            if (total == 0).any():
                raise ValueError('Every document must have a text chunk')
            cls = cls.new_zeros((num_texts, cls.shape[1])).index_add(0, chunk_owner, cls * weights[:, None])
            cls = cls / total[:, None]
        elif chunk_weights is not None or num_texts is not None:
            raise ValueError('Chunk aggregation requires chunk_owner')
        return F.normalize(self.projection(cls.to(self.projection.weight.dtype)).float(), dim=-1)
