import math
import types
import warnings
import typing

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)
from transformers.utils import ModelOutput

def _update_model_kwargs_for_generation(
    self,
    outputs: ModelOutput,
    model_kwargs:typing.Dict[str, typing.Any],
    is_encoder_decoder: bool = False,
    standardize_cache_format: bool = False,
) -> typing.Dict[str, typing.Any]:

    # update past_key_values
    model_kwargs["past_key_values"] = self._extract_past_from_model_output(
        outputs, standardize_cache_format=standardize_cache_format
    )
    if getattr(outputs, "state", None) is not None:
        model_kwargs["state"] = outputs.state

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat(
            [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
        )

    if not is_encoder_decoder:
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if len(attention_mask.shape) == 4:
                # NOTE: this if-statement loop can be refactored to avoid redundant code
                next_position_id = (
                    model_kwargs["position_ids"][0][-1] + 1
                    if "position_ids" in model_kwargs
                    else attention_mask.shape[-1]
                )

                # given an attention_mask of shape (bsz, 1, tgt_seq_len, src_seq_len) in model kwargs, generate
                # new 2D attention_mask of ones with shape (bsz, src_seq_len+1) for subsequent forward() calls
                # EDIT:
                # Convert 4D mask to 2D mask, while recording any padding tokens
                model_kwargs["attention_mask"] = (
                    torch.any(attention_mask[0][0] != 0, dim=0)
                    .to(attention_mask.dtype)
                    .unsqueeze(0)
                )
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        model_kwargs["attention_mask"],
                        model_kwargs["attention_mask"].new_ones(
                            (model_kwargs["attention_mask"].shape[0], 1)
                        ),
                    ],
                    dim=-1,
                )  # Add a new 1 to the end for the newly generated token
                model_kwargs["position_ids"] = torch.tensor([[next_position_id]]).to(
                    model_kwargs["position_ids"].device
                )
            else:
                # Extend the length of the attention mask by 1 to reflect the fact that we have generated one additional token
                assert len(attention_mask.shape) == 2
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )
                if "position_ids" in model_kwargs:
                    next_position_id = (
                        model_kwargs["position_ids"][0][-1] + 1
                        if "position_ids" in model_kwargs
                        else attention_mask.shape[-1]
                    )
                    model_kwargs["position_ids"] = torch.tensor(
                        [[next_position_id]]
                    ).to(model_kwargs["position_ids"].device)
    else:
        # update decoder attention mask
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [
                    decoder_attention_mask,
                    decoder_attention_mask.new_ones(
                        (decoder_attention_mask.shape[0], 1)
                    ),
                ],
                dim=-1,
            )

    return model_kwargs

# Used for transformers version git+https://github.com/huggingface/transformers@1c31b7aa3bb4e7ef24c77596d2a76f45a770159f
def get_2D_attention_accepting_model_llama(model):
    from peft import PeftModel  # Safe import. If not installed, you'd need try-except.
    
    if isinstance(model, PeftModel):
        model.base_model.model._update_model_kwargs_for_generation = types.MethodType(
            _update_model_kwargs_for_generation, model
        )
    else:
        model._update_model_kwargs_for_generation = types.MethodType(
            _update_model_kwargs_for_generation, model
        )
    return model
