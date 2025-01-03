import typing

import torch
import transformers
import transformers.modeling_outputs
import transformers.models.gpt2.modeling_gpt2
from torch import nn

# GPT2Attention, GPT2Block

# import logging
# logger = logging.get_logger(__name__)

# causal_mask = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(1, 1, max_positions, max_positions)
VERBOSE = False


class GPT2Attention_Attention(transformers.models.gpt2.modeling_gpt2.GPT2Attention):
    def forward(
        self,
        hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]],
        layer_past: typing.Optional[typing.Tuple[torch.Tensor]] = None,
        attention_mask: typing.Optional[torch.FloatTensor] = None,
        head_mask: typing.Optional[torch.FloatTensor] = None,
        encoder_hidden_states: typing.Optional[torch.Tensor] = None,
        encoder_attention_mask: typing.Optional[torch.FloatTensor] = None,
        use_cache: typing.Optional[bool] = False,
        output_attentions: typing.Optional[bool] = False,
        attention_mask_2d=None,
    ) -> typing.Tuple[typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(
                self.split_size, dim=2
            )
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query, key, value, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = self._attn(
                query,
                key,
                value,
                attention_mask,
                head_mask,
                attention_mask_2d=attention_mask_2d,
            )
            if VERBOSE:
                alt_attn_output, alt_attn_weights = self._attn(
                    query, key, value, attention_mask, head_mask, attention_mask_2d=None
                )
                print(
                    f"Finish self._attn() with attn_output.shape={attn_output.shape}, attn_weights.shape={attn_weights.shape}"
                )
                print(
                    f"GPT2Attention.forward(), alt_attn_output ?= attn_output: {torch.allclose(alt_attn_output, attn_output)}, alt_attn_weights ?= attn_weights: {torch.allclose(alt_attn_weights, attn_weights)}"
                )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
        attention_mask_2d=None,
    ):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [],
                value.size(-1) ** 0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            # Note: this is where the 2D causal mask is generated, we override this to replace the causual mask with attention_mask_2D, if it is given
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[
                :, :, key_length - query_length : key_length, :key_length
            ]
            """
            Causal mask looks similar to
            tensor([[[[ True, False, False, False],
                [ True,  True, False, False],
                [ True,  True,  True, False],
                [ True,  True,  True,  True]]]])
            """
            # Causal_mask has shape (1, 1, query_length, key_length)
            if VERBOSE:
                print(
                    f"Key length: {key_length}, Query length: {query_length}, causal_mask.shape: {causal_mask.shape}, attention_mask.shape {attention_mask_2d.shape if attention_mask_2d is not None else None}"
                )
                if attention_mask_2d is not None:
                    print(f"Given attention_mask_2d of shape {attention_mask_2d.shape}")
            if (
                attention_mask_2d is not None
                and attention_mask_2d.shape == causal_mask.shape
            ):
                causal_mask = attention_mask_2d
                if VERBOSE:
                    print(f"Override causal_mask with shape {causal_mask.shape}")
                    print(
                        f"Override attention_mask with shape {attention_mask_2d.shape}"
                    )
            else:
                if VERBOSE:
                    print(
                        f"Use default causual attention_mask with shape {causal_mask.shape}, attention_mask_2D is None {attention_mask_2d is None}"
                    )
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
                attn_weights.device
            )
            alternate_attn_weights = torch.where(
                self.bias[:, :, key_length - query_length : key_length, :key_length],
                attn_weights.to(attn_weights.dtype),
                mask_value,
            )
            attn_weights = torch.where(
                causal_mask, attn_weights.to(attn_weights.dtype), mask_value
            )
            if VERBOSE:
                print(
                    f"alternate_attn_weights ?= attn_weights: {torch.allclose(alternate_attn_weights, attn_weights)}"
                )

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
            if VERBOSE:
                print(
                    f".attn() attn_weights.shape={attn_weights.shape}, attention_mask.shape: {attention_mask.shape}"
                )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


class GPT2BlockAttention(transformers.models.gpt2.modeling_gpt2.GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.attn = GPT2Attention_Attention(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]],
        layer_past: typing.Optional[typing.Tuple[torch.Tensor]] = None,
        attention_mask: typing.Optional[torch.FloatTensor] = None,
        head_mask: typing.Optional[torch.FloatTensor] = None,
        encoder_hidden_states: typing.Optional[torch.Tensor] = None,
        encoder_attention_mask: typing.Optional[torch.FloatTensor] = None,
        use_cache: typing.Optional[bool] = False,
        output_attentions: typing.Optional[bool] = False,
        attention_mask_2d=None,
    ) -> typing.Union[
        typing.Tuple[torch.Tensor],
        typing.Optional[
            typing.Tuple[torch.Tensor, typing.Tuple[torch.FloatTensor, ...]]
        ],
    ]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            attention_mask_2d=attention_mask_2d,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = (
                outputs + cross_attn_outputs[2:]
            )  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2ModelAttention(GPT2Model):
    """
    Override the forward method to accept a 2D attention mask, rather than a 1D attention mask
    """

    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList(
            [
                GPT2BlockAttention(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        input_ids: typing.Optional[torch.LongTensor] = None,
        past_key_values: typing.Optional[
            typing.Tuple[typing.Tuple[torch.Tensor]]
        ] = None,
        attention_mask: typing.Optional[torch.FloatTensor] = None,
        token_type_ids: typing.Optional[torch.LongTensor] = None,
        position_ids: typing.Optional[torch.LongTensor] = None,
        head_mask: typing.Optional[torch.FloatTensor] = None,
        inputs_embeds: typing.Optional[torch.FloatTensor] = None,
        encoder_hidden_states: typing.Optional[torch.Tensor] = None,
        encoder_attention_mask: typing.Optional[torch.FloatTensor] = None,
        use_cache: typing.Optional[bool] = None,
        output_attentions: typing.Optional[bool] = None,
        output_hidden_states: typing.Optional[bool] = None,
        return_dict: typing.Optional[bool] = None,
        attention_mask_2d=None,
    ) -> typing.Union[
        typing.Tuple,
        transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
    ]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None or past_length > 0:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(
                        past_state.to(hidden_states.device) for past_state in layer_past
                    )
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    attention_mask_2d=attention_mask_2d,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (
                        outputs[3 if use_cache else 2],
                    )

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class GPT2LMHeadModelAttention(transformers.transformers.utils):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2ModelAttention(config)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        attention_mask_2d = kwargs.get("attention_mask_2d", None)
        model_inputs.update({"attention_mask_2d": attention_mask_2d})
        model_inputs.update({"position_ids": kwargs.get("position_ids", None)})
        return model_inputs

    def forward(
        self,
        input_ids: typing.Optional[torch.LongTensor] = None,
        past_key_values: typing.Optional[
            typing.Tuple[typing.Tuple[torch.Tensor]]
        ] = None,
        attention_mask: typing.Optional[torch.FloatTensor] = None,
        token_type_ids: typing.Optional[torch.LongTensor] = None,
        position_ids: typing.Optional[torch.LongTensor] = None,
        head_mask: typing.Optional[torch.FloatTensor] = None,
        inputs_embeds: typing.Optional[torch.FloatTensor] = None,
        encoder_hidden_states: typing.Optional[torch.Tensor] = None,
        encoder_attention_mask: typing.Optional[torch.FloatTensor] = None,
        labels: typing.Optional[torch.LongTensor] = None,
        use_cache: typing.Optional[bool] = None,
        output_attentions: typing.Optional[bool] = None,
        output_hidden_states: typing.Optional[bool] = None,
        return_dict: typing.Optional[bool] = None,
        attention_mask_2d=None,
    ) -> typing.Union[
        typing.Tuple, transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
    ]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *typing.Optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attention_mask_2d=attention_mask_2d,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return transformers.modeling_outputs.CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
