import itertools

import tensorflow as tf
import torch
import torch.nn.functional as F
import transformers

"""
Attention mask:
- 1 for tokens that are **not masked**,
- 0 for tokens that are **masked**.
"""
MAX_NEW_TOKENS = 10


def load_gpt2_model():
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # model = transformers.GPT2LMHeadModelAttention.from_pretrained("gpt2", output_attentions=True)
    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)
    return model, tokenizer, torch_device


modelGPT, tokenizerGPT, torch_device = load_gpt2_model()


# modelGPTOld,tokenizerGPTOld,torch_device = load_gpt2_model()
def get_position_ids(tokA, tokB, tokC, tokD):
    nTokA = len(tokA["attention_mask"][0])
    nTokB = len(tokB["attention_mask"][0])
    nTokC = len(tokC["attention_mask"][0])
    nTokD = len(tokD["attention_mask"][0])
    nTokAll = nTokA + nTokB + nTokC + nTokD
    position_ids = torch.arange(0, nTokAll).unsqueeze(0)
    position_ids[0][nTokA + nTokB : nTokA + nTokB + nTokC] = torch.arange(
        nTokA, nTokA + nTokC
    )
    return position_ids


def get_position_ids_padded(tokA, tokB, tokC, tokD, tokenizer=tokenizerGPT):
    nTokA = len(tokA["attention_mask"][0])
    nTokB = len(tokB["attention_mask"][0])
    nTokC = len(tokC["attention_mask"][0])
    nTokD = len(tokD["attention_mask"][0])
    nTokBCPadded = max(nTokB, nTokC)
    nTokAll = nTokA + 2 * nTokBCPadded + nTokD
    if nTokB < nTokBCPadded:
        # pad tokB
        tokB["input_ids"] = torch.cat(
            (
                tokB["input_ids"][0],
                torch.tensor([tokenizer.pad_token_id] * (nTokBCPadded - nTokB)),
            )
        ).unsqueeze(0)
        tokB["attention_mask"] = torch.cat(
            (tokB["attention_mask"][0], torch.zeros(nTokBCPadded - nTokB))
        ).unsqueeze(0)
    if nTokC < nTokBCPadded:
        # pad tokC
        tokC["input_ids"] = torch.cat(
            (
                tokC["input_ids"][0],
                torch.tensor([tokenizer.pad_token_id] * (nTokBCPadded - nTokC)),
            )
        ).unsqueeze(0)
        # print(tokC['attention_mask'][0],torch.zeros(nTokBCPadded-nTokC))
        tokC["attention_mask"] = torch.cat(
            (tokC["attention_mask"][0], torch.zeros(nTokBCPadded - nTokC))
        ).unsqueeze(0)
    print("Gen tokAll", tokB, tokC, tokenizer.pad_token_id)
    tokAll = {
        "input_ids": torch.cat(
            (
                tokA["input_ids"][0],
                tokB["input_ids"][0],
                tokC["input_ids"][0],
                tokD["input_ids"][0],
            )
        ).unsqueeze(0),
        "attention_mask": torch.cat(
            (
                tokA["attention_mask"][0],
                tokB["attention_mask"][0],
                tokC["attention_mask"][0],
                tokD["attention_mask"][0],
            )
        ).unsqueeze(0),
    }
    tokRev = {
        "input_ids": torch.cat(
            (
                tokA["input_ids"][0],
                tokC["input_ids"][0],
                tokB["input_ids"][0],
                tokD["input_ids"][0],
            )
        ).unsqueeze(0),
        "attention_mask": torch.cat(
            (
                tokA["attention_mask"][0],
                tokC["attention_mask"][0],
                tokB["attention_mask"][0],
                tokD["attention_mask"][0],
            )
        ).unsqueeze(0),
    }
    assert nTokAll == len(tokAll["attention_mask"][0])
    position_ids = torch.cat(
        (
            torch.arange(0, nTokA + nTokBCPadded),
            torch.arange(nTokA, nTokA + nTokBCPadded),
            torch.arange(nTokA + nTokBCPadded, nTokA + nTokBCPadded + nTokD),
        )
    ).unsqueeze(0)

    return position_ids, tokB, tokC, tokAll, tokRev



def get_position_ids_interpolated(tokA, tokB, tokC, tokD):
    nTokA = len(tokA["attention_mask"][0])
    nTokB = len(tokB["attention_mask"][0])
    nTokC = len(tokC["attention_mask"][0])
    nTokD = len(tokD["attention_mask"][0])
    nTokBCMinLength = min(nTokB, nTokC)
    nTokAll = nTokA + nTokB + nTokC + nTokD
    position_ids = torch.arange(0, nTokAll, dtype=torch.float).unsqueeze(0)
    position_ids[0][nTokA + nTokB + nTokC :] = torch.arange(
        nTokA + nTokBCMinLength, nTokA + nTokBCMinLength + nTokD
    )
    if nTokB == nTokC:
        return position_ids  # no interpolation necessary
    if nTokB < nTokC:
        # interplate position ids for tokC
        position_ids[0][nTokA + nTokB : nTokA + nTokB + nTokC] = torch.linspace(
            nTokA, nTokA + nTokB, nTokC
        )
    else:
        # interpolate position ids for tokB
        position_ids[0][nTokA : nTokA + nTokB] = torch.linspace(
            nTokA, nTokA + nTokC - 1, nTokB
        )
        position_ids[0][nTokA + nTokB : nTokA + nTokB + nTokC] = torch.arange(
            nTokA, nTokA + nTokC
        )

    return position_ids


# position_ids,tokB, tokC, tokAll, tokRev=get_position_ids_padded(tokA, tokB, tokC, tokD)
# Get attention mask where tokens in tokC don't attend to tokens in tokB
def get_attention_mask_2d(tokA, tokB, tokC, tokD, tokAll=None):
    """
    Outputs a [1,1,s,s] attention mask where s is the total number of tokens in tokA, tokB, tokC, tokD
        Values are False where tokens are masked, True where attention should be paid
    """
    nTokA, nTokB, nTokC, nTokD = (
        len(tokA["attention_mask"][0]),
        len(tokB["attention_mask"][0]),
        len(tokC["attention_mask"][0]),
        len(tokD["attention_mask"][0]),
    )
    s = nTokA + nTokB + nTokC + nTokD
    if tokAll is None:
        tokAll = {
            "input_ids": torch.cat(
                (
                    tokA["input_ids"][0],
                    tokB["input_ids"][0],
                    tokC["input_ids"][0],
                    tokD["input_ids"][0],
                )
            ).unsqueeze(0),
            "attention_mask": torch.cat(
                (
                    tokA["attention_mask"][0],
                    tokB["attention_mask"][0],
                    tokC["attention_mask"][0],
                    tokD["attention_mask"][0],
                )
            ).unsqueeze(0),
        }
    assert s == len(tokAll["attention_mask"][0])
    causal_mask = torch.tril(torch.ones((s, s), dtype=torch.bool))
    # Set values in causal_mask to False where seqC tokens would attend to seqB tokens
    mask = tf.Variable(causal_mask)

    # Create tf.Variable object filled with False of dimension (nTokC, nTokB)
    mask[nTokA + nTokB : nTokA + nTokB + nTokC, nTokA : nTokA + nTokB].assign(
        tf.zeros([nTokC, nTokB], tf.bool)
    )

    # All tokens should ignore the padding tokens, which occur at indices where tokAll['attention_mask'][0]==0
    paddingTokenIndices = torch.where(tokAll["attention_mask"][0] == 0)[0]
    for ptI in paddingTokenIndices.numpy():
        mask[:, ptI].assign(tf.zeros([s], tf.bool))

    mask = tf.convert_to_tensor([tf.Variable(mask)])
    mask = torch.tensor(mask.numpy())
    mask = mask.view(1, 1, s, s)
    assert mask.shape == (1, 1, s, s)
    return mask


def scores_diff(generated1, generated2, add_epsilon=False):
    # Compute the KL divergence between the softmax of the logits of two outputs of a gpt2 model
    # Requires inputs to be tensors in log-probability space
    if len(generated1.scores) != len(generated2.scores):
        return torch.tensor(float("inf"))
    # Add a 1e-40 constant to avoid log(0) in cases where the softmax of the logits is 0 for some tokens
    if add_epsilon:
        return F.kl_div(
            (1e-40 + F.softmax(torch.stack(generated1.scores, dim=1), dim=-1)).log(),
            (1e-40 + F.softmax(torch.stack(generated2.scores, dim=1), dim=-1)).log(),
            log_target=True,
        )
    else:
        return F.kl_div(
            (F.softmax(torch.stack(generated1.scores, dim=1), dim=-1)).log(),
            (F.softmax(torch.stack(generated2.scores, dim=1), dim=-1)).log(),
            log_target=True,
        )


def testAttentionMaskEdits(seqA, seqB, seqC, seqD, get_position_ids=get_position_ids):
    # Use approach #1 for editing positional encoding
    tokA, tokB, tokC, tokD = tuple(
        [
            tokenizer(input_text, return_tensors="pt").to(torch_device)
            for input_text in [seqA, seqB, seqC, seqD]
        ]
    )
    tokAll = tokenizer(seqA + seqB + seqC + seqD, return_tensors="pt").to(torch_device)
    tokRev = tokenizer(seqA + seqC + seqB + seqD, return_tensors="pt").to(torch_device)
    assert len(tokA["attention_mask"][0]) + len(tokB["attention_mask"][0]) + len(
        tokC["attention_mask"][0]
    ) + len(tokD["attention_mask"][0]) == len(tokAll["attention_mask"][0])
    nTokA, nTokB, nTokC, nTokD = (
        len(tokA["attention_mask"][0]),
        len(tokB["attention_mask"][0]),
        len(tokC["attention_mask"][0]),
        len(tokD["attention_mask"][0]),
    )
    s = nTokA + nTokB + nTokC + nTokD

    # Standard causal mask - ABCD
    causal_mask = torch.tril(torch.ones((s, s), dtype=torch.bool)).view(1, 1, s, s)
    generateStandard = model.generate(
        **tokAll,
        max_new_tokens=MAX_NEW_TOKENS,
        attention_mask_2d=causal_mask,
        return_dict_in_generate=True,
        output_scores=True,
    )
    textStandard = tokenizer.decode(
        generateStandard.sequences[0], skip_special_tokens=True
    )

    # Edited attention mask - ABCD
    attention_mask_2d = get_attention_mask_2d(tokA, tokB, tokC, tokD)
    generateEdited = model.generate(
        **tokAll,
        max_new_tokens=MAX_NEW_TOKENS,
        attention_mask_2d=attention_mask_2d,
        return_dict_in_generate=True,
        output_scores=True,
    )
    textEdited = tokenizer.decode(generateEdited.sequences[0], skip_special_tokens=True)

    # Standard causual mask - ACBD
    generateStandardRev = model.generate(
        **tokRev,
        max_new_tokens=MAX_NEW_TOKENS,
        attention_mask_2d=causal_mask,
        return_dict_in_generate=True,
        output_scores=True,
    )
    textStandardRev = tokenizer.decode(
        generateStandardRev.sequences[0], skip_special_tokens=True
    )

    # Edited attention mask - ACBD
    attention_mask_2dRev = get_attention_mask_2d(tokA, tokC, tokB, tokD)
    generateEditedRev = model.generate(
        **tokRev,
        max_new_tokens=MAX_NEW_TOKENS,
        attention_mask_2d=attention_mask_2dRev,
        return_dict_in_generate=True,
        output_scores=True,
    )
    textEditedRev = tokenizer.decode(
        generateEditedRev.sequences[0], skip_special_tokens=True
    )

    # positional edits - ABCD
    position_ids = get_position_ids(tokA, tokB, tokC, tokD)
    generateEditedPos = model.generate(
        **tokAll,
        max_new_tokens=MAX_NEW_TOKENS,
        attention_mask_2d=causal_mask,
        position_ids=position_ids,
        return_dict_in_generate=True,
        output_scores=True,
    )
    textEditedPos = tokenizer.decode(
        generateEditedPos.sequences[0], skip_special_tokens=True
    )

    # positional edits - ACBD
    position_ids_rev = get_position_ids(tokA, tokC, tokB, tokD)
    generateEditedPosRev = model.generate(
        **tokRev,
        max_new_tokens=MAX_NEW_TOKENS,
        attention_mask_2d=causal_mask,
        position_ids=position_ids_rev,
        return_dict_in_generate=True,
        output_scores=True,
    )
    textEditedPosRev = tokenizer.decode(
        generateEditedPosRev.sequences[0], skip_special_tokens=True
    )

    # Edited causal mask + positional edits - ABCD
    position_ids = get_position_ids(tokA, tokB, tokC, tokD)
    generateEditedAttentionPos = model.generate(
        **tokAll,
        max_new_tokens=MAX_NEW_TOKENS,
        attention_mask_2d=attention_mask_2d,
        position_ids=position_ids,
        return_dict_in_generate=True,
        output_scores=True,
    )
    textEditedAttentionPos = tokenizer.decode(
        generateEditedAttentionPos.sequences[0], skip_special_tokens=True
    )

    # Edited causal mask + positional edits - ACBD
    position_ids_rev = get_position_ids(tokA, tokC, tokB, tokD)
    generateEditedAttentionPosRev = model.generate(
        **tokRev,
        max_new_tokens=MAX_NEW_TOKENS,
        attention_mask_2d=attention_mask_2dRev,
        position_ids=position_ids_rev,
        return_dict_in_generate=True,
        output_scores=True,
    )
    textEditedAttentionPosRev = tokenizer.decode(
        generateEditedAttentionPosRev.sequences[0], skip_special_tokens=True
    )

    # Truncate so generated text doesn't include the input prompt
    s = len(seqA) + len(seqB) + len(seqC) + len(seqD)
    textStandard = textStandard[s:]
    textStandardRev = textStandardRev[s:]
    textEdited = textEdited[s:]
    textEditedRev = textEditedRev[s:]
    textEditedPos = textEditedPos[s:]
    textEditedPosRev = textEditedPosRev[s:]
    textEditedAttentionPos = textEditedAttentionPos[s:]
    textEditedAttentionPosRev = textEditedAttentionPosRev[s:]

    # Compute KL divergence for input sequences
    logit_diffs = {}
    logit_diffs["Standard"] = scores_diff(generateStandard, generateStandardRev)
    logit_diffs["Attention"] = scores_diff(generateEdited, generateEditedRev)
    logit_diffs["Position"] = scores_diff(generateEditedPos, generateEditedPosRev)
    logit_diffs["Attention Position"] = scores_diff(
        generateEditedAttentionPos, generateEditedAttentionPosRev
    )

    return (
        textStandard,
        textStandardRev,
        textEdited,
        textEditedRev,
        textEditedPos,
        textEditedPosRev,
        textEditedAttentionPos,
        textEditedAttentionPosRev,
        logit_diffs,
    )


def testAttentionPosInterpolatedEdits(seqA, seqB, seqC, seqD):
    return testAttentionMaskEdits(
        seqA, seqB, seqC, seqD, get_position_ids=get_position_ids_interpolated
    )


def testAttentionPosPaddingEdits(seqA, seqB, seqC, seqD):
    # Use padding when modifying the positional encoding
    tokA, tokB, tokC, tokD = tuple(
        [
            tokenizer(
                input_text, return_tensors="pt", add_special_tokens=(input_text == seqA)
            ).to(torch_device)
            for input_text in [seqA, seqB, seqC, seqD]
        ]
    )
    tokAll = tokenizer(seqA + seqB + seqC + seqD, return_tensors="pt").to(torch_device)
    tokRev = tokenizer(seqA + seqC + seqB + seqD, return_tensors="pt").to(torch_device)
    assert len(tokA["attention_mask"][0]) + len(tokB["attention_mask"][0]) + len(
        tokC["attention_mask"][0]
    ) + len(tokD["attention_mask"][0]) == len(tokAll["attention_mask"][0])
    nTokA, nTokB, nTokC, nTokD = (
        len(tokA["attention_mask"][0]),
        len(tokB["attention_mask"][0]),
        len(tokC["attention_mask"][0]),
        len(tokD["attention_mask"][0]),
    )
    s = nTokA + nTokB + nTokC + nTokD

    # Standard causal mask - ABCD
    causal_mask = torch.tril(torch.ones((s, s), dtype=torch.bool)).view(1, 1, s, s)
    generateStandard = model.generate(
        **tokAll,
        max_new_tokens=MAX_NEW_TOKENS,
        attention_mask_2d=causal_mask,
        return_dict_in_generate=True,
        output_scores=True,
    )
    textStandard = tokenizer.decode(
        generateStandard.sequences[0], skip_special_tokens=True
    )

    # Edited causal mask - ABCD
    attention_mask_2d = get_attention_mask_2d(tokA, tokB, tokC, tokD)
    generateEdited = model.generate(
        **tokAll,
        max_new_tokens=MAX_NEW_TOKENS,
        attention_mask_2d=attention_mask_2d,
        return_dict_in_generate=True,
        output_scores=True,
    )
    textEdited = tokenizer.decode(generateEdited.sequences[0], skip_special_tokens=True)

    # Standard causual mask - ACBD
    generateStandardRev = model.generate(
        **tokRev,
        max_new_tokens=MAX_NEW_TOKENS,
        attention_mask_2d=causal_mask,
        return_dict_in_generate=True,
        output_scores=True,
    )
    textStandardRev = tokenizer.decode(
        generateStandardRev.sequences[0], skip_special_tokens=True
    )

    # Edited causal mask - ACBD
    attention_mask_2dRev = get_attention_mask_2d(tokA, tokC, tokB, tokD)
    generateEditedRev = model.generate(
        **tokRev,
        max_new_tokens=MAX_NEW_TOKENS,
        attention_mask_2d=attention_mask_2dRev,
        return_dict_in_generate=True,
        output_scores=True,
    )
    textEditedRev = tokenizer.decode(
        generateEditedRev.sequences[0], skip_special_tokens=True
    )

    # Reconfigure for positional edits with padding
    position_ids, tokB, tokC, tokAll, tokRev = get_position_ids_padded(
        tokA, tokB, tokC, tokD
    )

    # positional edits - ABCD
    generateEditedPos = model.generate(
        **tokAll,
        max_new_tokens=MAX_NEW_TOKENS,
        attention_mask_2d=causal_mask,
        position_ids=position_ids,
        return_dict_in_generate=True,
        output_scores=True,
    )
    textEditedPos = tokenizer.decode(
        generateEditedPos.sequences[0], skip_special_tokens=True
    )

    # positional edits - ACBD
    generateEditedPosRev = model.generate(
        **tokRev,
        max_new_tokens=MAX_NEW_TOKENS,
        attention_mask_2d=causal_mask,
        position_ids=position_ids,
        return_dict_in_generate=True,
        output_scores=True,
    )
    textEditedPosRev = tokenizer.decode(
        generateEditedPosRev.sequences[0], skip_special_tokens=True
    )

    # Edited causal mask + positional edits - ABCD
    attention_mask_2d = get_attention_mask_2d(tokA, tokB, tokC, tokD, tokAll)
    generateEditedAttentionPos = model.generate(
        **tokAll,
        max_new_tokens=MAX_NEW_TOKENS,
        attention_mask_2d=attention_mask_2d,
        position_ids=position_ids,
        return_dict_in_generate=True,
        output_scores=True,
    )
    textEditedAttentionPos = tokenizer.decode(
        generateEditedAttentionPos.sequences[0], skip_special_tokens=True
    )

    # Edited causal mask + positional edits - ACBD
    attention_mask_2dRev = get_attention_mask_2d(tokA, tokC, tokB, tokD, tokRev)
    generateEditedAttentionPosRev = model.generate(
        **tokRev,
        max_new_tokens=MAX_NEW_TOKENS,
        attention_mask_2d=attention_mask_2dRev,
        position_ids=position_ids,
        return_dict_in_generate=True,
        output_scores=True,
    )
    textEditedAttentionPosRev = tokenizer.decode(
        generateEditedAttentionPosRev.sequences[0], skip_special_tokens=True
    )

    # truncate so generated text doesn't include input prompt
    s = len(seqA) + len(seqB) + len(seqC) + len(seqD)
    textStandard = textStandard[s:]
    textStandardRev = textStandardRev[s:]
    textEdited = textEdited[s:]
    textEditedRev = textEditedRev[s:]
    textEditedPos = textEditedPos[s:]
    textEditedPosRev = textEditedPosRev[s:]
    textEditedAttentionPos = textEditedAttentionPos[s:]
    textEditedAttentionPosRev = textEditedAttentionPosRev[s:]

    # Compute KL divergence for input sequences
    logit_diffs = {}
    logit_diffs["Standard"] = scores_diff(generateStandard, generateStandardRev)
    logit_diffs["Attention"] = scores_diff(generateEdited, generateEditedRev)
    logit_diffs["Position"] = scores_diff(generateEditedPos, generateEditedPosRev)
    logit_diffs["Attention Position"] = scores_diff(
        generateEditedAttentionPos, generateEditedAttentionPosRev
    )

    return (
        textStandard,
        textStandardRev,
        textEdited,
        textEditedRev,
        textEditedPos,
        textEditedPosRev,
        textEditedAttentionPos,
        textEditedAttentionPosRev,
        logit_diffs,
    )


def getTokCombined(toks):
    return {
        "input_ids": torch.cat([tok["input_ids"][0] for tok in toks]).unsqueeze(0),
        "attention_mask": torch.cat(
            [tok["attention_mask"][0] for tok in toks]
        ).unsqueeze(0),
    }


# like testAttentionPosPaddingEdits, but refactored to work with llama and gpt2 models that accept 2D attention_mask parameters
def testInterventionsRefactored(
    seqA, seqB, seqC, seqD, model=modelGPT, tokenizer=tokenizerGPT
):
    tokA = tokenizer(
        seqA, return_tensors="pt", add_special_tokens=True, return_token_type_ids=False
    ).to(torch_device)
    tokD = tokenizer(
        seqD, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False
    ).to(torch_device)
    seqMCQ = [f"A) {seqB},", f"B) {seqC}."]
    # seqMCQRev=[f"A) {seqC},",f"B) {seqB}."]
    seqMCQRev = seqMCQ[::-1]
    tokMCQOptions = tuple(
        [
            tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(
                torch_device
            )
            for input_text in seqMCQ
        ]
    )
    tokMCQOptionsRev = tuple(
        [
            tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(
                torch_device
            )
            for input_text in seqMCQRev
        ]
    )
    tokAll = tokenizer(
        seqA + "".join(seqMCQ) + seqD, return_tensors="pt", return_token_type_ids=False
    ).to(torch_device)
    assert len(tokA["attention_mask"][0]) + sum(
        [len(tokOption["attention_mask"][0]) for tokOption in tokMCQOptions]
    ) + len(tokD["attention_mask"][0]) == len(tokAll["attention_mask"][0])
    tokRev = tokenizer(seqA + "".join(seqMCQRev) + seqD, return_tensors="pt").to(
        torch_device
    )
    assert len(tokRev["attention_mask"][0]) == len(tokAll["attention_mask"][0])
    s = len(tokAll["input_ids"][0])
    inputTextLen = len(seqA) + len(seqB) + len(seqC) + len(seqD)

    # Run tests with no position_id intervention
    generated, text = [], []
    causal_mask = torch.tril(torch.ones((s, s), dtype=torch.bool)).view(1, 1, s, s)
    attention_mask_2d = get_attention_mask_2d(
        tokA, tokMCQOptions[0], tokMCQOptions[1], tokD
    )
    attention_mask_2dRev = get_attention_mask_2d(
        tokA, tokMCQOptions[1], tokMCQOptions[0], tokD
    )
    position_ids = torch.arange(s).unsqueeze(0)
    # TODO: get below for-loop working with refactored gpt2 model
    for attention_mask, tok in zip(
        [causal_mask, causal_mask, attention_mask_2d, attention_mask_2dRev],
        [tokAll, tokRev, tokAll, tokRev],
    ):
        generated.append(
            model.generate(
                tok["input_ids"],
                max_new_tokens=MAX_NEW_TOKENS,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict_in_generate=True,
                output_scores=True,
            )
        )
        text.append(
            tokenizer.decode(generated[-1].sequences[0], skip_special_tokens=True)[
                inputTextLen:
            ]
        )

    # Reconfigure for positional edits with padding
    position_ids, tokB, tokC, tokAll, tokRev = get_position_ids_padded(
        tokA, tokMCQOptions[0], tokMCQOptions[1], tokD, tokenizer
    )
    _, _, _, tokRev, _ = get_position_ids_padded(
        tokA, tokMCQOptionsRev[0], tokMCQOptionsRev[1], tokD, tokenizer
    )
    assert len(tokA["attention_mask"][0]) + len(tokB["attention_mask"][0]) + len(
        tokC["attention_mask"][0]
    ) + len(tokD["attention_mask"][0]) == len(tokAll["attention_mask"][0])
    assert len(tokRev["attention_mask"][0]) == len(tokAll["attention_mask"][0])
    attention_mask_2d = get_attention_mask_2d(tokA, tokB, tokC, tokD, tokAll)
    attention_mask_2dRev = get_attention_mask_2d(tokA, tokC, tokB, tokD, tokRev)

    # Run tests with position_id intervention
    for attention_mask, tok in zip(
        [
            tokAll["attention_mask"],
            tokRev["attention_mask"],
            attention_mask_2d,
            attention_mask_2dRev,
        ],
        [tokAll, tokRev, tokAll, tokRev],
    ):
        generated.append(
            model.generate(
                tok["input_ids"],
                max_new_tokens=MAX_NEW_TOKENS,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict_in_generate=True,
                output_scores=True,
            )
        )
        text.append(
            tokenizer.decode(generated[-1].sequences[0], skip_special_tokens=True)[
                inputTextLen:
            ]
        )

    # Compute KL divergence for input sequences
    # With old GPT2 modified model
    # {'Standard': tensor(3.2158e-05), 'Attention': tensor(2.9942e-08), 'Position': tensor(3.2408e-05), 'Attention Position': tensor(3.4344e-13)}
    # {'Standard': tensor(2.0837e-07), 'Attention': tensor(2.9786e-07), 'Position': tensor(3.2408e-05), 'Attention Position': tensor(3.4344e-13)}
    logit_diffs = {}
    logit_diffs["Standard"] = scores_diff(generated[0], generated[1], add_epsilon=True)
    logit_diffs["Attention"] = scores_diff(generated[2], generated[3], add_epsilon=True)
    logit_diffs["Position"] = scores_diff(generated[4], generated[5], add_epsilon=True)
    logit_diffs["Attention Position"] = scores_diff(
        generated[6], generated[7], add_epsilon=True
    )

    # TODO: why is the KL divergence not smaller here for Attention+position edits???

    return tuple(text + [logit_diffs])


def testAttentionPosPaddingEditsFiveOptions(
    seqA, mcq_options, seqD, n_sequences=None, n_output_tokens=10
):
    # Use padding when modifying the positional encoding
    tokA, tokD = tuple(
        [
            tokenizer(input_text, return_tensors="pt").to(torch_device)
            for input_text in [seqA, seqD]
        ]
    )

    textStandardSet = set()
    textEditedSet = set()
    orderings = list(itertools.permutations(list(range(len(mcq_options)))))
    if n_sequences is not None:
        orderings = orderings[:n_sequences]
    for ordering in orderings:
        seqMCQ = []
        for i, prefix in zip(ordering, ["A", "B", "C", "D", "E"][: len(mcq_options)]):
            seqMCQ.append(
                f"{prefix}) {mcq_options[i]}" + ("," if prefix != "E" else ".")
            )
        tokMCQOptions = tuple(
            [
                tokenizer(input_text, return_tensors="pt").to(torch_device)
                for input_text in seqMCQ
            ]
        )
        print(seqA + "".join(seqMCQ) + seqD)
        tokAll = tokenizer(seqA + "".join(seqMCQ) + seqD, return_tensors="pt").to(
            torch_device
        )
        assert len(tokA["attention_mask"][0]) + sum(
            [len(tokOption["attention_mask"][0]) for tokOption in tokMCQOptions]
        ) + len(tokD["attention_mask"][0]) == len(tokAll["attention_mask"][0])

        # Reconfigure for positional edits with padding
        position_ids, tokMCQOptions, tokAll = get_position_ids_padded_n_options(
            tokA, tokMCQOptions, tokD, ordering=ordering
        )
        s = len(tokAll["attention_mask"][0])

        # Standard causal mask - ABCD
        generateStandard = model.generate(
            **tokAll,
            max_new_tokens=n_output_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
        textStandard = tokenizer.decode(
            generateStandard.sequences[0][s:], skip_special_tokens=True
        )
        textStandardSet.add(textStandard)

        # Edited mask + position ids
        attention_mask_2d = get_attention_mask_2d_n_options(
            tokA, tokMCQOptions, tokD, tokAll
        )
        generateEditedAttentionPos = model.generate(
            **tokAll,
            max_new_tokens=n_output_tokens,
            attention_mask_2d=attention_mask_2d,
            position_ids=position_ids,
            return_dict_in_generate=True,
            output_scores=True,
        )
        textEditedAttentionPos = tokenizer.decode(
            generateEditedAttentionPos.sequences[0][s:], skip_special_tokens=True
        )

        textEditedSet.add(textEditedAttentionPos)

    return textStandardSet, textEditedSet


seqA = 'Answer the following question after the prompt "Answer". Question: Consider the following pets.'
seqB = "The dog is fluffy and brown."
seqC = "The cat is black and white and scraggly."
seqD = "Which animal is cuter? Answer:"

# testAttentionMaskEdits(seqA,seqB,seqC,seqD)


def get_mcq_prompt(stem, optionA, optionB, optionC, optionD):
    seqA = f"Answer the following multiple choice question, by first stating which of the following options fits best, and then explaining your reasoning. Here is the question: {stem}. The options are:"
    # seqB = f" A. {optionA} B. {optionB}"
    # seqC = f" C. {optionC} D. {optionD}"
    seqB = f" {optionA}, {optionB},"
    seqC = f" {optionC}, {optionD},"
    seqD = " The answer is: "
    return seqA, seqB, seqC, seqD


def get_mcq_prompt_five_options(mcq_row):
    options = list([mcq_row["question"]["choices"][i]["text"] for i in range(5)])
    stem = mcq_row["question"]["stem"]
    # seqA = f"Consider the following single multiple choice question. First state which option is the correct answer: the first word after the prompt \'Answer:\' should be one of A,B,C,D, or E. Then explain your reasoning. Question: {stem}. Options:"
    seqA = f"Question: What animal makes a good low maintenance pet? Options: A) tiger, B) lion, C) giraffe D) hamster, E) dragon. Anwer: D) hamster. Question: {stem}. Options:"
    seqD = f" Answer: "
    return seqA, options, seqD


def get_model_predictions(mcq_row):
    seqA, mcq_options, seqD = get_mcq_prompt_five_options(mcq_row)
    textStandardSet, textEditedSet = testAttentionPosPaddingEditsFiveOptions(
        seqA, mcq_options, seqD, n_sequences=1, n_output_tokens=50
    )
    standardResponse, editedResponse = list(textStandardSet)[0], list(textEditedSet)[0]
    correctAnswer = mcq_row["answerKey"]
    options = {
        mcq_row["question"]["choices"][i]["label"]: mcq_row["question"]["choices"][i][
            "text"
        ]
        for i in range(5)
    }
    standardPred, standardPredIdx, editedPred, editedPredIdx = None, None, None, None
    for option in ["A", "B", "C", "D", "E"]:
        if option in standardResponse and (
            standardPredIdx is None or standardResponse.index(option) < standardPredIdx
        ):
            standardPred = option
            standardPredIdx = standardResponse.index(option)
        if option in editedResponse and (
            editedPredIdx is None or editedResponse.index(option) < editedPredIdx
        ):
            editedPred = option
            editedPredIdx = editedResponse.index(option)
    print(f"Standard text: {standardResponse}\n\nEdited text: {editedResponse}")
    print(
        f"Correct: {correctAnswer}\n\n Standard: {standardPred}, Edited: {editedPred}"
    )


def get_mcq_prompt_two_options(mcq_row):
    correct_answer = mcq_row["answerKey"]
    correct_option = [
        mcq_row["question"]["choices"][i]["text"]
        for i in range(5)
        if mcq_row["question"]["choices"][i]["label"] == correct_answer
    ][0]
    incorrect_option = [
        mcq_row["question"]["choices"][i]["text"]
        for i in range(5)
        if mcq_row["question"]["choices"][i]["label"] != correct_answer
    ][0]
    options = [correct_option, incorrect_option]
    stem = mcq_row["question"]["stem"]
    seqA = f"{stem} Options:"
    seqD = f"Answer: "
    # seqA = f"Consider the following single multiple choice question. First state which option is the correct answer: the first word after the prompt \'Answer:\' should be one of A or B. Then explain your reasoning. Question: {stem}. Options:"
    # seqD = f" Is the answer A, or B? "
    return seqA, options, seqD


def get_model_predictions_two_options(mcq_row):
    seqA, mcq_options, seqD = get_mcq_prompt_two_options(mcq_row)
    textStandardSet, textEditedSet = testAttentionPosPaddingEditsFiveOptions(
        seqA, mcq_options, seqD, n_sequences=1, n_output_tokens=50
    )
    standardResponse, editedResponse = list(textStandardSet)[0], list(textEditedSet)[0]
    correctAnswer = "A"  # hardcode correct answer as answer A#mcq_row['answerKey']
    options = {
        mcq_row["question"]["choices"][i]["label"]: mcq_row["question"]["choices"][i][
            "text"
        ]
        for i in range(5)
    }
    standardPred, standardPredIdx, editedPred, editedPredIdx = None, None, None, None
    for option in ["A", "B"]:
        if option in standardResponse and (
            standardPredIdx is None or standardResponse.index(option) < standardPredIdx
        ):
            standardPred = option
            standardPredIdx = standardResponse.index(option)
        if option in editedResponse and (
            editedPredIdx is None or editedResponse.index(option) < editedPredIdx
        ):
            editedPred = option
            editedPredIdx = editedResponse.index(option)
    print(f"Standard text: {standardResponse}\n\nEdited text: {editedResponse}")
    print(
        f"Correct: {correctAnswer}\n\n Standard: {standardPred}, Edited: {editedPred}"
    )


# seqA,mcq_options,seqD=get_mcq_prompt_five_options(mcq_rows[10])
# textStandardSet,textEditedSet=testAttentionPosPaddingEditsFiveOptions(seqA,mcq_options,seqD)
# get_model_predictions(mcq_rows[13])

seqA, seqB, seqC, seqD = get_mcq_prompt(
    "Sammy wanted to go to where the people were. Where might he go?",
    "race track",
    "populated areas",
    "the desert",
    "apartment",
)

# testAttentionMaskEdits(seqA,seqB,seqC,seqD)

seqA, seqB, seqC, seqD = get_mcq_prompt(
    "To locate a choker not located in a jewelry box or boutique where would you go?",
    "jewelry box",
    "jewelry store",
    "boutique",
    "store",
)

# testAttentionMaskEdits(seqA,seqB,seqC,seqD)
