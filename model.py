"""
Direct Preference Optimization (DPO) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - log_softmax
import numpy as np

def log_softmax(logits, axis=-1):
    logits = np.asarray(logits, dtype=np.float64)

    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    log_sum_exp = np.log(
        np.sum(np.exp(shifted), axis=axis, keepdims=True)
    )

    return shifted - log_sum_exp

# Step 2 - softmax
import numpy as np

def softmax(logits, axis=-1):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)

# Step 3 - gather_token_logprobs
def gather_token_logprobs(log_probs, token_ids):
    log_probs = np.asarray(log_probs)
    token_ids = np.asarray(token_ids, dtype=np.int64)

    return np.take_along_axis(
        log_probs,
        token_ids[..., None],
        axis=-1
    ).squeeze(-1)

# Step 4 - masked_sequence_logprob
def masked_sequence_logprob(token_logprobs, mask):
    token_logprobs = np.asarray(token_logprobs)
    mask = np.asarray(mask, dtype=bool)

    return np.sum(
        np.where(mask, token_logprobs, 0.0),
        axis=-1
    )

# Step 5 - init_policy_params
def init_policy_params(vocab_size, d_model, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    return {
        "embed": rng.normal(0.0, 0.02, size=(vocab_size, d_model)),
        "W_out": rng.normal(0.0, 0.02, size=(d_model, vocab_size)),
        "b_out": np.zeros(vocab_size, dtype=np.float64),
    }

# Step 6 - policy_token_logits
def policy_token_logits(params, token_ids):
    token_ids = np.asarray(token_ids, dtype=np.int64)
    hidden = params["embed"][token_ids]
    return hidden @ params["W_out"] + params["b_out"]

# Step 7 - policy_sequence_logprob
import numpy as np

def policy_sequence_logprob(params, token_ids, mask):
    token_ids = np.asarray(token_ids, dtype=np.int64)

    logits = policy_token_logits(params, token_ids)
    log_probs = log_softmax(logits, axis=-1)

    token_logprobs = gather_token_logprobs(log_probs, token_ids)

    return masked_sequence_logprob(token_logprobs, mask)

# Step 8 - sequence_logprob_grad
import numpy as np

def sequence_logprob_grad(params, token_ids, mask):
    token_ids = np.asarray(token_ids, dtype=np.int64)
    mask = np.asarray(mask, dtype=np.float64)

    embed = params["embed"]
    W_out = params["W_out"]
    b_out = params["b_out"]

    # 前向计算
    hidden = embed[token_ids]                  # (B, T, D)
    logits = hidden @ W_out + b_out            # (B, T, V)

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    # d log P(token) / d logits = one_hot(token) - probs
    V = W_out.shape[1]
    grad_logits = -probs
    grad_logits.reshape(-1, V)[
        np.arange(token_ids.size), token_ids.ravel()
    ] += 1.0
    grad_logits *= mask[..., None]

    # 输出投影与偏置的梯度
    D = W_out.shape[0]
    grad_W_out = hidden.reshape(-1, D).T @ grad_logits.reshape(-1, V)
    grad_b_out = grad_logits.sum(axis=(0, 1))

    # embedding 的梯度：同一 token 多次出现时，需要累加
    grad_hidden = grad_logits @ W_out.T        # (B, T, D)
    grad_embed = np.zeros_like(embed)
    np.add.at(
        grad_embed,
        token_ids.ravel(),
        grad_hidden.reshape(-1, D),
    )

    return {
        "embed": grad_embed,
        "W_out": grad_W_out,
        "b_out": grad_b_out,
    }

# Step 9 - bradley_terry_loss
def bradley_terry_loss(reward_chosen, reward_rejected):
    reward_chosen = np.asarray(reward_chosen, dtype=np.float64)
    reward_rejected = np.asarray(reward_rejected, dtype=np.float64)

    margins = reward_chosen - reward_rejected
    return float(np.mean(np.logaddexp(0.0, -margins)))

# Step 10 - reward_accuracy
def reward_accuracy(reward_chosen, reward_rejected):
    reward_chosen = np.asarray(reward_chosen)
    reward_rejected = np.asarray(reward_rejected)

    return float(np.mean(reward_chosen > reward_rejected))

# Step 11 - build_preference_pairs
def build_preference_pairs(prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    return [
        {
            "prompt": prompts[i],
            "chosen_ids": chosen_ids[i],
            "rejected_ids": rejected_ids[i],
            "chosen_mask": chosen_mask[i],
            "rejected_mask": rejected_mask[i],
        }
        for i in range(len(prompts))
    ]

# Step 12 - sample_preference_batch
import numpy as np

def sample_preference_batch(pairs, batch_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    indices = rng.choice(
        len(pairs),
        size=batch_size,
        replace=batch_size > len(pairs),
    )
    sampled = [pairs[i] for i in indices]

    keys = [
        "chosen_ids",
        "rejected_ids",
        "chosen_mask",
        "rejected_mask",
    ]
    if "prompt" in pairs[0]:
        keys.append("prompt")

    return {
        key: np.stack([pair[key] for pair in sampled], axis=0)
        for key in keys
    }

# Step 13 - freeze_reference_logprobs
import numpy as np

def freeze_reference_logprobs(ref_params, pairs):
    frozen = []

    for pair in pairs:
        chosen = policy_sequence_logprob(
            ref_params,
            np.asarray(pair["chosen_ids"])[None, :],
            np.asarray(pair["chosen_mask"])[None, :],
        )
        rejected = policy_sequence_logprob(
            ref_params,
            np.asarray(pair["rejected_ids"])[None, :],
            np.asarray(pair["rejected_mask"])[None, :],
        )

        frozen.append({
            "chosen": float(chosen[0]),
            "rejected": float(rejected[0]),
        })

    return frozen

# Step 14 - policy_reference_logratio
import numpy as np

def policy_reference_logratio(policy_logprobs, reference_logprobs):
    policy_logprobs = np.asarray(policy_logprobs, dtype=np.float64)
    reference_logprobs = np.asarray(reference_logprobs, dtype=np.float64)

    return policy_logprobs - reference_logprobs

# Step 15 - dpo_pair_margin (not yet solved)
# TODO: implement

# Step 16 - dpo_loss (not yet solved)
# TODO: implement

# Step 17 - dpo_loss_grad (not yet solved)
# TODO: implement

# Step 18 - dpo_train_step (not yet solved)
# TODO: implement

# Step 19 - train_dpo (not yet solved)
# TODO: implement

# Step 20 - length_normalized_logprob (not yet solved)
# TODO: implement

# Step 21 - ipo_loss (not yet solved)
# TODO: implement

# Step 22 - implicit_reward (not yet solved)
# TODO: implement

# Step 23 - preference_accuracy (not yet solved)
# TODO: implement

# Step 24 - kl_to_reference (not yet solved)
# TODO: implement

# Step 25 - reward_margin_stats (not yet solved)
# TODO: implement

# Step 26 - evaluate_dpo (not yet solved)
# TODO: implement

# Step 27 - run_dpo_pipeline (not yet solved)
# TODO: implement

