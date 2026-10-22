import torch

from .common import digest


def tokenize_example(tokenizer, prompt, response):
    """Find the response character span in the complete rendered template.

    Prefix equality is checked on rendered strings, not separately tokenized IDs.
    Offsets from the single full-sequence tokenization determine the loss mask.
    """
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": response}], tokenize=False)
    if not full.startswith(prefix + response):
        raise ValueError("Template does not preserve the expected assistant content span")
    start, end = len(prefix), len(prefix) + len(response)
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    mask = []
    for left, right in encoded["offset_mapping"]:
        overlaps = left < end and right > start
        if overlaps and not (left >= start and right <= end):
            raise ValueError("A token crosses an assistant content boundary")
        mask.append(int(overlaps and right > left))
    if not sum(mask) or mask[0]:
        raise ValueError("Empty or unscorable response")
    ids = encoded["input_ids"]
    if any(t in tokenizer.all_special_ids for t, m in zip(ids, mask) if m):
        raise ValueError("Response includes special/control token IDs")
    first = mask.index(1)
    return {
        "input_ids": ids, "response_mask": mask,
        "prompt_tokens": first, "response_tokens": sum(mask),
        "sequence_tokens": len(ids),
        "tokenization_hash": digest({"input_ids": ids, "response_mask": mask}),
    }


def collate(rows, pad_token_id, device="cuda"):
    n = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), n), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        length = len(row["input_ids"])
        ids[i, :length] = torch.tensor(row["input_ids"], device=device)
        attention[i, :length] = 1
        mask[i, :length] = torch.tensor(row["response_mask"], dtype=torch.bool, device=device)
    return {"input_ids": ids, "attention_mask": attention}, mask
