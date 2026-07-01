from scripts.lora_phase0 import collate, encode_row


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = ""
        for message in messages:
            text += f"<{message['role']}>{message['content']}"
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(char) % 50 + 3 for char in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_encode_row_masks_prompt_tokens():
    tokenizer = DummyTokenizer()
    encoded = encode_row(tokenizer, {"input": "question", "output": "{\"name\":\"fn\"}"}, max_length=512)

    assert len(encoded["input_ids"]) == len(encoded["labels"])
    assert -100 in encoded["labels"]
    assert any(label != -100 for label in encoded["labels"])


def test_collate_pads_input_and_masks_labels():
    batch = collate(
        [
            {"input_ids": [1, 2], "labels": [-100, 2]},
            {"input_ids": [3], "labels": [3]},
        ],
        pad_token_id=0,
    )

    assert batch["input_ids"].shape == (2, 2)
    assert batch["input_ids"][1, 1].item() == 0
    assert batch["labels"][1, 1].item() == -100
