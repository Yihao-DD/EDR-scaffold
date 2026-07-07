"""Model/tokenizer loading and generation — the only module that touches HF.

Three loading paths, all ported verbatim from the frozen round-1 code:
- load_model_for_training: QLoRA (or bf16) base for LoRA training.
- load_model_for_eval:     base [+ merged frozen adapter] [+ eval adapter].
- ModelRunner:             generation-only runner; optionally merges a frozen
                           adapter (round-2 loop runs on merged M1).
"""

from __future__ import annotations

import random

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def set_seed(seed):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_id):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model_for_training(model_id, qlora=True):
    import torch
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if qlora:
        model = prepare_model_for_kbit_training(model)
    return model


def apply_lora(model, rank, alpha=None, dropout=0.05):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha or rank * 2,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def merge_base_adapter(model, adapter_dir):
    """Merge a frozen adapter into model weights before attaching a new LoRA.

    Round 2 trains A2 on top of M1 = M0 + A1. After this merge, the trainable
    adapter added by ``apply_lora`` is A2, and ``model.disable_adapter()`` in
    the KL-anchor path evaluates the frozen M1 reference rather than raw M0.
    """

    from peft import PeftModel

    if not adapter_dir:
        return model
    print(f"[model] merging frozen base adapter into weights: {adapter_dir}", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model = model.merge_and_unload()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def load_model_for_eval(model_id, adapter_dir=None, qlora=True, base_adapter_dir=None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs = {"device_map": "auto", "trust_remote_code": True}
    if base_adapter_dir and qlora:
        print(
            "[eval:warn] base_adapter_dir requires merging before loading the evaluated adapter; "
            "loading bf16/full precision for the merge. Use a >=48GB GPU.",
            flush=True,
        )
        qlora = False
    if qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model = merge_base_adapter(model, base_adapter_dir)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model


def render_user_prompt(tokenizer, input_text):
    messages = [{"role": "user", "content": input_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return input_text


def render_training_text(tokenizer, input_text, output_text):
    messages = [{"role": "user", "content": input_text}, {"role": "assistant", "content": output_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"{input_text}\n{output_text}"


def generate_one(model, tokenizer, prompt, max_new_tokens):
    import torch

    text = render_user_prompt(tokenizer, prompt)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[:, inputs.input_ids.shape[-1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]


class GenerationRunner:
    """Batch sampling runner (round-1 probe runner, ported verbatim)."""

    def __init__(self, model_id, cache_dir=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.device = self.model.device

    def generate_many(self, prompt, n=1, max_new_tokens=256, do_sample=False, temperature=0.0, seed=None):
        if seed is not None:
            self.torch.manual_seed(seed)
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed_all(seed)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": n,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = temperature
        with self.torch.no_grad():
            generated = self.model.generate(**inputs, **kwargs)
        new_tokens = generated[:, inputs.input_ids.shape[-1] :]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


class ModelRunner:
    """Greedy/temperature generation runner with optional frozen-adapter merge.

    The round-2 evolve loop MUST run on merged M1 (raw-M0 patch search is
    invalid); that invariant is enforced by the loop entrypoint, which passes
    require_adapter=True.
    """

    def __init__(self, model_id, cache_dir=None, base_adapter_dir=None, require_adapter=False):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.base_adapter_dir = str(base_adapter_dir) if base_adapter_dir else None
        if require_adapter and not self.base_adapter_dir:
            raise SystemExit("this entrypoint requires a merged adapter; raw base-model execution is invalid here.")
        if self.base_adapter_dir:
            print(f"[model] merging adapter into memory: {self.base_adapter_dir}", flush=True)
            self.model = PeftModel.from_pretrained(self.model, self.base_adapter_dir, is_trainable=False)
            self.model = self.model.merge_and_unload()
        self.model.eval()

    def generate(self, prompt, max_new_tokens=256, do_sample=False, temperature=None):
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if hasattr(self.tokenizer, "apply_chat_template") else prompt
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample and temperature is not None:
            generation_kwargs["temperature"] = temperature
        with torch.no_grad():
            generated = self.model.generate(**inputs, **generation_kwargs)
        new_tokens = generated[:, inputs.input_ids.shape[-1] :]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))
