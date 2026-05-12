#!/usr/bin/env python3
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import torch

# pip install IndicTransTokenizer
# pip install git+https://github.com/AI4Bharat/IndicTrans2.git

MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer and processor...")
tokenizer = IndicTransTokenizer(direction="en-indic")
ip = IndicProcessor(inference=True)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
)
model = model.to(device)
model.eval()

# Input: list of sentences with target language tag
src_lang = "eng_Latn"
tgt_lang = "kok_Deva"  # Konkani in Devanagari

sentences = [
    "UN Chief says there is no military solution in Syria",
    "The president addressed the nation on the economic crisis",
]

print("Preprocessing...")
batch = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)

inputs = tokenizer(
    batch,
    src=True,
    truncation=True,
    padding="longest",
    return_tensors="pt",
    return_attention_mask=True,
).to(device)

print("Translating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        num_beams=5,
        num_return_sequences=1,
        max_length=256,
        min_length=1,
    )

# Decode
with tokenizer.as_target_tokenizer():
    decoded = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

# Postprocess
translations = ip.postprocess_batch(decoded, lang=tgt_lang)

for src, tgt in zip(sentences, translations):
    print(f"\nOriginal:    {src}")
    print(f"Translation: {tgt}")