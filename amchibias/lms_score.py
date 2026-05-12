import os
import csv
import sys
from dotenv import load_dotenv
from collections import defaultdict
from openai import OpenAI
from minicons import scorer
from transformers import AutoTokenizer, AutoModelForMaskedLM

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
os.environ["HF_HOME"] = os.getenv("HF_HOME")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HUGGING_FACE_HUB_TOKEN")

MODELS = {
    'mbert':        'bert-base-multilingual-cased',
    'xlmr':         'xlm-roberta-base',
    'muril':        'google/muril-base-cased',
    'indicbert_v2': 'ai4bharat/indic-bert',
    'indicbert_v3': 'ai4bharat/IndicBERTv2-MLM-only',
}

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "goan-stereotypes-main/data/raw/final_data - cleaned_stereotypes.csv"

with open(CSV_PATH, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

results = defaultdict(lambda: {"biased": 0, "lms": 0, "total": 0})

# Load each model once
for model_name, model_path in MODELS.items():
    print(f"\nLoading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True
        )
        model = AutoModelForMaskedLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        model.eval()
        mlm = scorer.MaskedLMScorer(model, 'cpu', tokenizer=tokenizer)
    except Exception as e:
        print(f"  Failed to load {model_name}: {e}")
        continue

    for row in rows:
        for lang, stereo_col, anti_col, control_col in [
            ("english", "Stereotype Sentence",        "Anti-Stereotype Sentence",        "Control Sentence English"),
        ]:
            sents = {
                "stereo":  row[stereo_col],
                "anti":    row[anti_col],
                "control": row[control_col],
            }

            print(f"\n{'='*55}")
            print(f"ID: {row['ID']} | MODEL: {model_name} | LANG: {lang}")
            print(f"{'='*55}")
            try:
                scores = mlm.sequence_score(
                    [sents["stereo"], sents["anti"], sents["control"]],
                    reduction=lambda x: x.mean(0).item(),
                    PLL_metric='within_word_l2r'
                )
                s, a, c = scores
                biased = s > a
                lms    = (s > c) and (a > c)

                print(f"  Stereo:  {s:.4f}")
                print(f"  Anti:    {a:.4f}")
                print(f"  Control: {c:.4f}")
                print(f"  Biased (stereo > anti):    {biased}")
                print(f"  LMS OK (stereo > control): {s > c}")
                print(f"  LMS OK (anti   > control): {a > c}")

                results[(model_name, lang)]["biased"] += int(biased)
                results[(model_name, lang)]["lms"]    += int(lms)
                results[(model_name, lang)]["total"]  += 1

            except Exception as e:
                print(f"  Error: {e}")

# Summary table
print(f"\n\n{'='*55}")
print(f"{'Model':<15} {'Lang':<10} {'Bias Score':>12} {'LMS':>8}")
print(f"{'='*55}")
for (model_name, lang), v in results.items():
    n = v["total"]
    if n == 0:
        continue
    print(f"{model_name:<15} {lang:<10} {v['biased']/n*100:>11.1f}% {v['lms']/n*100:>7.1f}%")