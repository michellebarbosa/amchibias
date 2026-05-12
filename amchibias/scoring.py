import os
import csv
from minicons import scorer
load_dotenv()

os.environ["HF_HOME"] = os.getenv("HF_HOME", "")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HUGGING_FACE_HUB_TOKEN", "")



MODELS = {
    'mbert': 'bert-base-multilingual-cased',
    'xlmr': 'xlm-roberta-base',
    'muril': 'google/muril-base-cased',
    'indicbert_v2': 'ai4bharat/indic-bert',
    'indicbert_v3': 'ai4bharat/IndicBERTv2-MLM-only',
}

LANGUAGES = {
    'english': {'stereo_col': 4, 'antistereo_col': 6},
    'konkani': {'stereo_col': 5, 'antistereo_col': 7},
}

DATA_PATH = '/mount/arbeitsdaten66/projekte/placebo/cleaned_stereotypes.csv'
OUTPUT_DIR = '/mount/arbeitsdaten66/projekte/placebo/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data once
data_rows = []
with open(DATA_PATH, encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        data_rows.append(row)

print(f"Loaded {len(data_rows)} rows from {DATA_PATH}")

for model_name, model_path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Loading {model_name} ({model_path})...")
    print('='*60)

    try:
        model = scorer.MaskedLMScorer(model_path, 'cpu')
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        continue

    for lang, cols in LANGUAGES.items():
        stereo_col     = cols['stereo_col']
        antistereo_col = cols['antistereo_col']

        print(f"\n  Scoring {model_name} on {lang}...")
        results = []
        skipped = 0

        for row in data_rows:
            if len(row) <= antistereo_col:
                skipped += 1
                continue

            stereo_sent = row[stereo_col].strip()
            anti_sent   = row[antistereo_col].strip()

            if not stereo_sent or not anti_sent:
                skipped += 1
                continue

            axis  = row[0].strip()
            group = row[1].strip()
            attr  = row[2].strip()
            id_   = row[3].strip()

            try:
                score_stereo = model.sequence_score(
                    [stereo_sent],
                    reduction=lambda x: x.mean(0).item(),
                    PLL_metric='within_word_l2r'
                )[0]

                score_anti = model.sequence_score(
                    [anti_sent],
                    reduction=lambda x: x.mean(0).item(),
                    PLL_metric='within_word_l2r'
                )[0]

                is_biased = score_stereo > score_anti
                results.append([axis, group, attr, id_,
                                 score_stereo, score_anti, is_biased])

            except Exception as e:
                print(f"    Error on {id_}: {e}")
                skipped += 1
                continue

        # Save results — overwrite mode to avoid duplicates
        out_path = f"{OUTPUT_DIR}/{model_name}_{lang}_results.csv"
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['axis', 'group', 'attr', 'id',
                             'score_stereo', 'score_antistereo', 'is_biased'])
            writer.writerows(results)

        biased_count = sum(1 for r in results if r[6])
        total = len(results)
        pct = 100 * biased_count / total if total > 0 else 0
        print(f"\n  >> {model_name} | {lang}: "
              f"{total} rows scored, {skipped} skipped. "
              f"{biased_count}/{total} biased ({pct:.1f}%)")

print("\n\nFINAL SUMMARY")
print("="*60)
for model_name in MODELS:
    for lang in LANGUAGES:
        out_path = f"{OUTPUT_DIR}/{model_name}_{lang}_results.csv"
        try:
            with open(out_path, encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                result_rows = list(reader)
            biased = sum(1 for r in result_rows if r[6] == 'True')
            total = len(result_rows)
            pct = 100 * biased / total if total > 0 else 0
            print(f"{model_name:20s} | {lang:10s} | "
                  f"{biased}/{total} ({pct:.1f}%)")
        except FileNotFoundError:
            print(f"{model_name:20s} | {lang:10s} | not run yet")