import os
import csv
from collections import defaultdict

RESULTS_DIR = '/Users/michellebarbosa/goan-stereotypes/data/raw'

# Categorize identity groups
HYPERLOCAL = {
    'Tarvottis', 'Seafarers (Tarvottis)',
    'Mundkars', 'Tenants (Mundkars)',
    'Rendus', 'Toddy Tappers (Rendus)',
    'Gaudas', 'Gaudas (Indigenous Community)',
    'Kharvis', 'Kharvis (Fishing Community)',
    'Gulfies', 'Gulfies (Gulf Returnees/Kuwaitkars)',
    'Poders', 'Poders (Traditional Bakers)',
    'Bhatkars', 'Landlords (Bhatkars)',
    'Nustekars', 'Fishermen (Nustekars)',
    'Bamon', 'Chardo', 'GSB', 'GSB (Goud Saraswat Brahmin)',
    'Bardezkars', 'North Goans (Bardezkars)',
    'Sashtikars', 'South Goans (Sashtikars)',
    'Bhaile', 'Migrants (Bhaile)',
    'Locals (Goans)',
    'Sashti (Xashti) Dialect Speakers',
    'Romi Konkani Speakers',
    'Portuguese Speakers (Lusophones)',
    'Politicians (Goan)',
    'Northern Goans (Bardezkars)',
    'Devanagari Konkani Speakers',
    'Panjimkar',
    'Sudhirs',
}

PAN_INDIAN = {
    'Catholics', 'Hindus', 'Muslims',
    'Men', 'Women',
    'Youth', 'Elderly',
    'Kshatriyas', 'Vaishyas', 'Shudras',
    'English Speakers',
    'Konkani Speakers',
    'Marathi Speakers',
    'Kannada Speakers',
    'Tourists',
    'Hospitality Workers',
    'Maratha',
    'Rural Goans',
}

MODELS = ['mbert', 'xlmr', 'muril', 'indicbert_v2', 'indicbert_v3']
LANGUAGES = ['english', 'konkani']

print("=== HYPERLOCAL vs PAN-INDIAN BIAS SCORES ===\n")

for lang in LANGUAGES:
    print(f"\n{'='*60}")
    print(f"LANGUAGE: {lang.upper()}")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Hyperlocal':>12} {'Pan-Indian':>12} {'Difference':>12}")
    print("-" * 55)

    for model in MODELS:
        path = f"{RESULTS_DIR}/{model}_{lang}_results.csv"
        if not os.path.exists(path):
            continue

        hyperlocal_biased = 0
        hyperlocal_total = 0
        pan_indian_biased = 0
        pan_indian_total = 0
        unknown = set()

        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                group = row['group'].strip()
                is_biased = row['is_biased'].strip() == 'True'

                if group in HYPERLOCAL:
                    hyperlocal_total += 1
                    hyperlocal_biased += int(is_biased)
                elif group in PAN_INDIAN:
                    pan_indian_total += 1
                    pan_indian_biased += int(is_biased)
                else:
                    unknown.add(group)

        if hyperlocal_total > 0 and pan_indian_total > 0:
            h_pct = 100 * hyperlocal_biased / hyperlocal_total
            p_pct = 100 * pan_indian_biased / pan_indian_total
            diff = h_pct - p_pct
            print(f"{model:<15} {h_pct:>11.1f}% {p_pct:>11.1f}% {diff:>+11.1f}%")

        if unknown:
            print(f"  Unclassified groups: {unknown}")