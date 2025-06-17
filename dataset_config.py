#!/usr/bin/env python3
"""
Dataset configuration for multilingual tokenizer training.
Includes all 80 largest languages with bias weights based on data size.
"""

# Constants - modify these to adjust training
TOTAL_SAMPLES = 5000  # Reduced for faster training (was 500000)
STREAMING_ENABLED = True

# Language configurations with bias weights based on data size
# Bias assignment: English=10, Top 5=6-8, Next 10=4-5, Next 20=2-3, Rest=1-2
LANGUAGE_CONFIGS = [
    # English (special case - different dataset)
    ("sample-10BT", 10),  # English gets highest weight
    # Top tier languages (massive data)
    ("rus_Cyrl", 8),  # Russian
    ("cmn_Hani", 7),  # Mandarin Chinese
    ("deu_Latn", 6),  # German
    ("jpn_Jpan", 6),  # Japanese
    ("spa_Latn", 5),  # Spanish
    # High tier languages (large data)
    # ("fra_Latn", 5),  # French
    # ("ita_Latn", 4),  # Italian
    # ("por_Latn", 4),  # Portuguese
    # ("pol_Latn", 4),  # Polish
    # ("nld_Latn", 4),  # Dutch
    # ("ind_Latn", 4),  # Indonesian
    # ("tur_Latn", 4),  # Turkish
    # ("ces_Latn", 3),  # Czech
    # ("kor_Hang", 3),  # Korean
    # ("arb_Arab", 3),  # Standard Arabic
    # # Mid tier languages (substantial data)
    # ("hun_Latn", 3),  # Hungarian
    # ("fas_Arab", 3),  # Persian
    # ("ron_Latn", 3),  # Romanian
    # ("vie_Latn", 3),  # Vietnamese
    # ("ukr_Cyrl", 3),  # Ukrainian
    # ("nob_Latn", 3),  # Norwegian Bokmål
    # ("tha_Thai", 3),  # Thai
    # ("ell_Grek", 3),  # Modern Greek
    # ("swe_Latn", 2),  # Swedish
    # ("dan_Latn", 2),  # Danish
    # ("fin_Latn", 2),  # Finnish
    # ("bul_Cyrl", 2),  # Bulgarian
    # ("slk_Latn", 2),  # Slovak
    # ("hrv_Latn", 2),  # Croatian
    # ("hin_Deva", 2),  # Hindi
    # ("lit_Latn", 2),  # Lithuanian
    # ("bos_Latn", 2),  # Bosnian
    # ("heb_Hebr", 2),  # Hebrew
    # ("ben_Beng", 2),  # Bengali
    # ("slv_Latn", 2),  # Slovenian
    # # Lower tier languages (moderate data)
    # ("ekk_Latn", 2),  # Standard Estonian
    # ("cat_Latn", 2),  # Catalan
    # ("lvs_Latn", 2),  # Standard Latvian
    # ("zsm_Latn", 2),  # Standard Malay
    # ("azj_Latn", 2),  # North Azerbaijani
    # ("tam_Taml", 2),  # Tamil
    # ("srp_Cyrl", 2),  # Serbian (Cyrillic)
    # ("als_Latn", 2),  # Tosk Albanian
    # ("kat_Geor", 2),  # Georgian
    # ("kaz_Cyrl", 2),  # Kazakh
    # ("urd_Arab", 2),  # Urdu
    # ("ary_Arab", 2),  # Moroccan Arabic
    # ("mar_Deva", 2),  # Marathi
    # ("npi_Deva", 2),  # Nepali
    # ("mal_Mlym", 2),  # Malayalam
    # ("tel_Telu", 2),  # Telugu
    # ("mkd_Cyrl", 2),  # Macedonian
    # ("isl_Latn", 1),  # Icelandic
    # ("bel_Cyrl", 1),  # Belarusian
    # ("afr_Latn", 1),  # Afrikaans
    # # Smallest tier languages (limited data)
    # ("kan_Knda", 1),  # Kannada
    # ("fil_Latn", 1),  # Filipino
    # ("mya_Mymr", 1),  # Burmese
    # ("glg_Latn", 1),  # Galician
    # ("guj_Gujr", 1),  # Gujarati
    # ("anp_Deva", 1),  # Angika
    # ("khk_Cyrl", 1),  # Halh Mongolian
    # ("gmh_Latn", 1),  # Middle High German
    # ("khm_Khmr", 1),  # Khmer
    # ("eus_Latn", 1),  # Basque
    # ("ars_Arab", 1),  # Najdi Arabic
    # ("sin_Sinh", 1),  # Sinhala
    # ("hye_Armn", 1),  # Armenian
    # ("uzn_Latn", 1),  # Northern Uzbek (Latin)
    # ("uzn_Cyrl", 1),  # Northern Uzbek (Cyrillic)
    # ("lat_Latn", 1),  # Latin
    # ("arz_Arab", 1),  # Egyptian Arabic
    # ("pan_Guru", 1),  # Panjabi
    # ("kir_Cyrl", 1),  # Kirghiz
    # ("swh_Latn", 1),  # Swahili
    # ("srp_Latn", 1),  # Serbian (Latin)
    # ("bew_Latn", 1),  # Betawi
    # ("nno_Latn", 1),  # Norwegian Nynorsk
    # ("ory_Orya", 1),  # Odia
    # ("tgk_Cyrl", 1),  # Tajik
]

# Build dataset configuration
DATASETS_CONFIG = []

# Add English (special case with different dataset path)
DATASETS_CONFIG.append(
    {
        "path": "HuggingFaceFW/fineweb",
        "name": "sample-10BT",
        "split": "train",
        "bias": 10,
    }
)

# Add all other languages (same dataset path)
DATASETS_CONFIG.extend(
    [
        {
            "path": "HuggingFaceFW/fineweb-2",
            "name": name,
            "split": "train",
            "bias": bias,
        }
        for name, bias in LANGUAGE_CONFIGS[1:]  # Skip English (first item)
    ]
)

# Calculate percentages from bias weights
total_bias = sum(config["bias"] for config in DATASETS_CONFIG)
print(
    f"Dataset bias configuration ({len(DATASETS_CONFIG)} languages, total bias weight: {total_bias}):"
)

for config in DATASETS_CONFIG:
    config["percent"] = config["bias"] / total_bias
    config["samples"] = int(config["percent"] * TOTAL_SAMPLES)
    print(
        f"  {config['name']}: bias={config['bias']} -> {config['percent']:.1%} ({config['samples']:,} samples)"
    )

print("\nLanguage distribution:")
print(f"  English: {DATASETS_CONFIG[0]['percent']:.1%}")
print(f"  Top 5 non-English: {sum(c['percent'] for c in DATASETS_CONFIG[1:6]):.1%}")
print(f"  All others: {sum(c['percent'] for c in DATASETS_CONFIG[6:]):.1%}")
