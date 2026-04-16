"""
Analyze what percentage of slot tokens are -100 in real training data.
"""

# Key sources of -100 tokens:
# 1. Special tokens ([CLS], [SEP], [PAD]) → word_id = None → -100
# 2. Subword continuation tokens (word_id == prev_word_id) → -100
# 3. IGNORE bio_tags from supplements (Change 1) → -100

# Estimate for typical French utterance in MASSIVE:
# - Avg utterance: ~8 words
# - CamemBERT subword rate: ~1.3 tokens/word (French has more subwords than English)
# - Special tokens: 2 ([CLS], [SEP])

# Calculation:
# Total tokens = 8 * 1.3 + 2 = 12.4 ≈ 12
# Valid tokens (first subword of each word) = 8
# -100 tokens = 12 - 8 = 4
# -100 ratio = 4/12 = 33%

# For supplements with IGNORE tags (Change 1):
# - All bio_tags are "IGNORE" → all word tokens become -100
# - Only special tokens processed
# Total tokens = 8 * 1.3 + 2 = 12
# Valid tokens = 0 (all words are IGNORE)
# -100 ratio = 12/12 = 100%

print("=" * 60)
print("Slot Token -100 Distribution Analysis")
print("=" * 60)

print("\n1. MASSIVE examples (with real slot annotations):")
print("   - Avg words: 8")
print("   - Subword factor: 1.3 tokens/word")
print("   - Total tokens: 8*1.3 + 2 = 12")
print("   - Valid slot tokens (first subword only): 8")
print("   - -100 tokens: 4 (special + subword continuations)")
print("   - -100 ratio: 33%")

print("\n2. Supplement examples (IGNORE tags, Change 1):")
print("   - Avg words: 8")
print("   - Subword factor: 1.3")
print("   - Total tokens: 12")
print("   - Valid slot tokens: 0 (all IGNORE)")
print("   - -100 tokens: 12")
print("   - -100 ratio: 100%")

print("\n3. Combined (need upsample factor from config):")
import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)
upsample = config.get("supplement_upsample_factor", 3)

# Rough estimate: MASSIVE has ~50k examples, supplements ~5k
# With upsample=3, supplements = 5k * 3 = 15k
# Total = 50k + 15k = 65k
massive_examples = 50000
supplement_examples = 5000 * upsample
total_examples = massive_examples + supplement_examples

massive_ratio = 0.33
supplement_ratio = 1.0

weighted_ratio = (
    massive_examples * massive_ratio + 
    supplement_examples * supplement_ratio
) / total_examples

print(f"   - MASSIVE examples: ~{massive_examples:,}")
print(f"   - Supplement examples (upsampled {upsample}x): ~{supplement_examples:,}")
print(f"   - Total: ~{total_examples:,}")
print(f"   - Weighted -100 ratio: {weighted_ratio:.1%}")

print("\n" + "=" * 60)
print("EFFICIENCY IMPACT")
print("=" * 60)
print(f"\nWith {weighted_ratio:.0%} -100 tokens:")
print(f"- FocalLoss computes exp/power on 100% of tokens")
print(f"- Only {100-weighted_ratio*100:.0%} are used in final loss")
print(f"- Wasted computation: ~{weighted_ratio:.0%}")
