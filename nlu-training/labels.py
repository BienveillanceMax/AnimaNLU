"""
Labels and mappings for Anima NLU multi-head model.

Source of truth for:
- 13 speech acts, 15 domains, 18 slot types (37 BIO labels)
- MASSIVE intent → (speech_act, domain) mapping
- MASSIVE slot → our slot type mapping
"""

# ──────────────────────────────────────────────
# Speech Acts (13)
# ──────────────────────────────────────────────
SPEECH_ACTS = [
    "Command",
    "Request",
    "Question",
    "Statement",
    "ExpressEmotion",
    "Greeting",
    "Farewell",
    "Confirm",
    "Deny",
    "Correction",
    "PositiveFeedback",
    "NegativeFeedback",
    "Cancel",
]

# ──────────────────────────────────────────────
# Domains (15)
# ──────────────────────────────────────────────
DOMAINS = [
    "Home",
    "Media",
    "Communication",
    "Calendar",
    "Reminder",
    "Weather",
    "News",
    "Knowledge",
    "Cooking",
    "Shopping",
    "Transport",
    "DateTime",
    "Social",
    "System",
    "Meta",
]

# ──────────────────────────────────────────────
# Slot Types (18) → BIO Labels (37)
# ──────────────────────────────────────────────
SLOT_TYPES = [
    "device_type",
    "room",
    "setting_value",
    "setting_type",
    "time_value",
    "date_value",
    "duration",
    "frequency",
    "person_name",
    "emotion",
    "topic",
    "media_item",
    "genre_type",
    "artist_name",
    "reminder_content",
    "scene_name",
    "event_name",
    "location",
]

# BIO labels: O + B-type + I-type for each slot type
SLOT_LABELS = ["O"]
for st in SLOT_TYPES:
    SLOT_LABELS.append(f"B-{st}")
    SLOT_LABELS.append(f"I-{st}")

assert len(SPEECH_ACTS) == 13
assert len(DOMAINS) == 15
assert len(SLOT_TYPES) == 18
assert len(SLOT_LABELS) == 37  # 1 + 18*2


# ──────────────────────────────────────────────
# label2id / id2label helpers
# ──────────────────────────────────────────────
def _build_maps(labels):
    l2i = {l: i for i, l in enumerate(labels)}
    i2l = {i: l for i, l in enumerate(labels)}
    return l2i, i2l


SPEECH_ACT_L2I, SPEECH_ACT_I2L = _build_maps(SPEECH_ACTS)
DOMAIN_L2I, DOMAIN_I2L = _build_maps(DOMAINS)
SLOT_LABEL_L2I, SLOT_LABEL_I2L = _build_maps(SLOT_LABELS)


# ──────────────────────────────────────────────
# MASSIVE intent → (speech_act, domain)
# ──────────────────────────────────────────────
# MASSIVE has 60 intents (names from dataset features).
# Each maps to exactly one (speech_act, domain) pair.
MASSIVE_INTENT_MAP = {
    # ── alarm (scenario 16) ──
    "alarm_set":              ("Command",  "Reminder"),
    "alarm_remove":           ("Command",  "Reminder"),
    "alarm_query":            ("Question", "Reminder"),
    # ── audio (scenario 10) ──
    "audio_volume_up":        ("Command",  "Media"),
    "audio_volume_down":      ("Command",  "Media"),
    "audio_volume_mute":      ("Command",  "Media"),
    "audio_volume_other":     ("Command",  "Media"),
    # ── calendar (scenario 2) ──
    "calendar_set":           ("Command",  "Calendar"),
    "calendar_remove":        ("Command",  "Calendar"),
    "calendar_query":         ("Question", "Calendar"),
    # ── cooking (scenario 13) ──
    "cooking_recipe":         ("Question", "Cooking"),
    "cooking_query":          ("Question", "Cooking"),
    # ── datetime (scenario 5) ──
    "datetime_query":         ("Question", "DateTime"),
    "datetime_convert":       ("Question", "DateTime"),
    # ── email (scenario 7) ──
    "email_sendemail":        ("Command",  "Communication"),
    "email_addcontact":       ("Command",  "Communication"),
    "email_query":            ("Question", "Communication"),
    "email_querycontact":     ("Question", "Communication"),
    # ── general (scenario 9) — MASSIVE only has 3 general intents ──
    "general_greet":          ("Greeting",  "Social"),
    "general_joke":           ("Question",  "Social"),
    "general_quirky":         ("Question",  "Social"),
    # ── iot (scenario 8) ──
    "iot_hue_lighton":        ("Command",  "Home"),
    "iot_hue_lightoff":       ("Command",  "Home"),
    "iot_hue_lightchange":    ("Command",  "Home"),
    "iot_hue_lightdim":       ("Command",  "Home"),
    "iot_hue_lightup":        ("Command",  "Home"),
    "iot_wemo_on":            ("Command",  "Home"),
    "iot_wemo_off":           ("Command",  "Home"),
    "iot_cleaning":           ("Command",  "Home"),
    "iot_coffee":             ("Command",  "Home"),
    # ── lists (scenario 11) ──
    "lists_createoradd":      ("Command",  "Shopping"),
    "lists_remove":           ("Command",  "Shopping"),
    "lists_query":            ("Question", "Shopping"),
    # ── music (scenario 15) ──
    "music_likeness":         ("Statement", "Media"),
    "music_dislikeness":      ("Statement", "Media"),
    "music_query":            ("Question",  "Media"),
    "music_settings":         ("Command",   "Media"),
    # ── news (scenario 4) ──
    "news_query":             ("Question",  "News"),
    # ── play (scenario 3) ──
    "play_music":             ("Command",  "Media"),
    "play_radio":             ("Command",  "Media"),
    "play_podcasts":          ("Command",  "Media"),
    "play_audiobook":         ("Command",  "Media"),
    "play_game":              ("Command",  "Media"),
    # ── qa (scenario 12) ──
    "qa_factoid":             ("Question", "Knowledge"),
    "qa_definition":          ("Question", "Knowledge"),
    "qa_maths":               ("Question", "Knowledge"),
    "qa_currency":            ("Question", "Knowledge"),
    "qa_stock":               ("Question", "Knowledge"),
    # ── recommendation (scenario 6) ──
    "recommendation_events":    ("Question", "Social"),
    "recommendation_locations": ("Question", "Transport"),
    "recommendation_movies":    ("Question", "Media"),
    # ── social (scenario 0) ──
    "social_query":           ("Question", "Communication"),
    "social_post":            ("Command",  "Communication"),
    # ── takeaway (scenario 14) ──
    "takeaway_order":         ("Command",  "Shopping"),
    "takeaway_query":         ("Question", "Shopping"),
    # ── transport (scenario 1) ──
    "transport_query":        ("Question",  "Transport"),
    "transport_traffic":      ("Question",  "Transport"),
    "transport_ticket":       ("Command",   "Transport"),
    "transport_taxi":         ("Command",   "Transport"),
    # ── weather (scenario 17) ──
    "weather_query":          ("Question",  "Weather"),
}

assert len(MASSIVE_INTENT_MAP) == 60, f"Expected 60 intents, got {len(MASSIVE_INTENT_MAP)}"


# ──────────────────────────────────────────────
# MASSIVE slot → our slot type
# ──────────────────────────────────────────────
# MASSIVE has 55 slot types. Mapped ones get re-tagged B-/I- with our label.
# Unmapped ones (None) → label -100 in training (ignore_index).
MASSIVE_SLOT_MAP = {
    # Direct or close matches
    "device_type":          "device_type",
    "house_place":          "room",
    "time":                 "time_value",
    "timeofday":            "time_value",
    "date":                 "date_value",
    "general_frequency":    "frequency",
    "music_genre":          "genre_type",
    "artist_name":          "artist_name",
    "person":               "person_name",
    "relation":             "person_name",
    "place_name":           "location",
    "event_name":           "event_name",
    "song_name":            "media_item",
    "radio_name":           "media_item",
    "podcast_name":         "media_item",
    "audiobook_name":       "media_item",
    "movie_name":           "media_item",
    "playlist_name":        "media_item",
    "game_name":            "media_item",
    "music_album":          "media_item",
    "news_topic":           "topic",
    "sport_type":           "topic",
    "food_type":            "topic",
    "meal_type":            "topic",
    "cooking_type":         "topic",
    "definition_word":      "topic",
    "movie_type":           "genre_type",
    "game_type":            "genre_type",
    "audiobook_author":     "artist_name",
    "color_type":           "setting_value",
    "change_amount":        "setting_value",
    "player_setting":       "setting_value",
    "list_name":            "reminder_content",
    "transport_type":       "topic",
    "transport_name":       "location",
    "time_zone":            "time_value",
    # Newly mapped (was None → IGNORE, now mapped to closest type)
    "alarm_type":           "setting_type",
    "app_name":             "media_item",
    "business_name":        "location",
    "business_type":        "topic",
    "coffee_type":          "topic",
    "currency_name":        "topic",
    "drink_type":           "topic",
    "ingredient":           "topic",
    "joke_type":            "genre_type",
    "media_type":           "genre_type",
    "music_descriptor":     "genre_type",
    "podcast_descriptor":   "genre_type",
    "transport_agency":     "location",
    "transport_descriptor": "topic",
    "weather_descriptor":   "topic",
    # Unmapped → None → label -100 (ignore in loss)
    "email_address":        None,
    "email_folder":         None,
    "order_type":           None,
    "personal_info":        None,
}

# Validate all 55 MASSIVE slot types are covered
_EXPECTED_MASSIVE_SLOTS = {
    "alarm_type", "app_name", "artist_name", "audiobook_author", "audiobook_name",
    "business_name", "business_type", "change_amount", "coffee_type", "color_type",
    "cooking_type", "currency_name", "date", "definition_word", "device_type",
    "drink_type", "email_address", "email_folder", "event_name", "food_type",
    "game_name", "game_type", "general_frequency", "house_place", "ingredient",
    "joke_type", "list_name", "meal_type", "media_type", "movie_name",
    "movie_type", "music_album", "music_descriptor", "music_genre", "news_topic",
    "order_type", "person", "personal_info", "place_name", "player_setting",
    "playlist_name", "podcast_descriptor", "podcast_name", "radio_name", "relation",
    "song_name", "sport_type", "time", "time_zone", "timeofday",
    "transport_agency", "transport_descriptor", "transport_name", "transport_type",
    "weather_descriptor",
}

_mapped_slots = set(MASSIVE_SLOT_MAP.keys())
_missing = _EXPECTED_MASSIVE_SLOTS - _mapped_slots
_extra = _mapped_slots - _EXPECTED_MASSIVE_SLOTS
assert not _missing, f"Missing MASSIVE slots in map: {_missing}"
assert not _extra, f"Extra slots in map not in MASSIVE: {_extra}"


# ──────────────────────────────────────────────
# Utility: write label files (for Java side)
# ──────────────────────────────────────────────
def write_label_files(output_dir: str):
    """Write speech_act_labels.txt, domain_labels.txt, slot_labels.txt."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    for filename, labels in [
        ("speech_act_labels.txt", SPEECH_ACTS),
        ("domain_labels.txt", DOMAINS),
        ("slot_labels.txt", SLOT_LABELS),
    ]:
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for label in labels:
                f.write(label + "\n")


if __name__ == "__main__":
    print(f"Speech acts: {len(SPEECH_ACTS)}")
    print(f"Domains: {len(DOMAINS)}")
    print(f"Slot types: {len(SLOT_TYPES)} → BIO labels: {len(SLOT_LABELS)}")
    print(f"MASSIVE intents mapped: {len(MASSIVE_INTENT_MAP)}")
    print(f"MASSIVE slots mapped: {sum(1 for v in MASSIVE_SLOT_MAP.values() if v is not None)}")
    print(f"MASSIVE slots ignored: {sum(1 for v in MASSIVE_SLOT_MAP.values() if v is None)}")

    # Coverage check
    mapped_combos = set(MASSIVE_INTENT_MAP.values())
    print(f"\nUnique (speech_act, domain) combos from MASSIVE: {len(mapped_combos)}")
    for sa, dom in sorted(mapped_combos):
        print(f"  {sa} × {dom}")

    write_label_files("data/labels")
    print("\nLabel files written to data/labels/")
