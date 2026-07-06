"""PA-100K's 26 attributes with a natural-language prompt for each.

These prompts are the "questions" we ask SigLIP about every image. SigLIP scores
how well each prompt matches the image -> that score is the attribute probability.

Groups marked below are MUTUALLY EXCLUSIVE in PA-100K (only one can be true):
  - AGE_GROUP : exactly one age band
  - VIEW_GROUP: exactly one viewpoint
Use --groups at run time to enforce argmax within each group.
"""

# (attribute_name, positive_prompt)
ATTRIBUTES = [
    ("Hat",                "a photo of a person wearing a hat"),
    ("Glasses",            "a photo of a person wearing glasses"),
    ("ShortSleeve",        "a photo of a person wearing short sleeves"),
    ("LongSleeve",         "a photo of a person wearing long sleeves"),
    ("UpperStride",        "a photo of a person wearing striped upper clothing"),
    ("UpperLogo",          "a photo of a person wearing upper clothing with a logo"),
    ("UpperPlaid",         "a photo of a person wearing plaid upper clothing"),
    ("UpperSplice",        "a photo of a person wearing patchwork upper clothing"),
    ("LowerStripe",        "a photo of a person wearing striped lower clothing"),
    ("LowerPattern",       "a photo of a person wearing patterned lower clothing"),
    ("LongCoat",           "a photo of a person wearing a long coat"),
    ("Trousers",           "a photo of a person wearing trousers"),
    ("Shorts",             "a photo of a person wearing shorts"),
    ("Skirt&Dress",        "a photo of a person wearing a skirt or a dress"),
    ("boots",              "a photo of a person wearing boots"),
    ("HandBag",            "a photo of a person carrying a handbag"),
    ("ShoulderBag",        "a photo of a person carrying a shoulder bag"),
    ("Backpack",           "a photo of a person carrying a backpack"),
    ("HoldObjectsInFront", "a photo of a person holding objects in front of them"),
    ("AgeOver60",          "a photo of an elderly person over 60 years old"),
    ("Age18-60",           "a photo of an adult person between 18 and 60 years old"),
    ("AgeLess18",          "a photo of a child or teenager under 18 years old"),
    ("Front",              "a photo of a person seen from the front"),
    ("Side",               "a photo of a person seen from the side"),
    ("Back",               "a photo of a person seen from the back"),
    ("Female",             "a photo of a woman"),
]

# index groups for mutual-exclusivity (by attribute name)
AGE_GROUP  = ["AgeOver60", "Age18-60", "AgeLess18"]
VIEW_GROUP = ["Front", "Side", "Back"]

NAMES   = [a for a, _ in ATTRIBUTES]
PROMPTS = [p for _, p in ATTRIBUTES]
