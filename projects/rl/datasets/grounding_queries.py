"""Canonical template queries for label-supervised text-to-mask samples.

These templates intentionally encode only labels supplied by the source
dataset.  They are not synthetic referring expressions and do not add spatial
relations or unannotated attributes.
"""

from __future__ import annotations


# COCO-Stuff ``stuffthingmaps`` reserve PNG values 91..181 for these 91
# semantic-Stuff labels, in the listed order.  The map stores category id - 1.
COCO_STUFF_CLASS_NAMES = (
    "banner", "blanket", "branch", "bridge", "building-other", "bush", "cabinet", "cage",
    "cardboard", "carpet", "ceiling-other", "ceiling-tile", "cloth", "clothes", "clouds",
    "counter", "cupboard", "curtain", "desk-stuff", "dirt", "door-stuff", "fence",
    "floor-marble", "floor-other", "floor-stone", "floor-tile", "floor-wood", "flower",
    "fog", "food-other", "fruit", "furniture-other", "grass", "gravel", "ground-other",
    "hill", "house", "leaves", "light", "mat", "metal", "mirror-stuff", "moss",
    "mountain", "mud", "napkin", "net", "paper", "pavement", "pillow", "plant-other",
    "plastic", "platform", "playingfield", "railing", "railroad", "river", "road", "rock",
    "roof", "rug", "salad", "sand", "sea", "shelf", "sky-other", "skyscraper", "snow",
    "solid-other", "stairs", "stone", "straw", "structural-other", "table", "tent",
    "textile-other", "towel", "tree", "vegetable", "wall-brick", "wall-concrete",
    "wall-other", "wall-panel", "wall-stone", "wall-tile", "wall-wood", "water-other",
    "waterdrops", "window-blind", "window-other", "wood",
)

COCO_STUFF_NAME_BY_PIXEL_VALUE = {
    91 + index: name for index, name in enumerate(COCO_STUFF_CLASS_NAMES)
}


def label_text(value: str) -> str:
    """Convert official dash/underscore category names into prompt text."""
    text = str(value).replace("_", " ").replace("-", " ").strip()
    if not text:
        raise ValueError("Grounding label must be non-empty.")
    return " ".join(text.split())


def cocostuff_grounding_query(pixel_value: int) -> str:
    try:
        return f"the {label_text(COCO_STUFF_NAME_BY_PIXEL_VALUE[int(pixel_value)])}"
    except KeyError as error:
        raise ValueError(f"Unsupported COCO-Stuff PNG value: {pixel_value}") from error


def paco_part_grounding_query(part_name: str, parent_name: str) -> str:
    return f"the {label_text(part_name)} of the {label_text(parent_name)}"
