"""
CMU Graphics Lab Motion Capture Database — subject / trial catalog.

The CMU mocap database contains 2605 trials across 144 subjects performing
a wide range of motions.  Of these, 2435 were converted to BVH format
(by B. Hahne, cgspeed.com, 2008/2010) and are available for download.

This module provides a **complete** catalog of all 2435 BVH-converted
trials with automatic category assignment from subject-level metadata.

URL pattern
-----------
  https://codewelt.com/dl/cmuconvert-mb2/{subject:02d}_{trial:02d}.bvh

Motion categories (aligned with CMU's own labeling)
----------------------------------------------------
  walk, run, jump, dance, climb, sport, exercise, sit_stand, reach_lift,
  balance, misc
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrialInfo:
    subject: int
    trial: int
    category: str
    description: str

    @property
    def filename(self) -> str:
        return f"{self.subject:02d}_{self.trial:02d}.bvh"

    @property
    def url(self) -> str:
        return f"https://codewelt.com/dl/cmuconvert-mb2/{self.filename}"


# ---------------------------------------------------------------------------
# Known motion categories
# ---------------------------------------------------------------------------

CATEGORIES: List[str] = [
    "walk",
    "run",
    "jump",
    "dance",
    "climb",
    "sport",
    "exercise",
    "sit_stand",
    "reach_lift",
    "balance",
    "misc",
]

# Category descriptions for user-facing messages
CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "walk":       "Walking at various speeds, directions, and surfaces",
    "run":        "Running and jogging",
    "jump":       "Jumping, hopping, and landing",
    "dance":      "Dancing and rhythmic movement",
    "climb":      "Stair climbing and stepping over obstacles",
    "sport":      "Sports motions (basketball, soccer, golf, etc.)",
    "exercise":   "Exercises, stretching, and calisthenics",
    "sit_stand":  "Sitting, standing, and transitions between them",
    "reach_lift": "Reaching, bending, lifting, and carrying",
    "balance":    "Balancing, stumbling, and recovery motions",
    "misc":       "Miscellaneous / unclassified",
}


# ---------------------------------------------------------------------------
# Complete subject registry — derived from B. Hahne's BVH conversion index.
# ---------------------------------------------------------------------------
# Format: subject_number → (first_trial, last_trial, category, description)
#
# All 109 subjects that have BVH conversions are listed here.  Trial ranges
# are contiguous (first_trial .. last_trial inclusive).
#
# Subjects 4, 44, 48, 50–53, 57–59, 65–68, 71–73, 84, 92, 95–101,
# 109–110, 112, 116–117, 119, 121, 129–130 have NO BVH files.

_SUBJECT_REGISTRY: Dict[int, Tuple[int, int, str, str]] = {
    # subj: (first_trial, last_trial, category, description)
    1:   (1, 14, "climb",      "playground — climb, swing, hang"),
    2:   (1, 10, "misc",       "various expressions and behaviors"),
    3:   (1,  4, "walk",       "walk on uneven terrain"),
    5:   (1, 20, "dance",      "modern dance"),
    6:   (1, 15, "sport",      "basketball — dribble, shoot"),
    7:   (1, 12, "walk",       "walk"),
    8:   (1, 11, "walk",       "walk"),
    9:   (1, 12, "run",        "run"),
    10:  (1,  6, "sport",      "kick soccer ball"),
    11:  (1,  1, "sport",      "kick soccer ball"),
    12:  (1,  4, "walk",       "tai chi, walk"),
    13:  (1, 42, "misc",       "various everyday behaviors"),
    14:  (1, 37, "misc",       "various everyday behaviors"),
    15:  (1, 14, "misc",       "everyday behaviors, dance moves"),
    16:  (1, 58, "walk",       "run, jump, walk"),
    17:  (1, 10, "walk",       "different walking styles"),
    18:  (1, 15, "misc",       "human interaction (subject A)"),
    19:  (1, 15, "misc",       "human interaction (subject B)"),
    20:  (1, 13, "misc",       "human interaction — play (subject A)"),
    21:  (1, 13, "misc",       "human interaction — play (subject B)"),
    22:  (1, 25, "misc",       "human interaction (subject A)"),
    23:  (1, 25, "misc",       "human interaction (subject B)"),
    24:  (1,  1, "misc",       "nursery rhymes"),
    25:  (1,  1, "misc",       "nursery rhymes"),
    26:  (1, 11, "misc",       "nursery rhymes, basketball, bending"),
    27:  (1, 11, "misc",       "recreation, nursery rhymes"),
    28:  (1, 19, "misc",       "recreation, nursery rhymes, animal pantomime"),
    29:  (1, 25, "misc",       "recreation, nursery rhymes, animal pantomime"),
    30:  (1, 23, "misc",       "recreation, nursery rhymes, animal pantomime"),
    31:  (1, 21, "misc",       "recreation, nursery rhymes, animal pantomime"),
    32:  (1, 22, "misc",       "recreation, nursery rhymes, animal pantomime"),
    33:  (1,  2, "sport",      "throw/catch football (subject A)"),
    34:  (1,  2, "sport",      "throw/catch football (subject B)"),
    35:  (1, 34, "walk",       "walk, run"),
    36:  (1, 37, "walk",       "walk on uneven terrain"),
    37:  (1,  1, "walk",       "walk"),
    38:  (1,  4, "walk",       "walk, run"),
    39:  (1, 14, "climb",      "playground — climb, swing, hang"),
    40:  (2, 12, "misc",       "navigate corners, interact with stepstool"),
    41:  (2, 11, "misc",       "navigate corners, interact with stepstool"),
    42:  (1,  1, "exercise",   "stretch"),
    43:  (1,  3, "climb",      "swing on playground equipment"),
    45:  (1,  1, "walk",       "walk"),
    46:  (1,  1, "walk",       "walk"),
    47:  (1,  1, "walk",       "walk"),
    49:  (1, 22, "dance",      "modern dance, gymnastics"),
    54:  (1, 27, "misc",       "animal behaviors (pantomime)"),
    55:  (1, 28, "misc",       "animal behaviors (pantomime)"),
    56:  (1,  8, "walk",       "vignettes — locomotion, motion transitions"),
    60:  (1, 15, "dance",      "salsa dance"),
    61:  (1, 15, "dance",      "salsa dance"),
    62:  (1, 25, "misc",       "construction work, random motions"),
    63:  (1,  1, "sport",      "golf swing"),
    64:  (1, 30, "sport",      "golf — swing, putt, placing tee/ball"),
    69:  (1, 75, "walk",       "walking — forward, turns, sideways, backwards"),
    70:  (1, 13, "reach_lift", "carry suitcase"),
    74:  (1, 20, "walk",       "kicks and walking on slopes"),
    75:  (1, 20, "jump",       "jumps, hopscotch, sits"),
    76:  (1, 11, "balance",    "avoidance motions"),
    77:  (1, 34, "walk",       "careful actions — sneaking, limping, searching"),
    78:  (1, 35, "walk",       "walking — basketball context, turns, drives"),
    79:  (1, 96, "misc",       "actor everyday activities"),
    80:  (1, 73, "misc",       "assorted motions"),
    81:  (1, 18, "walk",       "pushing box, jumping off ledge, walks"),
    82:  (1, 18, "walk",       "jumping, pushing, emotional walks"),
    83:  (1, 68, "walk",       "steps — sidesteps, stairs, hops, turns"),
    85:  (1, 15, "jump",       "jumps, flips, breakdance"),
    86:  (1, 15, "sport",      "sports and various activities"),
    87:  (1,  5, "exercise",   "acrobatics"),
    88:  (1, 11, "exercise",   "acrobatics"),
    89:  (1,  6, "exercise",   "acrobatics"),
    90:  (1, 36, "exercise",   "cartwheels, acrobatics, dances"),
    91:  (1, 62, "walk",       "walks and turns"),
    93:  (1,  8, "dance",      "Charleston dance"),
    94:  (1, 16, "dance",      "Indian dance"),
    102: (1, 33, "sport",      "basketball"),
    103: (1,  8, "dance",      "Charleston dance"),
    104: (1, 57, "walk",       "walks, jogs, runs — style variations"),
    105: (1, 62, "walk",       "walks, jumps, turns"),
    106: (1, 34, "misc",       "female general subject"),
    107: (1, 14, "walk",       "walking with obstacles"),
    108: (1, 28, "walk",       "walking with obstacles"),
    111: (1, 41, "walk",       "pregnant woman — walking"),
    113: (1, 29, "walk",       "post-pregnant woman — walking"),
    114: (1, 16, "walk",       "pregnant woman — walking"),
    115: (1, 10, "reach_lift", "bending over — picking up box"),
    118: (1, 32, "jump",       "jumping"),
    120: (1, 22, "walk",       "various style walks — Mickey, zombie, robot, alien"),
    122: (1, 68, "walk",       "varying height and length steps"),
    123: (1, 13, "reach_lift", "carry suitcase with varying weights"),
    124: (1, 13, "sport",      "sports — baseball, basketball, frisbee"),
    125: (1,  7, "sport",      "swimming"),
    126: (1, 14, "sport",      "swimming"),
    127: (1, 38, "run",        "action adventure — running, jumping, ducking, rolling"),
    128: (1, 11, "run",        "action adventure — running, ducking, rolling, stopping"),
    131: (1, 14, "dance",      "Michael Jackson styled motions"),
    132: (1, 56, "walk",       "varying weird walks"),
    133: (1, 26, "walk",       "baby styled walk — crawl, jump, zigzag"),
    134: (1, 15, "sport",      "skateboard motions"),
    135: (1, 11, "walk",       "martial arts walks — karate katas"),
    136: (1, 33, "walk",       "weird walks — bent, crouched, flamingo"),
    137: (1, 42, "misc",       "stylized motions — cat, dinosaur, drunk, etc."),
    138: (1, 55, "walk",       "marching, walking and talking"),
    139: (1, 34, "walk",       "action walks — sneaking, wounded, looking around"),
    140: (1,  9, "sit_stand",  "getting up from ground"),
    141: (1, 34, "misc",       "general subject — run, jump, walk, sit"),
    142: (1, 22, "walk",       "stylized walks — childish, cool, depressed"),
    143: (1, 42, "misc",       "general subject — run, jump, walk, carry, sweep"),
    144: (1, 34, "sport",      "martial arts — cartwheels, kicks, punches, yoga"),
}


# ---------------------------------------------------------------------------
# Build the typed catalog from the registry
# ---------------------------------------------------------------------------

def _build_catalog() -> List[TrialInfo]:
    """Generate the full catalog from the subject registry."""
    catalog: List[TrialInfo] = []
    for subj, (first, last, cat, desc) in sorted(_SUBJECT_REGISTRY.items()):
        for trial in range(first, last + 1):
            catalog.append(TrialInfo(
                subject=subj,
                trial=trial,
                category=cat,
                description=desc,
            ))
    return catalog


CATALOG: List[TrialInfo] = _build_catalog()


# ---------------------------------------------------------------------------
# Alternative / larger datasets for future expansion
# ---------------------------------------------------------------------------
#
# AMASS (Archive of Motion Capture as Surface Shapes)
#   - 40+ hours of motion; ~11,000 sequences across 300+ subjects
#   - Unifies 15 mocap databases (CMU, BMLrub, EKUT, KIT, SFU, etc.)
#   - Requires free registration: https://amass.is.tue.mpg.de
#   - Format: SMPL-H parameters (not BVH); needs conversion for our loader
#
# KIT Motion-Language Dataset
#   - 3,911 recordings, 11+ hours, 100+ motion types
#   - Annotated with natural-language descriptions
#   - Free download: https://motion-annotation.humanoids.kit.edu/dataset/
#   - Format: C3D + MMM XML; BVH conversion available
#
# HDM05 (MPI Informatik)
#   - 3.0 hours, 130 motion classes, 5 subjects
#   - Reliable labeling; good for classification / retrieval benchmarks
#   - https://resources.mpi-inf.mpg.de/HDM05/
#   - Format: native BVH — directly loadable by our bvh_parser
#
# BMLrub (Bielefeld)
#   - 115 subjects, 9 activities (walk, jog, run, etc.), 3,000+ trials
#   - Included in AMASS; also available standalone (free registration)
#   - https://motion-db.humanoids.kit.edu
#
# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def by_category(category: str) -> List[TrialInfo]:
    """Return all trials for a given category."""
    return [t for t in CATALOG if t.category == category]


def by_subject(subject: int) -> List[TrialInfo]:
    """Return all cataloged trials for a given subject number."""
    return [t for t in CATALOG if t.subject == subject]


def available_categories() -> List[str]:
    """Return sorted list of categories that have at least one trial."""
    return sorted({t.category for t in CATALOG})


def summary() -> Dict[str, int]:
    """Return {category: n_trials} counts."""
    counts: Dict[str, int] = {}
    for t in CATALOG:
        counts[t.category] = counts.get(t.category, 0) + 1
    return dict(sorted(counts.items()))


def filter_trials(
    categories: Optional[Sequence[str]] = None,
    subjects: Optional[Sequence[int]] = None,
    max_per_category: Optional[int] = None,
) -> List[TrialInfo]:
    """
    Filter the catalog by category and/or subject.

    Parameters
    ----------
    categories        : keep only these categories (None = all)
    subjects          : keep only these subject numbers (None = all)
    max_per_category  : cap per category (None = unlimited)

    Returns
    -------
    Filtered list of TrialInfo.
    """
    result = CATALOG
    if categories is not None:
        cat_set: Set[str] = set(categories)
        result = [t for t in result if t.category in cat_set]
    if subjects is not None:
        sub_set: Set[int] = set(subjects)
        result = [t for t in result if t.subject in sub_set]
    if max_per_category is not None:
        capped: List[TrialInfo] = []
        counts: Dict[str, int] = {}
        for t in result:
            counts[t.category] = counts.get(t.category, 0) + 1
            if counts[t.category] <= max_per_category:
                capped.append(t)
        result = capped
    return result
