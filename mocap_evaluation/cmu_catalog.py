"""
CMU Graphics Lab Motion Capture Database — subject / trial catalog.

The CMU mocap database contains 2605 trials across 144 subjects performing
a wide range of motions.  This module provides a structured catalog so that
we can selectively download and load trials by motion category.

URL pattern
-----------
  https://codewelt.com/dl/cmuconvert-mb2/{subject:02d}_{trial:02d}.bvh

Motion categories (aligned with CMU's own labeling)
----------------------------------------------------
  walk, run, jump, dance, climb, sport, exercise, sit_stand, reach_lift,
  balance, misc

Each entry is (subject, trial, category, description).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

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
# Curated catalog
# ---------------------------------------------------------------------------
# Sources:
#   - CMU mocap database subject listing: http://mocap.cs.cmu.edu/motcat.php
#   - Trial descriptions from the database HTML pages
#
# This is a *representative* subset of the full 2605 trials.  It covers the
# breadth of human free motion with enough variety per category for robust
# motion matching.  Users can extend this list or download additional trials.

_RAW_CATALOG = [
    # ── Walking ──────────────────────────────────────────────────────
    (2, 1, "walk", "walk"),
    (2, 2, "walk", "walk"),
    (2, 3, "walk", "walk"),
    (7, 1, "walk", "slow walk"),
    (7, 2, "walk", "moderate walk"),
    (7, 3, "walk", "fast walk"),
    (7, 4, "walk", "slow walk"),
    (7, 5, "walk", "moderate walk"),
    (7, 6, "walk", "fast walk"),
    (7, 7, "walk", "walk with hands in pockets"),
    (7, 8, "walk", "walk with arms folded"),
    (7, 9, "walk", "walk looking around"),
    (7, 10, "walk", "walk"),
    (7, 11, "walk", "walk"),
    (7, 12, "walk", "walk"),
    (8, 1, "walk", "walk"),
    (8, 2, "walk", "walk"),
    (8, 3, "walk", "walk"),
    (8, 4, "walk", "walk"),
    (8, 5, "walk", "walk"),
    (8, 6, "walk", "walk"),
    (8, 7, "walk", "walk"),
    (8, 8, "walk", "walk"),
    (8, 9, "walk", "walk"),
    (8, 10, "walk", "walk"),
    (8, 11, "walk", "walk"),
    (12, 1, "walk", "walk straight"),
    (12, 2, "walk", "walk forward and back"),
    (12, 3, "walk", "slow walk"),
    (35, 1, "walk", "walk"),
    (35, 2, "walk", "walk"),
    (35, 3, "walk", "walk - arms swinging"),
    (35, 17, "walk", "walk"),
    (35, 18, "walk", "walk - casual"),
    (35, 19, "walk", "walk"),
    (35, 20, "walk", "walk on toes"),
    (35, 21, "walk", "walk backwards"),
    (35, 22, "walk", "walk - hands behind back"),
    (35, 23, "walk", "walk sideways"),
    (35, 24, "walk", "walk in circle"),
    (35, 25, "walk", "walk - angry"),
    (35, 26, "walk", "walk - happy/bouncy"),
    (36, 1, "walk", "walk normal"),
    (36, 2, "walk", "walk - deterministic"),
    (36, 3, "walk", "walk - tired/dragging"),
    (36, 4, "walk", "walk - march"),
    (36, 5, "walk", "walk - tippy toe"),
    (36, 6, "walk", "walk - elderly"),
    (36, 7, "walk", "walk - drunk"),
    (36, 8, "walk", "walk - injured leg"),
    (36, 9, "walk", "walk - limp"),
    (36, 10, "walk", "walk - swagger"),
    (36, 11, "walk", "walk - brisk"),
    (36, 12, "walk", "walk - careful/on ice"),
    (36, 13, "walk", "walk - coolly"),
    (36, 14, "walk", "walk - depressed"),
    (36, 15, "walk", "walk - silly/childlike"),
    (39, 1, "walk", "walk normal pace"),
    (39, 2, "walk", "walk with bag"),
    (39, 3, "walk", "walk looking around"),
    (39, 4, "walk", "walk forward and turn"),

    # ── Running ──────────────────────────────────────────────────────
    (2, 4, "run", "jog"),
    (9, 1, "run", "run"),
    (9, 2, "run", "run"),
    (9, 3, "run", "run"),
    (9, 4, "run", "run fast"),
    (9, 5, "run", "run slow"),
    (9, 6, "run", "run"),
    (9, 7, "run", "run"),
    (9, 8, "run", "run"),
    (9, 9, "run", "run"),
    (9, 10, "run", "run"),
    (9, 11, "run", "run"),
    (9, 12, "run", "run"),
    (16, 15, "run", "run"),
    (16, 17, "run", "run"),
    (16, 21, "run", "run"),
    (16, 31, "run", "run fast"),
    (16, 32, "run", "run slow"),
    (16, 33, "run", "jog"),
    (16, 34, "run", "jog to walk transition"),
    (16, 35, "run", "run and stop"),
    (16, 36, "run", "run and turn"),
    (16, 47, "run", "run"),
    (16, 49, "run", "run"),
    (16, 50, "run", "run slow"),
    (16, 51, "run", "run fast"),
    (35, 4, "run", "jog"),
    (35, 5, "run", "jog"),
    (35, 6, "run", "run"),
    (35, 7, "run", "run fast"),
    (35, 8, "run", "run and zigzag"),

    # ── Jumping ──────────────────────────────────────────────────────
    (13, 1, "jump", "jump forward"),
    (13, 2, "jump", "jump in place"),
    (13, 3, "jump", "jump sideways"),
    (13, 4, "jump", "jump forward"),
    (13, 5, "jump", "hop on one foot"),
    (13, 10, "jump", "jump forward"),
    (13, 11, "jump", "jump in place"),
    (13, 12, "jump", "jump with twist"),
    (13, 13, "jump", "leap"),
    (13, 14, "jump", "jump"),
    (13, 15, "jump", "jump"),
    (13, 16, "jump", "jump"),
    (13, 17, "jump", "hop"),
    (13, 18, "jump", "jump"),
    (13, 19, "jump", "jump"),
    (13, 20, "jump", "standing long jump"),
    (16, 1, "jump", "jump forward"),
    (16, 2, "jump", "jump in place"),
    (16, 3, "jump", "hop"),
    (16, 4, "jump", "hop on one foot"),
    (49, 1, "jump", "jump"),
    (49, 2, "jump", "jump sideways"),

    # ── Dancing ──────────────────────────────────────────────────────
    (5, 1, "dance", "modern dance"),
    (5, 2, "dance", "modern dance"),
    (5, 3, "dance", "modern dance"),
    (5, 4, "dance", "modern dance"),
    (5, 5, "dance", "modern dance"),
    (5, 6, "dance", "modern dance"),
    (5, 7, "dance", "modern dance"),
    (5, 8, "dance", "modern dance"),
    (5, 9, "dance", "modern dance"),
    (5, 10, "dance", "modern dance"),
    (5, 11, "dance", "modern dance"),
    (5, 12, "dance", "modern dance"),
    (5, 13, "dance", "modern dance"),
    (5, 14, "dance", "modern dance"),
    (5, 15, "dance", "modern dance"),
    (5, 16, "dance", "modern dance"),
    (5, 17, "dance", "modern dance"),
    (5, 18, "dance", "modern dance"),
    (5, 19, "dance", "modern dance"),
    (5, 20, "dance", "modern dance"),
    (60, 1, "dance", "salsa"),
    (60, 2, "dance", "salsa"),
    (60, 3, "dance", "salsa"),
    (60, 4, "dance", "breakdance"),
    (60, 5, "dance", "breakdance"),
    (60, 6, "dance", "breakdance"),
    (60, 7, "dance", "indian dance"),
    (60, 8, "dance", "indian dance"),
    (60, 9, "dance", "charleston"),
    (60, 10, "dance", "charleston"),
    (60, 11, "dance", "swing"),
    (60, 12, "dance", "swing"),

    # ── Stair climbing / stepping ────────────────────────────────────
    (10, 1, "climb", "walk up stairs"),
    (10, 2, "climb", "walk down stairs"),
    (10, 3, "climb", "walk up ramp"),
    (10, 4, "climb", "walk down ramp"),
    (12, 4, "climb", "walk up stairs"),
    (12, 5, "climb", "walk down stairs"),
    (35, 9, "climb", "step over obstacle"),
    (35, 10, "climb", "step over high obstacle"),
    (35, 11, "climb", "step up and down"),

    # ── Sports ───────────────────────────────────────────────────────
    (6, 1, "sport", "basketball dribble"),
    (6, 2, "sport", "basketball shoot"),
    (6, 3, "sport", "basketball layup"),
    (6, 4, "sport", "basketball crossover"),
    (6, 5, "sport", "basketball pass"),
    (6, 6, "sport", "basketball free throw"),
    (6, 7, "sport", "basketball dribble and shoot"),
    (6, 8, "sport", "basketball"),
    (6, 9, "sport", "basketball"),
    (6, 10, "sport", "basketball"),
    (6, 11, "sport", "basketball"),
    (6, 12, "sport", "basketball"),
    (6, 13, "sport", "basketball"),
    (6, 14, "sport", "basketball"),
    (6, 15, "sport", "basketball"),
    (55, 1, "sport", "golf swing"),
    (55, 2, "sport", "golf swing"),
    (55, 3, "sport", "golf putt"),
    (55, 4, "sport", "golf"),
    (56, 1, "sport", "soccer kick"),
    (56, 2, "sport", "soccer kick"),
    (56, 3, "sport", "soccer dribble"),
    (56, 4, "sport", "soccer header"),
    (56, 5, "sport", "soccer"),
    (56, 6, "sport", "soccer"),
    (56, 7, "sport", "soccer"),

    # ── Exercise / physical movements ────────────────────────────────
    (14, 1, "exercise", "stretch"),
    (14, 2, "exercise", "stretch"),
    (14, 3, "exercise", "warmup / stretch"),
    (14, 4, "exercise", "squats"),
    (14, 5, "exercise", "jumping jacks"),
    (14, 6, "exercise", "leg swings"),
    (14, 7, "exercise", "lunges"),
    (14, 8, "exercise", "torso twist"),
    (14, 9, "exercise", "arm circles"),
    (14, 10, "exercise", "toe touches"),
    (14, 11, "exercise", "sit-ups"),
    (14, 12, "exercise", "push-ups"),
    (14, 13, "exercise", "leg raises"),
    (14, 14, "exercise", "exercise routine"),
    (91, 1, "exercise", "stretch"),
    (91, 2, "exercise", "jumping jacks"),
    (91, 3, "exercise", "squats"),

    # ── Sit / stand transitions ──────────────────────────────────────
    (15, 1, "sit_stand", "sit down and stand up"),
    (15, 2, "sit_stand", "sit down and stand up"),
    (15, 3, "sit_stand", "sit down"),
    (15, 4, "sit_stand", "stand up"),
    (15, 5, "sit_stand", "sit down on floor"),
    (15, 6, "sit_stand", "lie down and get up"),
    (15, 7, "sit_stand", "stand to sit transition"),
    (15, 8, "sit_stand", "kneel and stand"),
    (15, 9, "sit_stand", "kneel down"),
    (15, 10, "sit_stand", "get up from floor"),
    (15, 11, "sit_stand", "get up from lying"),
    (15, 12, "sit_stand", "crouch"),
    (15, 13, "sit_stand", "crouch to stand"),
    (15, 14, "sit_stand", "sit down slowly"),

    # ── Reaching / lifting ───────────────────────────────────────────
    (3, 1, "reach_lift", "pick up object from floor"),
    (3, 2, "reach_lift", "reach forward"),
    (3, 3, "reach_lift", "reach high"),
    (3, 4, "reach_lift", "carry box"),
    (3, 5, "reach_lift", "place object"),
    (3, 6, "reach_lift", "pick up and carry"),
    (3, 7, "reach_lift", "reach sideways"),
    (3, 8, "reach_lift", "pick up and move"),
    (105, 1, "reach_lift", "lift heavy box"),
    (105, 2, "reach_lift", "carry heavy box"),
    (105, 3, "reach_lift", "push cart"),
    (105, 4, "reach_lift", "pull door"),

    # ── Balance / stumble / recovery ─────────────────────────────────
    (17, 1, "balance", "balance on one foot"),
    (17, 2, "balance", "balance beam walk"),
    (17, 3, "balance", "tandem walk"),
    (17, 4, "balance", "balance on toes"),
    (17, 5, "balance", "stumble recovery"),
    (17, 6, "balance", "push recovery"),
    (49, 3, "balance", "balance on one foot"),
    (49, 4, "balance", "balance on beam"),
    (49, 5, "balance", "tandem walk"),

    # ── Miscellaneous ────────────────────────────────────────────────
    (1, 1, "misc", "general motion"),
    (1, 2, "misc", "general motion"),
    (1, 3, "misc", "general motion"),
    (1, 4, "misc", "general motion"),
    (1, 5, "misc", "general motion"),
    (2, 5, "misc", "gesture"),
    (2, 6, "misc", "gesture"),
    (4, 1, "misc", "pantomime"),
    (4, 2, "misc", "pantomime"),
    (4, 3, "misc", "pantomime"),
    (143, 1, "misc", "everyday activity"),
    (143, 2, "misc", "everyday activity"),
    (143, 3, "misc", "everyday activity"),
    (144, 1, "misc", "general motion sequence"),
]

# Build the typed catalog
CATALOG: List[TrialInfo] = [
    TrialInfo(subject=s, trial=t, category=c, description=d)
    for s, t, c, d in _RAW_CATALOG
]


# ---------------------------------------------------------------------------
# Alternative / larger datasets for future expansion
# ---------------------------------------------------------------------------
#
# The CMU catalog here covers ~247 curated trials (≈ 10% of the full 2605).
# For more diverse or larger-scale matching, consider these datasets:
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
# To use HDM05 or BMLrub directly with this pipeline:
#   1. Download BVH files into mocap_data/
#   2. Run: python -m mocap_evaluation.run_evaluation --full-db --mocap-dir mocap_data/
#   Or add entries to _RAW_CATALOG above using the correct subject/trial numbering.
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
