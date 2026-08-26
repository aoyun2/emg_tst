from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np


# Exact CMU clips whose official descriptions normalize to one of:
# walk, walking, walkforward, normalwalkforward, walkstraight, normalwalk.
# Activity labels were read from the official CMU index on 2026-08-25.
CMU_ACTIVITY_INDEX_URL = (
    "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/"
    "master/cmu-mocap-index-text.txt"
)
CMU_ACTIVITY_INDEX_SHA256 = (
    "b611669b7e8e4cb653cbda3b47ebc42c4151d019e4d28f2ed3e052e4e240ffbd"
)
LEVEL_WALKING_NORMALIZED_LABELS = (
    "walk",
    "walking",
    "walkforward",
    "normalwalkforward",
    "walkstraight",
    "normalwalk",
)

LEVEL_WALKING_CLIP_IDS = frozenset(
    """
    CMU_002_01 CMU_002_02 CMU_005_01 CMU_006_01
    CMU_007_01 CMU_007_02 CMU_007_03 CMU_007_06 CMU_007_07 CMU_007_08 CMU_007_09 CMU_007_10 CMU_007_11
    CMU_008_01 CMU_008_02 CMU_008_03 CMU_008_06 CMU_008_08 CMU_008_09 CMU_008_10
    CMU_010_04
    CMU_016_15 CMU_016_16 CMU_016_21 CMU_016_22 CMU_016_31 CMU_016_32 CMU_016_47 CMU_016_58
    CMU_026_01 CMU_027_01 CMU_029_01 CMU_032_01 CMU_032_02
    CMU_035_01 CMU_035_02 CMU_035_03 CMU_035_04 CMU_035_05 CMU_035_06 CMU_035_07 CMU_035_08 CMU_035_09 CMU_035_10 CMU_035_11 CMU_035_12 CMU_035_13 CMU_035_14 CMU_035_15 CMU_035_16 CMU_035_28 CMU_035_29 CMU_035_30 CMU_035_31 CMU_035_32 CMU_035_33 CMU_035_34
    CMU_038_01 CMU_038_02 CMU_045_01 CMU_046_01 CMU_049_01
    CMU_069_01 CMU_069_02 CMU_069_03 CMU_069_04 CMU_069_05
    CMU_082_11
    CMU_091_02 CMU_091_04 CMU_091_29 CMU_091_31 CMU_091_34 CMU_091_36 CMU_091_57
    CMU_105_02 CMU_105_29 CMU_105_31 CMU_105_34 CMU_105_36 CMU_105_57
    CMU_114_13 CMU_114_14 CMU_114_15 CMU_120_20
    CMU_133_21 CMU_133_22 CMU_133_23 CMU_139_28 CMU_143_32
    """.split()
)


def filter_expert_bank_to_level_walking(bank: Any) -> tuple[Any, np.ndarray]:
    """Return an exact row-preserving view containing only labeled level walking."""
    clip_ids = np.asarray(bank.clip_id, dtype=object).reshape(-1)
    keep = np.asarray(
        [str(clip_id) in LEVEL_WALKING_CLIP_IDS for clip_id in clip_ids],
        dtype=np.bool_,
    )
    indices = np.flatnonzero(keep).astype(np.int64)
    if len(LEVEL_WALKING_CLIP_IDS) != 90:
        raise RuntimeError("The fixed level-walking whitelist must contain 90 unique clips")
    if indices.size != 157:
        raise RuntimeError(
            f"Expected 157 eligible expert snippets, found {int(indices.size)}"
        )
    values = {
        field.name: np.asarray(getattr(bank, field.name))[indices]
        for field in fields(bank)
    }
    return type(bank)(**values), indices
