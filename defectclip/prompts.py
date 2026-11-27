"""Prompt sets for zero-shot scoring and text-to-image defect search.

The ensemble follows the WinCLIP idea (several templates, a normal state and
an anomalous state, then average the text embeddings) but the wording is
written for this repo. It is not a copy of the WinCLIP prompt list.
"""

from __future__ import annotations

OBJECT_PHRASE = {
    "bottle": "bottle",
    "cable": "cable",
    "capsule": "capsule",
    "carpet": "carpet",
    "grid": "metal grid",
    "hazelnut": "hazelnut",
    "leather": "leather surface",
    "metal_nut": "metal nut",
    "pill": "pill",
    "screw": "screw",
    "tile": "tile",
    "toothbrush": "toothbrush",
    "transistor": "transistor",
    "wood": "wood surface",
    "zipper": "zipper",
}

# Short folder names read badly inside a sentence ("a carpet with a color").
DEFECT_PHRASE = {
    "bent": "bent part",
    "bent_lead": "bent lead",
    "bent_wire": "bent wire",
    "broken": "break",
    "broken_large": "large break",
    "broken_small": "small break",
    "broken_teeth": "broken teeth",
    "cable_swap": "swapped cable",
    "color": "color defect",
    "combined": "combination of defects",
    "contamination": "contamination",
    "crack": "crack",
    "cut": "cut",
    "cut_inner_insulation": "cut inner insulation",
    "cut_lead": "cut lead",
    "cut_outer_insulation": "cut outer insulation",
    "damaged_case": "damaged case",
    "fabric_border": "damaged fabric border",
    "fabric_interior": "damaged fabric interior",
    "faulty_imprint": "faulty imprint",
    "flip": "flipped part",
    "fold": "fold",
    "glue": "glue residue",
    "glue_strip": "glue strip",
    "gray_stroke": "gray stroke",
    "hole": "hole",
    "liquid": "liquid stain",
    "manipulated_front": "manipulated front",
    "metal_contamination": "metal contamination",
    "misplaced": "misplaced part",
    "missing_cable": "missing cable",
    "missing_wire": "missing wire",
    "oil": "oil stain",
    "pill_type": "wrong pill type",
    "poke": "poke mark",
    "poke_insulation": "poked insulation",
    "print": "print defect",
    "rough": "rough patch",
    "scratch": "scratch",
    "scratch_head": "scratched head",
    "scratch_neck": "scratched neck",
    "split_teeth": "split teeth",
    "squeeze": "squeeze deformation",
    "squeezed_teeth": "squeezed teeth",
    "thread": "stray thread",
    "thread_side": "damaged side thread",
    "thread_top": "damaged top thread",
}

NORMAL_STATES = [
    "flawless {obj}",
    "perfect {obj}",
    "unblemished {obj}",
    "{obj} without defect",
    "{obj} without flaw",
    "{obj} without damage",
]

ANOMALY_STATES = [
    "damaged {obj}",
    "defective {obj}",
    "{obj} with a defect",
    "{obj} with a flaw",
    "{obj} with damage",
    "broken {obj}",
]

TEMPLATES = [
    "a photo of a {state}.",
    "a close-up photo of a {state}.",
    "a cropped photo of the {state}.",
    "a bright photo of a {state}.",
    "a dark photo of a {state}.",
    "an industrial inspection photo of a {state}.",
]

RETRIEVAL_TEMPLATES = [
    "a photo of a {obj} with a {defect}",
    "a close-up photo of a {defect} on a {obj}",
    "an industrial inspection photo of a {obj} with a {defect}",
    "a {obj} that has a {defect}",
]


def object_phrase(category: str) -> str:
    return OBJECT_PHRASE.get(category, category.replace("_", " "))


def defect_phrase(defect: str) -> str:
    if defect in DEFECT_PHRASE:
        return DEFECT_PHRASE[defect]
    return defect.replace("_", " ")


def _expand(states: list[str], templates: list[str], obj: str) -> list[str]:
    prompts = []
    for state in states:
        filled = state.format(obj=obj)
        for template in templates:
            prompts.append(template.format(state=filled))
    return prompts


def state_prompts(
    category: str,
    variant: str = "ensemble",
    defect_names: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return (normal_prompts, anomaly_prompts) for one category.

    Variants
    --------
    minimal
        One sentence each. The baseline a prompt ablation has to beat.
    ensemble
        Templates crossed with normal and anomaly states. This is the main method.
    inspection
        Same states as the ensemble, but only the industrial-inspection template.
    defect_vocab
        The ensemble, plus anomaly sentences that name the category's defect types.
        Those names are a label vocabulary, not image supervision.
    """
    obj = object_phrase(category)
    if variant == "minimal":
        normal = [f"a photo of a flawless {obj}."]
        anomaly = [f"a photo of a damaged {obj}."]
        return normal, anomaly
    if variant == "inspection":
        templates = ["an industrial inspection photo of a {state}."]
    elif variant in {"ensemble", "defect_vocab"}:
        templates = TEMPLATES
    else:
        raise ValueError(f"Unknown prompt variant: {variant}")
    normal = _expand(NORMAL_STATES, templates, obj)
    anomaly = _expand(ANOMALY_STATES, templates, obj)
    if variant == "defect_vocab":
        if not defect_names:
            raise ValueError("defect_vocab prompts need the category's defect names")
        for name in defect_names:
            phrase = defect_phrase(name)
            anomaly.append(f"a photo of a {obj} with a {phrase}.")
            anomaly.append(f"a close-up photo of a {phrase} on a {obj}.")
    return normal, anomaly


def retrieval_prompts(category: str, defect: str) -> list[str]:
    obj = object_phrase(category)
    phrase = defect_phrase(defect)
    return [t.format(obj=obj, defect=phrase) for t in RETRIEVAL_TEMPLATES]


def retrieval_query_text(category: str, defect: str) -> str:
    """Single sentence used as the human-readable query in figures."""
    return retrieval_prompts(category, defect)[0]
