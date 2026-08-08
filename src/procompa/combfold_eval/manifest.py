"""Stoichiometry-manifest parsing + CombFold folder resolution.

When CombFold output folders are named by UniProt composition (e.g.
``P06782x1_P12904x1_P34164x1_pool_output``) rather than by ``complex_ac``,
the pipeline's built-in name-matching discovery (``find_complex_dirs``) cannot
find them.  This module reads a stoichiometry-prediction manifest that maps
each ``complex_ac`` to one or more candidate compositions, derives the expected
folder name for each, and checks which folders actually exist on disk.

The manifest is the CSV produced by the upstream stoichiometry-prediction
step.  Expected columns (only these are used):

    complex_ac    e.g. "CPX-2800"
    identifiers   "P06782(1)|P12904(1)|P34164(1)"  -- reference/"true" stoich
    pred_1        '{"P06782":1,"P12904":2,"P34164":1},{"rank":1,...}'
    pred_2        same shape as pred_1
    pred_3        same shape as pred_1

Non-UniProt tokens in ``identifiers`` (e.g. ``CHEBI:29105(1)`` ligands) are
silently dropped -- they are not subunits and never appear in folder names.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

# UniProt accession pattern (official) -- kept in sync with pipeline.py.
_UNIPROT = r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
_ACC_ONLY = re.compile(_UNIPROT)

# "P06782(1)|P12904(1)|P34164(1)" -> captures accession + count
_IDENT_RE = re.compile(r"([A-Za-z0-9_.:-]+)\((\d+)\)")

# First {...} object in a pred_N field (before the rank/probability dict)
_PRED_DICT_RE = re.compile(r"^\s*(\{[^{}]*\})")


def _is_uniprot(acc: str) -> bool:
    return bool(_ACC_ONLY.fullmatch(acc))


def parse_identifiers(text: str) -> Dict[str, int]:
    """'P06782(1)|P12904(1)|P34164(1)' -> {'P06782':1,'P12904':1,'P34164':1}.

    Non-UniProt tokens (e.g. 'CHEBI:29105(1)' ligands) are dropped.
    """
    out: Dict[str, int] = {}
    if not isinstance(text, str):
        return out
    for m in _IDENT_RE.finditer(text):
        acc, cnt = m.group(1), int(m.group(2))
        if _is_uniprot(acc):
            out[acc] = cnt
    return out


def parse_pred_field(text: str) -> Dict[str, int]:
    """'{"P06782":1,"P12904":2,"P34164":1},{"rank":1,...}' -> stoich dict.

    Only the first ``{...}`` object (before the rank/probability one) is used.
    """
    if not isinstance(text, str) or not text.strip():
        return {}
    m = _PRED_DICT_RE.match(text.strip())
    if not m:
        return {}
    try:
        d = json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}
    return {str(k): int(v) for k, v in d.items() if _is_uniprot(str(k))}


def folder_stub(stoich: Dict[str, int]) -> str:
    """{'P14736':1,'P32628':1} -> 'P14736x1_P32628x1' (accessions sorted)."""
    return "_".join(f"{acc}x{cnt}" for acc, cnt in sorted(stoich.items()))


def candidate_stoichiometries(manifest_row: pd.Series) -> List[Tuple[str, Dict[str, int]]]:
    """Return [(source_label, stoich_dict), ...] for one manifest row.

    Sources: identifiers (reference/"true"), pred_1, pred_2, pred_3.
    Deduplicated by folder stub; when two sources produce the same stub their
    labels are merged (e.g. 'identifiers+pred_1').
    """
    stub_to_label: Dict[str, str] = {}
    ordered: List[Tuple[str, Dict[str, int]]] = []
    sources = [
        ("identifiers", parse_identifiers(manifest_row.get("identifiers"))),
        ("pred_1", parse_pred_field(manifest_row.get("pred_1"))),
        ("pred_2", parse_pred_field(manifest_row.get("pred_2"))),
        ("pred_3", parse_pred_field(manifest_row.get("pred_3"))),
    ]
    for label, stoich in sources:
        if not stoich:
            continue
        stub = folder_stub(stoich)
        if stub in stub_to_label:
            stub_to_label[stub] = stub_to_label[stub] + "+" + label
        else:
            stub_to_label[stub] = label
            ordered.append((stub, stoich))
    return [(stub_to_label[stub], stoich) for stub, stoich in ordered]


def resolve_folders(combfold_base: str, stoich: Dict[str, int],
                    suffixes: Tuple[str, ...]) -> List[Tuple[str, str]]:
    """Return [(suffix, abs_path), ...] for each suffix whose folder exists.

    e.g. stoich={'P14736':1,'P32628':1}, suffixes=('pool_output','pair_output')
    -> [('pool_output', '/.../P14736x1_P32628x1_pool_output')] if only that
    folder exists on disk.
    """
    stub = folder_stub(stoich)
    hits: List[Tuple[str, str]] = []
    for suffix in suffixes:
        cand = os.path.join(combfold_base, f"{stub}_{suffix}")
        if os.path.isdir(cand):
            hits.append((suffix, os.path.abspath(cand)))
    return hits


def load_manifest(path: str) -> Dict[str, pd.Series]:
    """Read the manifest CSV and return {complex_ac: row}.

    Raises FileNotFoundError if the path does not exist, or ValueError if the
    required columns are missing (fail fast at batch start, not per-row).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest file not found: {path}")
    df = pd.read_csv(path)
    required = {"complex_ac", "identifiers"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"manifest missing required column(s): {sorted(missing)}; "
            f"found: {list(df.columns)}")
    return {str(r["complex_ac"]).strip(): r for _, r in df.iterrows()}
