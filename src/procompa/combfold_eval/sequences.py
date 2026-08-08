"""UniProt full-length sequence provider.

Full-length sequences are needed to (a) assign each structure chain to a UniProt
accession by alignment and (b) compute coverage relative to the complete protein
(not just the residues resolved in the reference crystal). Sequences are read
from the offline CSV first and fetched from the UniProt REST API only if missing,
with on-disk caching so a batch run hits the network at most once per accession.
"""
from __future__ import annotations

import os
import threading
import urllib.request
from typing import Dict, Optional

import pandas as pd


class UniProtSequences:
    def __init__(self, csv_path: Optional[str] = None, cache_dir: Optional[str] = None):
        self._seqs: Dict[str, str] = {}
        self.cache_dir = cache_dir
        # Guards the check-cache -> fetch-REST -> write-cache sequence in get(), so
        # concurrent batch workers requesting the same accession don't race on the
        # same on-disk .fasta cache file (harmless duplicate work at worst, but a
        # lock makes it a non-issue rather than relying on that). One lock PER
        # accession (not one global lock) -- mirrors reference.py's per-pdb_id
        # `_lock_for` pattern -- so a slow/failing REST call for one accession
        # never blocks other workers resolving a different, already-cached
        # accession. `_locks_meta` only guards creation of a new per-accession
        # lock, not the fetch itself.
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_meta = threading.Lock()
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        if csv_path and os.path.exists(csv_path):
            self._load_csv(csv_path)

    def _load_csv(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path)
        df = df.dropna()
        cols = {c.lower(): c for c in df.columns}
        id_col = cols.get("uniprot_id") or cols.get("uniprot") or list(df.columns)[0]
        seq_col = cols.get("sequence") or cols.get("seq") or list(df.columns)[1]
        for acc, seq in zip(df[id_col].astype(str), df[seq_col].astype(str)):
            acc = acc.strip()
            seq = seq.strip().upper().replace("*", "")
            if acc and seq:
                self._seqs[acc] = seq

    def _cache_file(self, acc: str) -> Optional[str]:
        return os.path.join(self.cache_dir, f"{acc}.fasta") if self.cache_dir else None

    def _fetch_rest(self, acc: str) -> Optional[str]:
        cf = self._cache_file(acc)
        if cf and os.path.exists(cf):
            seq = _read_fasta_seq(open(cf).read())
            if seq:
                return seq
        url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                text = r.read().decode()
        except Exception:
            return None
        seq = _read_fasta_seq(text)
        if seq and cf:
            with open(cf, "w") as fh:
                fh.write(text)
        return seq

    def _lock_for(self, acc: str) -> threading.Lock:
        with self._locks_meta:
            lock = self._locks.get(acc)
            if lock is None:
                lock = threading.Lock()
                self._locks[acc] = lock
            return lock

    def get(self, acc: str) -> Optional[str]:
        acc = acc.strip()
        if acc in self._seqs:
            return self._seqs[acc]
        with self._lock_for(acc):
            # re-check: another thread may have fetched it while we waited for the lock
            if acc in self._seqs:
                return self._seqs[acc]
            seq = self._fetch_rest(acc)
            if seq:
                self._seqs[acc] = seq
            return seq


def _read_fasta_seq(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    seq = "".join(ln for ln in lines if ln and not ln.startswith(">"))
    seq = seq.upper().replace("*", "")
    return seq or None
