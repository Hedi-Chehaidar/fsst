from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple


FSST_HASH_PRIME = 2971215073 
FSST_SHIFT = 15

def fsst_hash(w: int) -> int:
    """
    Equivalent to:
      ((w)*PRIME) ^ (((w)*PRIME) >> SHIFT)
    with 64-bit unsigned wraparound.
    """
    x = (w * FSST_HASH_PRIME) & 0xFFFFFFFFFFFFFFFF
    return (x ^ (x >> FSST_SHIFT)) & 0xFFFFFFFFFFFFFFFF



def make_sample(
    lines: List[bytes],
    lengths: List[int] | None = None,
    sample_target: int = 2**14,
    sample_line: int = 512,
    sample_maxsz: int = 2**15,
    seed: int = 4637947,
) -> List[bytes]:
    if lengths is None:
        lengths = [len(x) for x in lines]
    else:
        if len(lengths) != len(lines):
            raise ValueError("lengths must have same length as lines")
        # safety: ensure provided lengths don't exceed actual
        for i, (b, L) in enumerate(zip(lines, lengths)):
            if L > len(b):
                raise ValueError(f"lengths[{i}]={L} exceeds len(lines[{i}])={len(b)}")

    nlines = len(lines)
    if nlines == 0:
        return b"", []

    tot_size = sum(lengths)

    # Case 1: take everything
    if tot_size < sample_target:
        return lines

    # Case 2: random chunk sampling into a contiguous buffer
    max_chunks = nlines + (sample_maxsz // sample_line)
    out = bytearray()
    buf = []
    sample_rnd = fsst_hash(seed)

    # Equivalent stopping condition:
    # while(sampleBuf < sampleLim && sampleLen < sampleLenLim)
    while len(out) < sample_target and len(buf) < max_chunks:
        # choose a non-empty line
        sample_rnd = fsst_hash(sample_rnd)
        linenr = sample_rnd % nlines
        # advance until non-empty (wrapping), as in C++
        while lengths[linenr] == 0:
            linenr += 1
            if linenr == nlines:
                linenr = 0

        # choose a chunk
        Lline = lengths[linenr]
        chunks_count = 1 + ((Lline - 1) // sample_line)
        sample_rnd = fsst_hash(sample_rnd)
        chunk_start = sample_line * (sample_rnd % chunks_count)

        # add the chunk
        take_len = min(Lline - chunk_start, sample_line)
        src = lines[linenr]
        piece = src[chunk_start:chunk_start + take_len]
        out.extend(piece)
        buf.append(piece)

    return buf

