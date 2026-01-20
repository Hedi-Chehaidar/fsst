from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Iterable, Optional, Dict
import heapq
import sampling
import sys

# ------------------------------
# Utilities
# ------------------------------

MAX_REAL_SYMBOLS = 255          # codes 0..254 are available for real symbols; 255 is the escape
ESCAPE_CODE = 255               # reserved escape code
MAX_CODE_SPACE = 256 + MAX_REAL_SYMBOLS  # internal code space that includes bytes(0..255) + real symbols
MAX_SYMBOL_LEN = 8              # FSST uses up to 8-byte symbols

def _is_byteslike(x) -> bool:
    return isinstance(x, (bytes, bytearray, memoryview))

# ------------------------------
# Symbol Table
# ------------------------------

@dataclass
class SymbolTable:
    """
    Internal representation closely following the original paper's pseudocode:

    - 'symbols' is an array indexed in the *internal* code space [0 .. 256 + nSymbols).
      * 0..255     -> the 256 literal bytes
      * 256..      -> real multi-byte symbols we've selected for this table
    - 'nSymbols' is the count of real symbols currently in table (max 255).
    - 'sIndex' is a 257-sized index (0..256) that maps a starting byte to a range
      of *real* symbols in sorted order such that for any first byte b,
      real symbols for that b live at internal codes [sIndex[b], sIndex[b+1]) .
    """
    nSymbols: int = 0
    # For simplicity, store everything in a Python list of bytes objects.
    symbols: List[bytes] = field(default_factory=lambda: [bytes([i]) for i in range(256)] + [b""] * MAX_REAL_SYMBOLS)
    sIndex: List[int] = field(default_factory=lambda: [0] * 257)
    # Working area (sorted real symbols by first byte; length-descending per bucket)
    _sorted_real_syms: List[bytes] = field(default_factory=list, init=False)
    symbol_map: dict[bytes, int] = field(default_factory=dict, init=False)

    def insert(self, s: bytes) -> bool:
        """Insert a real symbol if there's room."""
        if not _is_byteslike(s):
            raise TypeError("Symbol must be bytes-like")
        if len(s) < 1 or len(s) > MAX_SYMBOL_LEN:
            return False  # real symbols must be 1..8 bytes (we already have all single bytes as literals)
        if self.nSymbols >= MAX_REAL_SYMBOLS:
            return False
        self.symbols[256 + self.nSymbols] = bytes(s)
        self.symbol_map[s] = 256 + self.nSymbols
        self.nSymbols += 1
        return True

    def makeIndex(self) -> "SymbolTable":
        """
        Sort real symbols and create sIndex so findLongestSymbol can scan only
        the symbols starting with the first byte of a window.
        Within each starting-byte bucket, sort by length descending to find the longest first.
        """
        real_syms = [self.symbols[256 + i] for i in range(self.nSymbols)]
        # sort by (first_byte, -len, then symbol bytes for tie-breaker stability)
        real_syms.sort(key=lambda s: (s[0], -len(s), s))
        self._sorted_real_syms = real_syms

        # Repack into symbols[256..] in this sorted order
        for i, sym in enumerate(real_syms):
            self.symbols[256 + i] = sym
            self.symbol_map[sym] = 256 + i

        # Build sIndex: for each starting byte b, sIndex[b] is the internal code
        # (i.e., 256 + offset) of the first symbol whose first byte == b
        # and sIndex[256] is the sentinel 256 + nSymbols
        # Initialize all starts to sentinel; then fill
        self.sIndex = [256 + self.nSymbols] * 257

        # We walk through all buckets 0..255 and set the start position where it first appears
        i = 0
        while i < self.nSymbols:
            b = self.symbols[256 + i][0]
            # first time we see b, set sIndex[b] to 256+i
            if self.sIndex[b] == 256 + self.nSymbols:
                self.sIndex[b] = 256 + i
            i += 1

        # Ensure monotonic non-decreasing (fill gaps forward)
        last = 256 + self.nSymbols
        for b in reversed(range(256)):
            if self.sIndex[b] == 256 + self.nSymbols:
                self.sIndex[b] = last
            else:
                last = self.sIndex[b]
        self.sIndex[256] = 256 + self.nSymbols
        return self

    def findLongestSymbol(self, data: bytes, pos: int) -> int:
        """
        Return a code:
         - 0..255 if no real symbol matches (i.e., literal byte)
         - >=256 for a real symbol (code points into self.symbols)
        """
        if pos >= len(data):
            return ESCAPE_CODE  # shouldn't be used; caller must guard end-of-input

        first = data[pos]
        start = self.sIndex[first]
        end = self.sIndex[first + 1] if first + 1 <= 256 else 256 + self.nSymbols

        best_code = first  # literal fallback
        # Scan real symbols that start with 'first', longest first (due to sorting)
        for code in range(start, end):
            sym = self.symbols[code]
            L = len(sym)
            if pos + L <= len(data) and data[pos:pos + L] == sym:
                # Because bucket is sorted by length desc, first match is longest
                return code
        return best_code
    
    def buildDP(self, data: bytes) -> List[int]:
        """
        Build the dp table to compress a given string data with the current symbol table
        with dp[i] = shortest compression for the suffix of the string beginning from position i (0 based)
        Also builds 'opt' table with opt[i] = the code of one symbol 
        from position i that can give the optimal compression for the suffix i...n
        """
        n = len(data)
        self.dp = [0] * (n+1)
        self.opt = [0] * n
        for i in reversed(range(n)):
            self.opt[i] = data[i] # assume that the best code is the escape character
            self.dp[i] = self.dp[i+1] + 2
            chosen = False
            for end in reversed(range(i + 1, min(n, i + 8) + 1)):
                sym = data[i : end]
                L = len(sym)
                '''or not(chosen) and self.dp[i] == 1 + self.dp[i + L]'''
                if (self.dp[i] > 1 + self.dp[i + L] or not(chosen) and self.dp[i] == 1 + self.dp[i + L])  and sym in self.symbol_map.keys():
                    self.dp[i] = 1 + self.dp[i + L]
                    self.opt[i] = self.symbol_map[sym]
                    chosen = True
        return self.dp

    def findBestSymbol(self, data: bytes, pos: int) -> int:
        """
        Return one of the best codes usnig the precomputed dp and opt tables (see buildDP):
         - 0..255 if no real symbol matches (i.e., literal byte)
         - >=256 for a real symbol (code points into self.symbols)
        """
        if pos >= len(data):
            return ESCAPE_CODE  # shouldn't be used; caller must guard end-of-input
        return self.opt[pos]




# ------------------------------
# Table Builder (using 3 counters instead of 2)
# ------------------------------

def compressCount(st: SymbolTable, texts: List[bytes],
                   dp: bool, g: int) -> Tuple[List[int], List[List[int]], dict[Tuple[int,int,int],int]]:
    """
    Returns (count1, count2) where:
      count1[code]++ counts occurrences of a code
      count2[prev][code]++ counts transitions prev -> code
    Codes are in the *internal* code space: 0..255 are literals; 256.. are real symbols.
    """
    C = MAX_CODE_SPACE
    count1 = [0] * C
    count2 = [[0] * C for _ in range(C)]
    count3 = {}

    for text in texts:
        pos = 0
        if not text:
            continue
        if dp:
            st.buildDP(text)
        # start with first code at pos
        code = st.findLongestSymbol(text, pos) if not(dp) else st.findBestSymbol(text, pos)
        prev = -1
        # We'll advance *after* we account for transitions in the loop
        while True:
            # Advance by the length of the current code
            if code >= 256:
                L = len(st.symbols[code])
            else:
                L = 1
            pos += L
            if pos >= len(text):
                # Record the last code occurrence (no following transition)
                count1[code] += 1
                break

            prepre = prev # prepre holds the before before last seen code
            prev = code
            code = st.findLongestSymbol(text, pos) if not(dp) else st.findBestSymbol(text, pos)

            # count frequencies
            count1[prev] += 1

            Lprev = len(st.symbols[prev])
            if Lprev != MAX_SYMBOL_LEN:
                count2[prev][code] += 1

            L = len(st.symbols[prepre]) + len(st.symbols[prev]) if prepre != -1 else MAX_SYMBOL_LEN

            if prepre != -1 and L < MAX_SYMBOL_LEN:
                tup = (prepre, prev, code)
                count3[tup] = count3.get(tup,0) + 1

            # "we also count frequencies for the next byte only" when code is a real symbol
            if code >= 256:
                nextByte = text[pos]
                count1[nextByte] += 1
                #if Lprev != MAX_SYMBOL_LEN:
                count2[prev][nextByte] += 1
                if L < MAX_SYMBOL_LEN:
                    tup = (prepre, prev, nextByte)
                    count3[tup] = count3.get(tup,0) + 1
            

    return count1, count2, count3


def makeTable(prev: SymbolTable, count1: List[int], count2: List[List[int]], 
    count3: dict[Tuple[int,int,int], int]) -> SymbolTable:
    """
    Build a new SymbolTable by choosing the most worthwhile candidates:
      - all single codes (0..255 literals and any existing real symbols)
      - all concatenations of two codes, truncated to 8 bytes
    'gain' heuristic: len(s) * frequency   (as in pseudocode)
    """
    C = 256 + prev.nSymbols

    # Candidate heap (max-heap using negative gains)
    # Each entry: (-gain, candidate_bytes)
    heap: List[Tuple[int, int, int, int]] = []

    # Helper to push a candidate
    def push_cand(code1: int, code2: int, code3: int, gain: int):
        if gain <= 0:
            return
        heapq.heappush(heap, (-gain, code1, code2, code3))

    # 1) singles (all 0..(256 + prev.nSymbols - 1))
    for code1 in range(C):
        s1 = prev.symbols[code1] 
        g1 = len(s1) * count1[code1]
        push_cand(code1, -1, -1, g1)

    # 2) concatenations
    for code1 in range(C):
        s1 = prev.symbols[code1]
        if(len(s1) == MAX_SYMBOL_LEN): 
            continue
        for code2 in range(C):
            s2 = prev.symbols[code2]
            s = (s1 + s2)[:MAX_SYMBOL_LEN]
            g = len(s) * count2[code1][code2]
            push_cand(code1, code2, -1, g)

    # 3) double concatenations
    for (code1, code2, code3), count in count3.items():
        s1 = prev.symbols[code1]
        s2 = prev.symbols[code2]
        s3 = prev.symbols[code3]
        if(len(s1) + len(s2) >= MAX_SYMBOL_LEN) :
            continue
        s = (s1 + s2 + s3)[:MAX_SYMBOL_LEN]
        g = len(s) * count
        push_cand(code1, code2, code3, g)

    # Fill a fresh table with the best unique candidates
    res = SymbolTable()
    seen: set[bytes] = set()  # avoid inserting duplicates
    while res.nSymbols < MAX_REAL_SYMBOLS and heap:
        # g is negative gain
        g, code1, code2, code3 = heapq.heappop(heap)
        if code2 == -1:
            curcnt = count1[code1]
            s = prev.symbols[code1]
        elif code3 == -1:
            curcnt = count2[code1][code2]
            s = (prev.symbols[code1] + prev.symbols[code2])[:MAX_SYMBOL_LEN]
        else:
            curcnt = count3[(code1, code2, code3)]
            s = (prev.symbols[code1] + prev.symbols[code2] + prev.symbols[code3])[:MAX_SYMBOL_LEN]
        

        if g != -curcnt * len(s) :
            continue
        if s in seen:
            continue
        
        seen.add(s)
        res.insert(s)
        if code2 == -1:
            continue
        L1 = len(prev.symbols[code1])
        L2 = len(prev.symbols[code2])
        curcnt /= 2
        if code3 == -1:
            # Update the counts of prefix and suffix symbols (and thus the gains)
            count1[code1] -= curcnt 

            if code1 != code2:
                push_cand(code1, -1, -1, L1 * count1[code1])

            count1[code2] -= curcnt 

            push_cand(code2, -1, -1, L2 * count1[code2])

        else:
            L3 = len(prev.symbols[code3])

            # Update the counts of one code symbols (and thus the gains)
            count1[code1] -= curcnt 
            if code1 != code2 and code1 != code3:
                push_cand(code1, -1, -1, L1 * count1[code1])
            count1[code2] -= curcnt 
            if code2 != code3:
                push_cand(code2, -1, -1, L2 * count1[code2])
            count1[code3] -= curcnt 
            push_cand(code3, -1, -1, L3 * count1[code3])

            #### 2 code symbol pruning
            L23 = min(MAX_SYMBOL_LEN, L2 + L3)

            count2[code1][code2] -= curcnt 
            if code1 != code2 or code2 != code3:
                push_cand(code1, code2, -1, (L1 + L2) * count2[code1][code2])
            count2[code2][code3] -= curcnt 
            push_cand(code2, code3, -1, L23 * count2[code2][code3])
            

    return res.makeIndex()


def buildSymbolTable(texts: List[bytes], generations: int = 5, dp: bool = False) -> SymbolTable:
    """
    Top-level entry point for building a table from training text.
    Runs multiple generations, updating counts and re-selecting symbols.
    """
    st = SymbolTable().makeIndex()
    for g in range(generations):
        count1, count2, count3 = compressCount(st, texts, dp, g)
        st = makeTable(st, count1, count2, count3)
    return st

# ------------------------------
# Encoder / Decoder
# ------------------------------



def encode(data: bytes, st: SymbolTable, dp: bool) -> bytes:
    """
    Emit a byte stream:
      - If a real symbol (internal code >= 256) matches, write code' = code-256 (0..254).
      - Otherwise, write ESCAPE_CODE(=255) followed by the literal byte.
    """
    out = bytearray()
    pos = 0
    n = len(data)

    if dp:
        st.buildDP(data)

    while pos < n:
        code = st.findLongestSymbol(data, pos) if not(dp) else st.findBestSymbol(data, pos)
        if code <= 255:
            # literal fallback: escape + byte
            out.append(ESCAPE_CODE)
            out.append(code)  # code is the literal byte value itself
            pos += 1
        else:
            # real symbol
            out.append(code - 256)  # external code space 0..254
            pos += len(st.symbols[code])
    return bytes(out)

def decode(encoded: bytes, st: SymbolTable) -> bytes:
    """
    Inverse of 'encode' using Algorithm 1:
      - if byte != 255: treat as a symbol code in 0..254 and expand
      - if byte == 255: read next byte literally
    """
    inb = memoryview(encoded)
    i = 0
    out = bytearray()
    sym: List[bytes] = [b""] * MAX_REAL_SYMBOLS
    ln: List[int] = [0] * MAX_REAL_SYMBOLS
    for k in range(MAX_REAL_SYMBOLS):
        if k < st.nSymbols:
            s = st.symbols[256 + k]
            sym[k] = s
            ln[k] = len(s)
        else:
            sym[k] = b""
            ln[k] = 0

    L = len(inb)
    while i < L:
        code = inb[i]
        i += 1
        if code != ESCAPE_CODE:
            # expand symbol
            if code >= st.nSymbols:
                raise ValueError(f"Decoding error: symbol code {code} >= nSymbols {st.nSymbols}")
            out.extend(sym[code])
        else:
            if i >= L:
                raise ValueError("Decoding error: escape at end of stream")
            out.append(inb[i])
            i += 1
    return bytes(out)

# ------------------------------
# Demo
# ------------------------------

def demo(file):
    sum_cf = 0
    # construct a sample 
    with open(file) as f: 
        sample = [s.encode() for s in f.readlines()]
        sample = sampling.make_sample(sample)
    # compress sample with fsst 
    st = buildSymbolTable(sample, generations=5, dp=True)
    enc = [encode(data, st, True) for data in sample]
    '''decoded = [decode(s, st) for s in enc]
    if(decoded != sample) :
        print("wrong encoding")'''
    orsz = sum([len(s) for s in sample])
    cf = round(orsz/sum([len(s) for s in enc]),3)
    sum_cf += cf
    print(cf)



# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    demo(sys.argv[1])
