from __future__ import annotations

from pathlib import Path, PurePath
import os
import duckdb
import subprocess
import traceback
import tempfile
import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Tuple, Optional

# ---------------- Config ----------------
# Point this to your FSST binary (or export FSST_BINARY=/path/to/fsst)
FSST_BINARY = os.environ.get("FSST_BINARY", "../build/fsst")

RAW_DIR = "./raw/ClickBench"
OUTPUT_DIR = "./refined/ClickBench"
SAMPLE_ROWS = 100_000
AVG_LEN_THRESHOLD = 8
FSST_VS_DICT_FACTOR = 2.0   # keep if fsst_bytes <= dict_bytes * factor
MAX_WORKERS_CAP = 32
# ----------------------------------------

import bz2
import shutil
import tempfile

def _maybe_decompress_csv(path: Path) -> tuple[Path, bool]:
    """
    If path ends with .bz2, decompress to a temp .csv file and return it.
    Returns (new_path, is_temp).
    """
    if not path.name.lower().endswith(".bz2"):
        return path, False

    tmp_dir = tempfile.mkdtemp(prefix="bz2csv_")
    out_path = Path(tmp_dir) / path.name[:-4]  # strip .bz2

    with bz2.open(path, "rb") as fin, open(out_path, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=1024 * 1024)


    def cleanup():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_path, cleanup


def _export_column_to_text(con: duckdb.DuckDBPyConnection,
                           rel: duckdb.DuckDBPyRelation,
                           col: str,
                           out_txt_path: Path) -> None:
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)

    con.register("rel_export", rel)
    try:
        qcol = _quote_ident(col)
        out_sql_path = str(out_txt_path).replace("'", "''")

        sql = f"""
        COPY (
            SELECT replace(replace(CAST({qcol} AS VARCHAR), '\r', ' '), '\n', ' ')
            FROM rel_export
            WHERE {qcol} IS NOT NULL
        )
        TO '{out_sql_path}'
        (HEADER FALSE, DELIMITER '\n');
        """
        con.execute(sql)
    finally:
        try:
            con.unregister("rel_export")
        except Exception:
            pass


def load_nextiajd_metadata(meta_csv_path: str) -> dict[str, dict]:
    m: dict[str, dict] = {}

    # utf-8-sig strips BOM if present
    with open(meta_csv_path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        print("Metadata headers:", r.fieldnames)  # keep once to verify

        for row in r:
            fn = row.get("filename")
            if not fn:
                continue

            fn = fn.strip()

            delim = row.get("delimiter", ",").strip()
            if len(delim) >= 2 and delim[0] == delim[-1] == '"':
                delim = delim[1:-1]
            if delim == r"\t":
                delim = "\t"

            multiline = row.get("multiline", "").strip().upper() == "TRUE"
            nullval = row.get("nullVal", "")

            m[fn] = {
                "delimiter": delim,
                "multiline": multiline,
                "nullVal": nullval,
            }

    return m

#NEXTIAJD_META = load_nextiajd_metadata("./raw/NextiaJD/metadata.csv")


def _quote_ident(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


def _base_name_for(relative_path: Path) -> str:
    n = relative_path.name
    ln = n.lower()
    if ln.endswith(".csv.bz2"):
        return n[:-8]          # strip .csv.bz2
    if ln.endswith(".csv"):
        return n[:-4]          # strip .csv
    if ln.endswith(".parquet"):
        return n[:-8]          # strip .parquet
    return relative_path.stem



def _safe_filename(s: str) -> str:
    # keep it simple and filesystem-safe
    s = s.strip()
    s = re.sub(r"[^\w.\-]+", "_", s)  # replace weird chars with _
    return s[:200] if len(s) > 200 else s


def _read_relation(con: duckdb.DuckDBPyConnection, input_path: PurePath) -> duckdb.DuckDBPyRelation:
    name = input_path.name.lower()
    real_path, cleanup = _maybe_decompress_csv(Path(input_path))
    p = str(real_path)

    if name.endswith(".parquet"):
        return con.read_parquet(p), cleanup

    if name.endswith(".csv") or name.endswith(".csv.bz2"):
        try:
            if "nextiajd" in str(input_path).lower():
                return _read_nextiajd_csv(con, real_path), cleanup
                
            return con.from_csv_auto(p, header=True, ignore_errors=True), cleanup
        except Exception:
            return con.from_csv_auto(
                p,
                header=True,
                ignore_errors=True,
                strict_mode=False,
                parallel=False,
                max_line_size=10_000_000,
            ), cleanup

    raise ValueError(f"Unsupported file format: {input_path}")



def _read_nextiajd_csv(con, input_path):
    p = str(input_path)
    key = input_path.name[:-4] if input_path.name.endswith(".bz2") else input_path.name
    info = NEXTIAJD_META.get(key)
    if info is None:
        raise RuntimeError(f"NextiaJD metadata missing for {key}")

    delimiter = info["delimiter"]
    multiline = bool(info.get("multiline", False))

    kwargs = dict(
        header=True,
        delimiter=delimiter,
        quotechar='"',
        strict_mode=False,
        ignore_errors=True,
        parallel=False,
        all_varchar=True,
        max_line_size=50_000_000,
    )

    # ✅ force bz2 decompression
    if str(input_path).lower().endswith(".bz2"):
        kwargs["compression"] = "bz2"

    # multiline option name varies; try none first
    try:
        return con.read_csv(p, **kwargs)
    except Exception:
        # try allowing quoted newlines if your build supports it
        for opt in ["allow_quoted_newlines", "quoted_newlines"]:
            try:
                return con.read_csv(p, **kwargs, **{opt: multiline})
            except Exception:
                pass
        raise




def _avg_len_and_nonnull(rel: duckdb.DuckDBPyRelation, col: str) -> tuple[float, int]:
    qcol = _quote_ident(col)
    row = rel.aggregate(
        f"avg(length(CAST({qcol} AS VARCHAR))) AS a, count({qcol}) AS nn"
    ).fetchone()
    avg_len = float(row[0]) if row and row[0] is not None else 0.0
    non_null = int(row[1]) if row and row[1] is not None else 0
    return avg_len, non_null



def _duckdb_dict_size(con: duckdb.DuckDBPyConnection, sample_rel: duckdb.DuckDBPyRelation, col: str) -> Optional[int]:
    """
    Estimate dictionary-like compressed size on the sample:
      dict_bytes + codes_bytes
    """
    con.register("rel", sample_rel)
    qcol = _quote_ident(col)

    query = f"""
    SELECT
      CAST(
        COALESCE(length(string_agg(DISTINCT {qcol}, '')), 0)
        +
        (COUNT({qcol}) * CASE
            WHEN COUNT(DISTINCT {qcol}) = 0 THEN 0
            ELSE ceil(log2(COUNT(DISTINCT {qcol})) / 8)
          END
        )
      AS BIGINT) AS total_compressed_size
    FROM rel
    """
    try:
        row = con.execute(query).fetchone()
        if not row:
            return None
        return int(row[0]) if row[0] is not None else None
    except Exception:
        return None
    finally:
        try:
            con.unregister("rel")
        except Exception:
            pass


def _write_column_sample_to_file(con, sample_rel, col: str, out_txt_path: Path) -> int:
    con.register("rel2", sample_rel)
    qcol = _quote_ident(col)

    # one value per line; no quoting; replace newlines to keep 1 record = 1 line
    sql = f"""
    COPY (
        SELECT replace(replace(CAST({qcol} AS VARCHAR), '\r', ' '), '\n', ' ')
        FROM rel2
        WHERE {qcol} IS NOT NULL
    )
    TO '{str(out_txt_path).replace("'", "''")}'
    (HEADER FALSE, DELIMITER '\n', QUOTE '', ESCAPE '');
    """
    con.execute(sql)
    con.unregister("rel2")
    return out_txt_path.stat().st_size



def _run_fsst_file(binary: str, input_file: Path, output_file: Path) -> Tuple[Optional[int], str, str, int]:
    """
    Run FSST compressor binary that accepts (input_file, output_file).
    Return (output_bytes, stdout, stderr, returncode).
    """
    cmd = [binary, str(input_file), str(output_file)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        return None, "", str(e), 127
    except Exception as e:
        return None, "", str(e), 1

    if res.returncode != 0:
        return None, res.stdout or "", res.stderr or "", int(res.returncode)

    # Prefer actual output size on disk (avoids ambiguity of "compression factor")
    try:
        out_bytes = output_file.stat().st_size
    except Exception:
        out_bytes = None

    return out_bytes, res.stdout or "", res.stderr or "", int(res.returncode)


def refine_dataset(input_path: PurePath, output_dir: PurePath) -> bool:
    print(f"\n\n ⚗️ ⚗️  Refining {input_path} -> {output_dir} ⚗️ ⚗️")
    #if(input_path.name not in DATASETS) :
    #    return False
    if not FSST_BINARY or not Path(FSST_BINARY).exists():
        print(f"🚨 FSST binary not found: {FSST_BINARY}")
        return False

    con = duckdb.connect()
    try:
        try:
            relation, cleanup = _read_relation(con, input_path)
            #relation, safe_to_orig = _make_safe_relation(con, relation)


        except Exception as e:
            print(f"⚠️ Failed to read {input_path}: {e}")
            return False

        # Use sample for stats / comparisons
        sample_rel = relation.limit(SAMPLE_ROWS)

        sample_count = sample_rel.aggregate("count(*)").fetchone()
        if not sample_count or sample_count[0] == 0:
            print("⏭️ Empty dataset sample, skipping.")
            return False
        

        columns = sample_rel.columns
        types = sample_rel.types
        col_types_map = dict(zip(columns, types))

        text_columns: List[str] = []
        for col in columns:
            col_type_str = str(col_types_map[col]).upper()

            if col_type_str != "VARCHAR":
                print(f"column of type {col_type_str}")
                continue
            avg_len, non_null = _avg_len_and_nonnull(sample_rel, col)
            if non_null == 0:
                print("ALL NULL:", input_path, col)
                continue

            if avg_len <= AVG_LEN_THRESHOLD:
                print(f"(avg_len={avg_len:.2f})")
                continue

            # Estimate dict size on sample
            dict_size = _duckdb_dict_size(con, sample_rel, col)
            if dict_size is None:
                print(f"⚠️ Could not compute dict estimate for '{col}', skipping column.")
                continue

            # Materialize column sample -> FSST compress -> get output size
            print(f"Running FSST on sampled values of column '{col}' (avg_len={avg_len:.2f})")

            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                tmp_in = td_path / "col_sample.txt"
                tmp_out = td_path / "col_sample.fsst"

                in_bytes = _write_column_sample_to_file(con, sample_rel, col, tmp_in)

                if in_bytes == 0:
                    print(f"⏭️ Column '{col}' sample wrote 0 bytes, skipping column.")
                    continue

                out_bytes, out, err, rc = _run_fsst_file(FSST_BINARY, tmp_in, tmp_out)
                if rc != 0 or out_bytes is None:
                    print(f"🚨 FSST failed for '{col}' (rc={rc}). stderr:\n{err[:1200]}")
                    continue

                
                text_columns.append(col)
                # stdout is a "compression factor" per your binary; log it for info
                factor_str = out.strip().splitlines()[0] if out.strip() else "(no factor printed)"
                print(f"👍 Include '{col}': fsst_bytes={out_bytes} <= {FSST_VS_DICT_FACTOR}x dict_bytes={dict_size} | factor={factor_str}")
                # Compare FSST output size vs dict estimate
                '''if out_bytes <= dict_size * FSST_VS_DICT_FACTOR:
                    text_columns.append(col)
                    # stdout is a "compression factor" per your binary; log it for info
                    factor_str = out.strip().splitlines()[0] if out.strip() else "(no factor printed)"
                    print(f"👍 Include '{col}': fsst_bytes={out_bytes} <= {FSST_VS_DICT_FACTOR}x dict_bytes={dict_size} | factor={factor_str}")
                else:
                    print(f"👎 Exclude '{col}': fsst_bytes={out_bytes} > {FSST_VS_DICT_FACTOR}x dict_bytes={dict_size}")'''

        if not text_columns:
            print(f"⏭️ No columns matched criteria in {input_path}. Skipping file.")
            return False

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)


        # Create output folder for this dataset
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for col in text_columns:
            orig_name = col #safe_to_orig.get(col, col)  # if you kept the safe rename mapping
            fname = _safe_filename(orig_name) + ".txt"
            out_txt = output_dir / fname

            print(f"📝 Writing column {orig_name} -> {out_txt}")
            _export_column_to_text(con, sample_rel, col, out_txt)

        print(f"✅ Exported {len(text_columns)} columns to {output_dir}")
        return True


    finally:
        con.close()
        #cleanup()


def process_raw_directory(raw_dir: str = RAW_DIR, output_dir: str = OUTPUT_DIR):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    supported_files: List[Path] = []
    supported_files.extend(raw_path.glob("**/*.parquet"))
    supported_files.extend(raw_path.glob("**/*.csv"))
    supported_files.extend(raw_path.glob("**/*.csv.bz2"))

    files_to_process: List[Tuple[Path, Path]] = []
    for data_file in supported_files:
        rel = data_file.relative_to(raw_path)
        base = _base_name_for(rel)
        out_dir = out_path / rel.parent / base  # folder per input file

        # simple “already processed” check: folder exists and has at least 1 txt
        if out_dir.exists() and any(out_dir.glob("*.txt")):
            print(f"✅ Already refined {data_file} -> {out_dir}")
            continue

        files_to_process.append((data_file, out_dir))


    if not files_to_process:
        print("No files to process (nothing matched or all outputs already exist).")
        return

    max_workers = min(4, os.cpu_count() or 2)

    print(f"🚀 Starting parallel processing with up to {max_workers} threads for {len(files_to_process)} files...")

    def worker(pair: Tuple[Path, Path]) -> bool:
        data_file, out_dir = pair

        try:
            return refine_dataset(data_file, out_dir)
        except Exception as e:
            print(f"🚨🚨 UNHANDLED ERROR processing {data_file}: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            return False

    results: List[bool] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_pair = {ex.submit(worker, p): p for p in files_to_process}
        for i, fut in enumerate(as_completed(fut_to_pair)):
            data_file, _ = fut_to_pair[fut]
            try:
                ok = bool(fut.result())
            except Exception as exc:
                print(f"‼️ Exception for {data_file}: {exc}")
                ok = False
            results.append(ok)
            #print(f"🏁 Completed {i + 1}/{len(files_to_process)}: {data_file} -> {'Success' if ok else 'Skipped'}")

    processed = sum(results)
    skipped = len(files_to_process) - processed

    print(f"\n🌐 Macrodata Refinement Complete 🌐")
    print(f"🕒 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processed {len(files_to_process)} files:")
    print(f"  ✅ {processed} files refined.")
    print(f"  ⏭️ {skipped} files skipped (no compatible columns or errors).")


if __name__ == "__main__":
    process_raw_directory()
