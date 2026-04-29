import argparse
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_DIR = Path("/app/Data") if Path("/app/Data").exists() else Path(__file__).parent.parent / "Data"
ALFRESCO_DIR = BASE_DIR / "Alfresco"
AWS_DIR = BASE_DIR / "Aws"
OUTPUT_DIR = BASE_DIR / "Output"
TMP_DIR = BASE_DIR / "tmp"
S3_FILE = AWS_DIR / "lista_s3.csv"
CHUNK_SIZE = 200_000
MAX_TABLE_ROWS = 1_000

ALF_COLS = ["Nome", "Path", "Dimensione (bytes)", "MIME Type"]
S3_COLS = ["NomeFile", "Path", "Dimensione", "Data"]


@dataclass
class AlfrescoResult:
    stem: str
    n_alfresco: int
    n_comuni: int
    df_solo_alf: pd.DataFrame


@dataclass
class AlfrescoSizeResult:
    stem: str
    n_alfresco: int
    n_in_s3: int
    n_solo_alf: int
    size_in_s3_bytes: float
    size_solo_alf_bytes: float
    df_solo_alf: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Step 1 – Split S3 into chunks
# ---------------------------------------------------------------------------

def split_s3_into_chunks() -> tuple[list[Path], int]:
    print(f"\n[STEP 1] Split di {S3_FILE.name} in chunk da {CHUNK_SIZE:,} righe...")
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    chunk_paths: list[Path] = []
    total_rows = 0
    for i, chunk in enumerate(tqdm(
        pd.read_csv(S3_FILE, chunksize=CHUNK_SIZE, dtype=str, low_memory=False),
        desc="  Chunk S3", unit="chunk",
    )):
        path = TMP_DIR / f"s3_chunk_{i:04d}.csv"
        chunk.to_csv(path, index=False)
        chunk_paths.append(path)
        total_rows += len(chunk)

    print(f"  → {len(chunk_paths)} chunk creati, {total_rows:,} righe totali in S3")
    return chunk_paths, total_rows


# ---------------------------------------------------------------------------
# Step 2 – Load all Alfresco files
# ---------------------------------------------------------------------------

def load_alfresco_files(alfresco_files: list[Path]) -> dict[Path, pd.DataFrame]:
    print(f"\n[STEP 2] Caricamento {len(alfresco_files)} file Alfresco...")
    dfs: dict[Path, pd.DataFrame] = {}
    for path in alfresco_files:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        dfs[path] = df
        print(f"  • {path.name}: {len(df):,} righe")
    return dfs


# ---------------------------------------------------------------------------
# Step 3 – Per-file Alfresco → S3 analysis (returns result, no I/O)
# ---------------------------------------------------------------------------

def analyze_alfresco_to_s3(
    alfresco_path: Path,
    df_alf: pd.DataFrame,
    chunk_paths: list[Path],
) -> AlfrescoResult:
    print(f"\n  {alfresco_path.name}")
    alf_names = set(df_alf["Nome"].dropna())
    remaining = set(alf_names)

    for chunk_path in tqdm(chunk_paths, desc="    Chunk", unit="chunk", leave=False):
        chunk = pd.read_csv(chunk_path, dtype=str, low_memory=False)
        remaining -= set(chunk["NomeFile"].dropna()) & remaining
        if not remaining:
            break

    df_solo_alf = df_alf[df_alf["Nome"].isin(remaining)].copy()
    n_comuni = len(alf_names) - len(remaining)
    pct = len(remaining) / len(df_alf) * 100 if df_alf is not None and len(df_alf) else 0
    print(f"    → Comuni: {n_comuni:,} | Solo Alfresco: {len(remaining):,} ({pct:.1f}%)")

    return AlfrescoResult(
        stem=alfresco_path.stem,
        n_alfresco=len(df_alf),
        n_comuni=n_comuni,
        df_solo_alf=df_solo_alf,
    )


# ---------------------------------------------------------------------------
# Step 4 – Global S3 → all Alfresco analysis
# ---------------------------------------------------------------------------

def analyze_s3_to_alfresco(
    chunk_paths: list[Path],
    all_alfresco_names: set[str],
    n_s3_total: int,
    n_alf_files: int,
    n_alf_total: int,
    timestamp: str,
) -> None:
    print(f"\n[STEP 4] Analisi globale S3 → Alfresco ({n_alf_files} file, {len(all_alfresco_names):,} nomi unici)...")

    parts: list[pd.DataFrame] = []
    for chunk_path in tqdm(chunk_paths, desc="  Chunk S3", unit="chunk"):
        chunk = pd.read_csv(chunk_path, dtype=str, low_memory=False)
        mask = ~chunk["NomeFile"].isin(all_alfresco_names)
        if mask.any():
            parts.append(chunk.loc[mask].copy())

    df_solo_s3 = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    n_solo_s3 = len(df_solo_s3)
    pct = n_solo_s3 / n_s3_total * 100 if n_s3_total else 0
    print(f"  → Solo S3: {n_solo_s3:,} ({pct:.1f}%)")

    _write_s3_report(
        timestamp=timestamp,
        n_alf_files=n_alf_files,
        n_alf_total=n_alf_total,
        n_alf_unique=len(all_alfresco_names),
        n_s3_total=n_s3_total,
        n_solo_s3=n_solo_s3,
        pct_solo_s3=pct,
        df_solo_s3=df_solo_s3,
    )


# ---------------------------------------------------------------------------
# diff-size: per-file Alfresco → S3 with size breakdown
# ---------------------------------------------------------------------------

def analyze_alfresco_to_s3_with_size(
    alfresco_path: Path,
    df_alf: pd.DataFrame,
    chunk_paths: list[Path],
) -> AlfrescoSizeResult:
    print(f"\n  {alfresco_path.name}")
    alf_names = set(df_alf["Nome"].dropna())
    remaining = set(alf_names)

    for chunk_path in tqdm(chunk_paths, desc="    Chunk", unit="chunk", leave=False):
        chunk = pd.read_csv(chunk_path, dtype=str, low_memory=False)
        remaining -= set(chunk["NomeFile"].dropna()) & remaining
        if not remaining:
            break

    in_s3_names = alf_names - remaining
    df_in_s3 = df_alf[df_alf["Nome"].isin(in_s3_names)]
    df_solo_alf = df_alf[df_alf["Nome"].isin(remaining)].copy()

    size_col = "Dimensione (bytes)"
    size_in_s3 = pd.to_numeric(df_in_s3[size_col], errors="coerce").sum() if size_col in df_alf.columns else 0.0
    size_solo = pd.to_numeric(df_solo_alf[size_col], errors="coerce").sum() if size_col in df_alf.columns else 0.0

    n_in_s3 = len(in_s3_names)
    n_solo = len(remaining)
    pct = n_solo / len(df_alf) * 100 if len(df_alf) else 0
    print(f"    → In S3: {n_in_s3:,} ({_fmt_bytes(size_in_s3)}) | Solo Alfresco: {n_solo:,} ({pct:.1f}%, {_fmt_bytes(size_solo)})")

    return AlfrescoSizeResult(
        stem=alfresco_path.stem,
        n_alfresco=len(df_alf),
        n_in_s3=n_in_s3,
        n_solo_alf=n_solo,
        size_in_s3_bytes=float(size_in_s3),
        size_solo_alf_bytes=float(size_solo),
        df_solo_alf=df_solo_alf,
    )


def analyze_s3_with_size(
    chunk_paths: list[Path],
    all_alfresco_names: set[str],
    n_s3_total: int,
) -> tuple[int, float, float]:
    """
    Returns (n_solo_s3, size_solo_s3_bytes, size_libretto_bytes).

    - solo S3: items in S3 not found in any Alfresco file
    - libretto: all S3 items whose Path contains the folder 'libretto'
    """
    print(f"\n[STEP S3-SIZE] Analisi S3 con pesi ({len(all_alfresco_names):,} nomi Alfresco unici)...")

    n_solo_s3 = 0
    size_solo_s3 = 0.0
    size_libretto = 0.0

    for chunk_path in tqdm(chunk_paths, desc="  Chunk S3", unit="chunk"):
        chunk = pd.read_csv(chunk_path, dtype=str, low_memory=False)
        dim = pd.to_numeric(chunk["Dimensione"], errors="coerce")

        mask_solo = ~chunk["NomeFile"].isin(all_alfresco_names)
        n_solo_s3 += int(mask_solo.sum())
        size_solo_s3 += float(dim[mask_solo].sum())

        mask_libretto = chunk["Path"].str.contains("libretto", case=False, na=False)
        size_libretto += float(dim[mask_libretto].sum())

    pct = n_solo_s3 / n_s3_total * 100 if n_s3_total else 0
    print(f"  → Solo S3: {n_solo_s3:,} ({pct:.1f}%, {_fmt_bytes(size_solo_s3)})")
    print(f"  → Libretto (path): {_fmt_bytes(size_libretto)}")

    return n_solo_s3, size_solo_s3, size_libretto


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _fmt_bytes(b: float) -> str:
    gb = b / 1024**3
    return f"{gb:,.3f} GB"


def _df_to_md_table(df: pd.DataFrame, cols: list[str], max_rows: int) -> str:
    available = [c for c in cols if c in df.columns]
    subset = df[available].head(max_rows)
    header = "| " + " | ".join(available) + " |"
    sep = "| " + " | ".join(["---"] * len(available)) + " |"
    rows = "\n".join(
        "| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |"
        for row in subset.itertuples(index=False)
    )
    return f"{header}\n{sep}\n{rows}"


def _write_alfresco_report(
    results: list[AlfrescoResult],
    n_s3_total: int,
    timestamp: str,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"report_alfresco_vs_s3_{timestamp}.md"

    n_alf_total = sum(r.n_alfresco for r in results)
    n_comuni_total = sum(r.n_comuni for r in results)
    n_solo_total = sum(len(r.df_solo_alf) for r in results)

    lines = [
        "# Report Alfresco → S3",
        "",
        f"**Data elaborazione:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**File Alfresco analizzati:** {len(results)}  ",
        "**File S3:** `lista_s3.csv`",
        "",
        "---",
        "",
        "## Riepilogo Globale",
        "",
        "| File Alfresco | Righe | Presenti in S3 | Mancanti in S3 | % Mancanti |",
        "| --- | --- | --- | --- | --- |",
    ]

    for r in results:
        n_solo = len(r.df_solo_alf)
        pct = n_solo / r.n_alfresco * 100 if r.n_alfresco else 0
        lines.append(f"| `{r.stem}.csv` | {r.n_alfresco:,} | {r.n_comuni:,} | {n_solo:,} | {pct:.2f}% |")

    pct_tot = n_solo_total / n_alf_total * 100 if n_alf_total else 0
    lines += [
        f"| **Totale** | **{n_alf_total:,}** | **{n_comuni_total:,}** | **{n_solo_total:,}** | **{pct_tot:.2f}%** |",
        "",
        f"> File totali in S3: **{n_s3_total:,}**",
        "",
        "---",
    ]

    for r in results:
        n_solo = len(r.df_solo_alf)
        pct = n_solo / r.n_alfresco * 100 if r.n_alfresco else 0
        lines += [
            "",
            f"## {r.stem}.csv",
            "",
            f"**Righe totali:** {r.n_alfresco:,} | **Mancanti in S3:** {n_solo:,} ({pct:.2f}%)",
            "",
        ]

        if n_solo == 0:
            lines.append("_Nessun file mancante: tutti presenti in S3._")
        else:
            truncated = n_solo > MAX_TABLE_ROWS
            if truncated:
                csv_path = OUTPUT_DIR / f"solo_alfresco_{r.stem}_{timestamp}.csv"
                r.df_solo_alf.to_csv(csv_path, index=False)
                lines.append(
                    f"> **Nota:** tabella troncata ai primi {MAX_TABLE_ROWS:,} risultati. "
                    f"Lista completa in `solo_alfresco_{r.stem}_{timestamp}.csv`."
                )
                lines.append("")
            lines.append(_df_to_md_table(r.df_solo_alf, ALF_COLS, MAX_TABLE_ROWS))

        lines.append("")
        lines.append("---")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  → Report Alfresco→S3: {report_path.name}")


def _write_s3_report(
    *,
    timestamp: str,
    n_alf_files: int,
    n_alf_total: int,
    n_alf_unique: int,
    n_s3_total: int,
    n_solo_s3: int,
    pct_solo_s3: float,
    df_solo_s3: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"report_s3_vs_alfresco_{timestamp}.md"

    n_in_alf = n_s3_total - n_solo_s3
    pct_in_alf = n_in_alf / n_s3_total * 100 if n_s3_total else 0

    truncated = len(df_solo_s3) > MAX_TABLE_ROWS
    if truncated:
        csv_path = OUTPUT_DIR / f"solo_s3_{timestamp}.csv"
        df_solo_s3.to_csv(csv_path, index=False)

    lines = [
        "# Report S3 → Alfresco (analisi globale)",
        "",
        f"**Data elaborazione:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**File Alfresco analizzati:** {n_alf_files}  ",
        "**File S3:** `lista_s3.csv`",
        "",
        "---",
        "",
        "## Riepilogo",
        "",
        "| Metrica | Valore | % |",
        "| --- | --- | --- |",
        f"| File totali in S3 | {n_s3_total:,} | — |",
        f"| Righe totali Alfresco (tutti i file) | {n_alf_total:,} | — |",
        f"| Nomi unici in Alfresco (union) | {n_alf_unique:,} | — |",
        f"| Presenti in almeno un file Alfresco | {n_in_alf:,} | {pct_in_alf:.2f}% di S3 |",
        f"| In S3 ma **NON** in nessun Alfresco | {n_solo_s3:,} | {pct_solo_s3:.2f}% di S3 |",
        "",
        "---",
        "",
        "## File in S3 ma assenti in tutti i file Alfresco",
        "",
        f"**Totale:** {n_solo_s3:,}",
    ]

    if n_solo_s3 == 0:
        lines.append("\n_Nessun file mancante: tutti i file S3 sono presenti in almeno un file Alfresco._")
    else:
        if truncated:
            lines.append(
                f"\n> **Nota:** tabella troncata ai primi {MAX_TABLE_ROWS:,} risultati. "
                f"Lista completa in `solo_s3_{timestamp}.csv`."
            )
        lines.append("")
        lines.append(_df_to_md_table(df_solo_s3, S3_COLS, MAX_TABLE_ROWS))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → Report S3→Alfresco: {report_path.name}")


def _write_diff_size_report(
    *,
    timestamp: str,
    alf_results: list[AlfrescoSizeResult],
    n_s3_total: int,
    n_solo_s3: int,
    size_solo_s3_bytes: float,
    size_libretto_bytes: float,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"report_diff_size_{timestamp}.md"

    total_alf_items = sum(r.n_alfresco for r in alf_results)
    total_in_s3_items = sum(r.n_in_s3 for r in alf_results)
    total_solo_alf_items = sum(r.n_solo_alf for r in alf_results)
    total_size_in_s3 = sum(r.size_in_s3_bytes for r in alf_results)
    total_size_solo_alf = sum(r.size_solo_alf_bytes for r in alf_results)
    total_alf_size = total_size_in_s3 + total_size_solo_alf

    # Grand total = all Alfresco items + S3-only items (no double-counting)
    grand_total_items = total_alf_items + n_solo_s3
    grand_total_size = total_alf_size + size_solo_s3_bytes

    lines = [
        "# Report Diff con Dimensioni (Alfresco ↔ S3)",
        "",
        f"**Data elaborazione:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**File Alfresco analizzati:** {len(alf_results)}  ",
        "**File S3:** `lista_s3.csv`",
        "",
        "---",
        "",
        "## 1. Alfresco → S3 (per file)",
        "",
        "| File Alfresco | Totale | In S3 | Dimensione In S3 | Solo Alfresco | Dimensione Solo Alf |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for r in alf_results:
        lines.append(
            f"| `{r.stem}.csv` | {r.n_alfresco:,} | {r.n_in_s3:,} | {_fmt_bytes(r.size_in_s3_bytes)} "
            f"| {r.n_solo_alf:,} | {_fmt_bytes(r.size_solo_alf_bytes)} |"
        )

    lines += [
        f"| **Totale** | **{total_alf_items:,}** | **{total_in_s3_items:,}** | **{_fmt_bytes(total_size_in_s3)}** "
        f"| **{total_solo_alf_items:,}** | **{_fmt_bytes(total_size_solo_alf)}** |",
        "",
        f"> **Dimensione totale Alfresco** (presenti + assenti in S3): **{_fmt_bytes(total_alf_size)}**",
        "",
        "---",
        "",
        "## 2. S3 → Alfresco",
        "",
        "| Metrica | Conteggio | Dimensione |",
        "| --- | --- | --- |",
        f"| Totale item S3 | {n_s3_total:,} | — |",
        f"| Presenti in almeno un file Alfresco | {n_s3_total - n_solo_s3:,} | — |",
        f"| Solo S3 (non in nessun Alfresco) | {n_solo_s3:,} | {_fmt_bytes(size_solo_s3_bytes)} |",
        f"| Path contiene cartella **libretto** | — | {_fmt_bytes(size_libretto_bytes)} |",
        "",
        "---",
        "",
        "## 3. Riepilogo Globale",
        "",
        "| Categoria | Item | Dimensione |",
        "| --- | --- | --- |",
        f"| Alfresco (tutti) | {total_alf_items:,} | {_fmt_bytes(total_alf_size)} |",
        f"| Solo S3 (non in Alfresco) | {n_solo_s3:,} | {_fmt_bytes(size_solo_s3_bytes)} |",
        f"| **Totale complessivo** | **{grand_total_items:,}** | **{_fmt_bytes(grand_total_size)}** |",
        f"| *(di cui libretto in S3)* | *—* | *{_fmt_bytes(size_libretto_bytes)}* |",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  → Report Diff+Size: {report_path.name}")


# ---------------------------------------------------------------------------
# Size analysis
# ---------------------------------------------------------------------------

def calc_alfresco_size() -> float:
    """Sum Dimensione (bytes) across all Alfresco CSVs. Returns total in bytes."""
    alfresco_files = sorted(ALFRESCO_DIR.glob("*.csv"))
    if not alfresco_files:
        print(f"Nessun file CSV trovato in {ALFRESCO_DIR}")
        return 0.0

    print(f"\n[SIZE] Calcolo dimensione Alfresco ({len(alfresco_files)} file)...")
    total_bytes = 0.0
    for path in alfresco_files:
        df = pd.read_csv(path, usecols=["Dimensione (bytes)"], dtype=str, low_memory=False)
        col = pd.to_numeric(df["Dimensione (bytes)"], errors="coerce")
        file_bytes = col.sum()
        total_bytes += file_bytes
        print(f"  • {path.name}: {file_bytes / 1024**3:.4f} GB ({len(df):,} righe)")
    return total_bytes


def calc_s3_size() -> float:
    """Sum Dimensione across lista_s3.csv in chunks. Returns total in bytes."""
    print(f"\n[SIZE] Calcolo dimensione S3 ({S3_FILE.name}, lettura a chunk)...")
    total_bytes = 0.0
    n_rows = 0
    for chunk in tqdm(
        pd.read_csv(S3_FILE, usecols=["Dimensione"], dtype=str, low_memory=False, chunksize=CHUNK_SIZE),
        desc="  Chunk S3", unit="chunk",
    ):
        col = pd.to_numeric(chunk["Dimensione"], errors="coerce")
        total_bytes += col.sum()
        n_rows += len(chunk)
    print(f"  → {n_rows:,} righe elaborate")
    return total_bytes


def _print_size_result(label: str, total_bytes: float) -> None:
    gb = total_bytes / 1024**3
    tb = gb / 1024
    print(f"\n  {'─' * 40}")
    print(f"  {label}")
    print(f"    Totale bytes : {total_bytes:,.0f}")
    print(f"    Totale GB    : {gb:,.3f} GB")
    print(f"    Totale TB    : {tb:,.4f} TB")
    print(f"  {'─' * 40}")


def cmd_size(source: str) -> None:
    print("=" * 60)
    print(f"  Analisi Dimensione — sorgente: {source.upper()}")
    print("=" * 60)

    if source in ("alfresco", "both"):
        alfresco_bytes = calc_alfresco_size()
        _print_size_result("ALFRESCO — totale", alfresco_bytes)

    if source in ("s3", "both"):
        s3_bytes = calc_s3_size()
        _print_size_result("S3 — totale", s3_bytes)

    if source == "both":
        combined = alfresco_bytes + s3_bytes
        _print_size_result("COMBINATO (Alfresco + S3)", combined)

    print(f"\n{'=' * 60}")
    print("  Calcolo completato.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# diff-size command
# ---------------------------------------------------------------------------

def cmd_diff_size() -> None:
    print("=" * 60)
    print("  Analisi Diff + Dimensioni (Alfresco ↔ S3)")
    print("=" * 60)

    alfresco_files = sorted(ALFRESCO_DIR.glob("*.csv"))
    if not alfresco_files:
        print(f"Nessun file CSV trovato in {ALFRESCO_DIR}")
        return

    print(f"\nFile Alfresco trovati: {len(alfresco_files)}")
    for f in alfresco_files:
        print(f"  • {f.name}")

    chunk_paths, n_s3_total = split_s3_into_chunks()
    alfresco_dfs = load_alfresco_files(alfresco_files)

    all_alfresco_names: set[str] = set()
    for df in alfresco_dfs.values():
        all_alfresco_names |= set(df["Nome"].dropna())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        print(f"\n[STEP 3] Analisi per-file Alfresco → S3 con dimensioni...")
        alf_results: list[AlfrescoSizeResult] = []
        for path in alfresco_files:
            alf_results.append(analyze_alfresco_to_s3_with_size(path, alfresco_dfs[path], chunk_paths))

        n_solo_s3, size_solo_s3, size_libretto = analyze_s3_with_size(
            chunk_paths, all_alfresco_names, n_s3_total
        )

        _write_diff_size_report(
            timestamp=timestamp,
            alf_results=alf_results,
            n_s3_total=n_s3_total,
            n_solo_s3=n_solo_s3,
            size_solo_s3_bytes=size_solo_s3,
            size_libretto_bytes=size_libretto,
        )
    finally:
        cleanup_tmp()

    print(f"\n{'=' * 60}")
    print("  Elaborazione completata.")
    print(f"  Report salvati in: {OUTPUT_DIR}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_tmp() -> None:
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
        print("\nChunk temporanei eliminati.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analisi Alfresco <-> AWS S3",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "comando",
        nargs="?",
        default="diff",
        choices=["diff", "size-alfresco", "size-s3", "size-both", "diff-size"],
        help=(
            "Tipo di analisi:\n"
            "  diff           Differenze Alfresco <-> S3 (default)\n"
            "  size-alfresco  Dimensione totale file Alfresco in GB\n"
            "  size-s3        Dimensione totale file S3 in GB\n"
            "  size-both      Dimensione totale di entrambe le sorgenti\n"
            "  diff-size      Differenze + pesi: presenti/assenti in S3, solo-S3, libretto"
        ),
    )
    args = parser.parse_args()

    if args.comando == "size-alfresco":
        cmd_size("alfresco")
        return
    if args.comando == "size-s3":
        cmd_size("s3")
        return
    if args.comando == "size-both":
        cmd_size("both")
        return
    if args.comando == "diff-size":
        cmd_diff_size()
        return

    # --- diff (default) ---
    print("=" * 60)
    print("  Analisi Differenze Alfresco <-> AWS S3")
    print("=" * 60)

    alfresco_files = sorted(ALFRESCO_DIR.glob("*.csv"))
    if not alfresco_files:
        print(f"Nessun file CSV trovato in {ALFRESCO_DIR}")
        return

    print(f"\nFile Alfresco trovati: {len(alfresco_files)}")
    for f in alfresco_files:
        print(f"  • {f.name}")

    chunk_paths, n_s3_total = split_s3_into_chunks()
    alfresco_dfs = load_alfresco_files(alfresco_files)

    all_alfresco_names: set[str] = set()
    n_alf_total = 0
    for df in alfresco_dfs.values():
        all_alfresco_names |= set(df["Nome"].dropna())
        n_alf_total += len(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        print(f"\n[STEP 3] Analisi per-file Alfresco → S3...")
        results: list[AlfrescoResult] = []
        for path in alfresco_files:
            results.append(analyze_alfresco_to_s3(path, alfresco_dfs[path], chunk_paths))

        _write_alfresco_report(results, n_s3_total, timestamp)

        analyze_s3_to_alfresco(
            chunk_paths, all_alfresco_names, n_s3_total,
            len(alfresco_files), n_alf_total, timestamp,
        )
    finally:
        cleanup_tmp()

    print(f"\n{'=' * 60}")
    print("  Elaborazione completata.")
    print(f"  Report salvati in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
