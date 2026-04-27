import shutil
from dataclasses import dataclass
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
# Report writers
# ---------------------------------------------------------------------------

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
    print("=" * 60)
    print("  Analisi Differenze Alfresco ↔ AWS S3")
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
