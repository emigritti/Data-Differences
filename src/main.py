import shutil
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


# ---------------------------------------------------------------------------
# Split S3
# ---------------------------------------------------------------------------

def split_s3_into_chunks() -> tuple[list[Path], int]:
    print(f"\n[STEP 1] Split di {S3_FILE.name} in chunk da {CHUNK_SIZE:,} righe...")
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    chunk_paths: list[Path] = []
    total_rows = 0
    reader = pd.read_csv(S3_FILE, chunksize=CHUNK_SIZE, dtype=str, low_memory=False)

    for i, chunk in enumerate(tqdm(reader, desc="  Chunk S3", unit="chunk")):
        path = TMP_DIR / f"s3_chunk_{i:04d}.csv"
        chunk.to_csv(path, index=False)
        chunk_paths.append(path)
        total_rows += len(chunk)

    print(f"  → {len(chunk_paths)} chunk creati, {total_rows:,} righe totali in S3")
    return chunk_paths, total_rows


# ---------------------------------------------------------------------------
# Load Alfresco files
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
# Analysis 1: per-file Alfresco → S3
# Cosa c'è in QUESTO file Alfresco ma non in nessun chunk S3
# ---------------------------------------------------------------------------

def analyze_alfresco_to_s3(
    alfresco_path: Path,
    df_alf: pd.DataFrame,
    chunk_paths: list[Path],
    n_s3_total: int,
    timestamp: str,
) -> None:
    stem = alfresco_path.stem
    print(f"\n  Alfresco→S3: {alfresco_path.name}")

    alf_names = set(df_alf["Nome"].dropna())
    n_alfresco = len(df_alf)
    remaining = set(alf_names)

    for chunk_path in tqdm(chunk_paths, desc="    Chunk", unit="chunk", leave=False):
        chunk = pd.read_csv(chunk_path, dtype=str, low_memory=False)
        found = set(chunk["NomeFile"].dropna()) & remaining
        remaining -= found
        if not remaining:
            break  # tutti trovati, inutile continuare

    df_solo_alf = df_alf[df_alf["Nome"].isin(remaining)].copy()
    n_comuni = n_alfresco - len(df_solo_alf)
    n_solo_alf = len(df_solo_alf)

    pct_comuni = n_comuni / n_alfresco * 100 if n_alfresco else 0
    pct_solo_alf = n_solo_alf / n_alfresco * 100 if n_alfresco else 0

    print(f"    → Comuni con S3: {n_comuni:,} ({pct_comuni:.1f}%) | Solo Alfresco: {n_solo_alf:,} ({pct_solo_alf:.1f}%)")

    _write_per_file_report(
        stem=stem,
        timestamp=timestamp,
        n_alfresco=n_alfresco,
        n_s3_total=n_s3_total,
        n_comuni=n_comuni,
        pct_comuni=pct_comuni,
        n_solo_alf=n_solo_alf,
        pct_solo_alf=pct_solo_alf,
        df_solo_alf=df_solo_alf,
    )


# ---------------------------------------------------------------------------
# Analysis 2: global S3 → all Alfresco
# Cosa c'è in S3 che non compare in NESSUNO dei file Alfresco
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

    solo_s3_parts: list[pd.DataFrame] = []

    for chunk_path in tqdm(chunk_paths, desc="  Chunk S3", unit="chunk"):
        chunk = pd.read_csv(chunk_path, dtype=str, low_memory=False)
        mask_not_in_alf = ~chunk["NomeFile"].isin(all_alfresco_names)
        if mask_not_in_alf.any():
            solo_s3_parts.append(chunk.loc[mask_not_in_alf].copy())

    df_solo_s3 = pd.concat(solo_s3_parts, ignore_index=True) if solo_s3_parts else pd.DataFrame()
    n_solo_s3 = len(df_solo_s3)
    pct_solo_s3 = n_solo_s3 / n_s3_total * 100 if n_s3_total else 0

    print(f"  → Solo S3 (non in nessun Alfresco): {n_solo_s3:,} ({pct_solo_s3:.1f}%)")

    _write_global_report(
        timestamp=timestamp,
        n_alf_files=n_alf_files,
        n_alf_total=n_alf_total,
        n_alf_unique=len(all_alfresco_names),
        n_s3_total=n_s3_total,
        n_solo_s3=n_solo_s3,
        pct_solo_s3=pct_solo_s3,
        df_solo_s3=df_solo_s3,
    )


# ---------------------------------------------------------------------------
# Report helpers
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


def _write_per_file_report(
    *,
    stem: str,
    timestamp: str,
    n_alfresco: int,
    n_s3_total: int,
    n_comuni: int,
    pct_comuni: float,
    n_solo_alf: int,
    pct_solo_alf: float,
    df_solo_alf: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"report_alfresco_vs_s3_{stem}_{timestamp}.md"
    alf_cols = ["Nome", "Path", "Dimensione (bytes)", "MIME Type"]

    truncated = len(df_solo_alf) > MAX_TABLE_ROWS
    if truncated:
        csv_path = OUTPUT_DIR / f"solo_alfresco_{stem}_{timestamp}.csv"
        df_solo_alf.to_csv(csv_path, index=False)

    lines = [
        f"# Report Alfresco → S3 — {stem}",
        f"",
        f"**Data elaborazione:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**File Alfresco:** `{stem}.csv`  ",
        f"**File S3:** `lista_s3.csv`",
        f"",
        f"---",
        f"",
        f"## Riepilogo",
        f"",
        f"| Metrica | Valore | % |",
        f"| --- | --- | --- |",
        f"| File totali in questo Alfresco | {n_alfresco:,} | — |",
        f"| File totali in S3 | {n_s3_total:,} | — |",
        f"| Presenti anche in S3 | {n_comuni:,} | {pct_comuni:.2f}% di Alfresco |",
        f"| In Alfresco ma **NON** in S3 | {n_solo_alf:,} | {pct_solo_alf:.2f}% di Alfresco |",
        f"",
        f"---",
        f"",
        f"## File in Alfresco ma assenti in S3",
        f"",
        f"**Totale:** {n_solo_alf:,}",
    ]

    if n_solo_alf == 0:
        lines.append("\n_Nessun file mancante: tutti i file di questo Alfresco sono presenti in S3._")
    else:
        if truncated:
            lines.append(
                f"\n> **Nota:** tabella troncata ai primi {MAX_TABLE_ROWS:,} risultati. "
                f"Lista completa in `solo_alfresco_{stem}_{timestamp}.csv`."
            )
        lines.append("")
        lines.append(_df_to_md_table(df_solo_alf, alf_cols, MAX_TABLE_ROWS))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"    → Report: {report_path.name}")


def _write_global_report(
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
    report_path = OUTPUT_DIR / f"report_s3_vs_alfresco_globale_{timestamp}.md"
    s3_cols = ["NomeFile", "Path", "Dimensione", "Data"]

    truncated = len(df_solo_s3) > MAX_TABLE_ROWS
    if truncated:
        csv_path = OUTPUT_DIR / f"solo_s3_globale_{timestamp}.csv"
        df_solo_s3.to_csv(csv_path, index=False)

    n_in_alf = n_s3_total - n_solo_s3
    pct_in_alf = n_in_alf / n_s3_total * 100 if n_s3_total else 0

    lines = [
        f"# Report Globale S3 → Alfresco",
        f"",
        f"**Data elaborazione:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**File Alfresco analizzati:** {n_alf_files}  ",
        f"**File S3:** `lista_s3.csv`",
        f"",
        f"---",
        f"",
        f"## Riepilogo",
        f"",
        f"| Metrica | Valore | % |",
        f"| --- | --- | --- |",
        f"| File totali in S3 | {n_s3_total:,} | — |",
        f"| Righe totali in Alfresco (tutti i file) | {n_alf_total:,} | — |",
        f"| Nomi unici in Alfresco (union) | {n_alf_unique:,} | — |",
        f"| Presenti in almeno un file Alfresco | {n_in_alf:,} | {pct_in_alf:.2f}% di S3 |",
        f"| In S3 ma **NON** in nessun Alfresco | {n_solo_s3:,} | {pct_solo_s3:.2f}% di S3 |",
        f"",
        f"---",
        f"",
        f"## File in S3 ma assenti in tutti i file Alfresco",
        f"",
        f"**Totale:** {n_solo_s3:,}",
    ]

    if n_solo_s3 == 0:
        lines.append("\n_Nessun file mancante: tutti i file S3 sono presenti in almeno un file Alfresco._")
    else:
        if truncated:
            lines.append(
                f"\n> **Nota:** tabella troncata ai primi {MAX_TABLE_ROWS:,} risultati. "
                f"Lista completa in `solo_s3_globale_{timestamp}.csv`."
            )
        lines.append("")
        lines.append(_df_to_md_table(df_solo_s3, s3_cols, MAX_TABLE_ROWS))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → Report globale: {report_path.name}")


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

    # Step 1: split S3
    chunk_paths, n_s3_total = split_s3_into_chunks()

    # Step 2: carica tutti i file Alfresco
    alfresco_dfs = load_alfresco_files(alfresco_files)

    # Unione globale di tutti i nomi Alfresco (usata per l'analisi S3→Alfresco)
    all_alfresco_names: set[str] = set()
    n_alf_total = 0
    for df in alfresco_dfs.values():
        all_alfresco_names |= set(df["Nome"].dropna())
        n_alf_total += len(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Step 3: per ogni file Alfresco → analisi verso S3
        print(f"\n[STEP 3] Analisi per-file Alfresco → S3...")
        for alfresco_path in alfresco_files:
            analyze_alfresco_to_s3(
                alfresco_path,
                alfresco_dfs[alfresco_path],
                chunk_paths,
                n_s3_total,
                timestamp,
            )

        # Step 4: analisi globale S3 → tutti i file Alfresco
        analyze_s3_to_alfresco(
            chunk_paths,
            all_alfresco_names,
            n_s3_total,
            len(alfresco_files),
            n_alf_total,
            timestamp,
        )

    finally:
        cleanup_tmp()

    print(f"\n{'=' * 60}")
    print("  Elaborazione completata.")
    print(f"  Report salvati in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
