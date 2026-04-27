import os
import shutil
import glob
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
MAX_TABLE_ROWS = 1000


def split_s3_into_chunks() -> tuple[list[Path], int]:
    print(f"\n[1/4] Split di {S3_FILE.name} in chunk da {CHUNK_SIZE:,} righe...")
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    chunk_paths = []
    total_rows = 0
    reader = pd.read_csv(S3_FILE, chunksize=CHUNK_SIZE, dtype=str, low_memory=False)

    for i, chunk in enumerate(tqdm(reader, desc="  Chunk S3", unit="chunk")):
        chunk_path = TMP_DIR / f"s3_chunk_{i:04d}.csv"
        chunk.to_csv(chunk_path, index=False)
        chunk_paths.append(chunk_path)
        total_rows += len(chunk)

    print(f"  → {len(chunk_paths)} chunk creati, {total_rows:,} righe totali in S3")
    return chunk_paths, total_rows


def analyze_alfresco_file(
    alfresco_path: Path, chunk_paths: list[Path], n_s3_total: int
) -> None:
    stem = alfresco_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n[2/4] Analisi file Alfresco: {alfresco_path.name}")

    df_alf = pd.read_csv(alfresco_path, dtype=str, low_memory=False)
    n_alfresco = len(df_alf)
    alfresco_names = set(df_alf["Nome"].dropna())
    print(f"  → {n_alfresco:,} righe Alfresco caricate")

    remaining_alf = set(alfresco_names)
    solo_s3_parts: list[pd.DataFrame] = []

    print("[3/4] Scansione chunk S3...")
    for chunk_path in tqdm(chunk_paths, desc="  Chunk", unit="chunk"):
        chunk = pd.read_csv(chunk_path, dtype=str, low_memory=False)
        mask_in_alf = chunk["NomeFile"].isin(alfresco_names)
        remaining_alf -= set(chunk.loc[mask_in_alf, "NomeFile"].dropna())
        solo_s3_parts.append(chunk.loc[~mask_in_alf].copy())

    df_solo_s3 = pd.concat(solo_s3_parts, ignore_index=True) if solo_s3_parts else pd.DataFrame()
    df_solo_alf = df_alf[df_alf["Nome"].isin(remaining_alf)].copy()

    n_comuni = n_alfresco - len(df_solo_alf)
    n_solo_alf = len(df_solo_alf)
    n_solo_s3 = len(df_solo_s3)

    pct_comuni_alf = n_comuni / n_alfresco * 100 if n_alfresco else 0
    pct_solo_alf = n_solo_alf / n_alfresco * 100 if n_alfresco else 0
    pct_solo_s3 = n_solo_s3 / n_s3_total * 100 if n_s3_total else 0

    print(f"  → Comuni: {n_comuni:,} | Solo Alfresco: {n_solo_alf:,} | Solo S3: {n_solo_s3:,}")

    print("[4/4] Generazione report Markdown...")
    _write_report(
        stem=stem,
        timestamp=timestamp,
        n_alfresco=n_alfresco,
        n_s3_total=n_s3_total,
        n_comuni=n_comuni,
        pct_comuni_alf=pct_comuni_alf,
        n_solo_alf=n_solo_alf,
        pct_solo_alf=pct_solo_alf,
        n_solo_s3=n_solo_s3,
        pct_solo_s3=pct_solo_s3,
        df_solo_alf=df_solo_alf,
        df_solo_s3=df_solo_s3,
    )


def _df_to_md_table(df: pd.DataFrame, cols: list[str], max_rows: int) -> str:
    subset = df[cols].head(max_rows)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = "\n".join(
        "| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |"
        for row in subset.itertuples(index=False)
    )
    return f"{header}\n{sep}\n{rows}"


def _write_report(
    *,
    stem: str,
    timestamp: str,
    n_alfresco: int,
    n_s3_total: int,
    n_comuni: int,
    pct_comuni_alf: float,
    n_solo_alf: int,
    pct_solo_alf: float,
    n_solo_s3: int,
    pct_solo_s3: float,
    df_solo_alf: pd.DataFrame,
    df_solo_s3: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"report_{stem}_{timestamp}.md"

    alf_cols = ["Nome", "Path", "Dimensione (bytes)", "MIME Type"]
    s3_cols = ["NomeFile", "Path", "Dimensione", "Data"]

    alf_table = _df_to_md_table(df_solo_alf, alf_cols, MAX_TABLE_ROWS)
    s3_table = _df_to_md_table(df_solo_s3, s3_cols, MAX_TABLE_ROWS)

    alf_truncated = len(df_solo_alf) > MAX_TABLE_ROWS
    s3_truncated = len(df_solo_s3) > MAX_TABLE_ROWS

    if alf_truncated:
        csv_alf = OUTPUT_DIR / f"solo_alfresco_{stem}_{timestamp}.csv"
        df_solo_alf.to_csv(csv_alf, index=False)

    if s3_truncated:
        csv_s3 = OUTPUT_DIR / f"solo_s3_{stem}_{timestamp}.csv"
        df_solo_s3.to_csv(csv_s3, index=False)

    lines = [
        f"# Report Analisi Differenze — {stem}",
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
        f"| File totali in Alfresco | {n_alfresco:,} | — |",
        f"| File totali in S3 | {n_s3_total:,} | — |",
        f"| Presenti in entrambi | {n_comuni:,} | {pct_comuni_alf:.2f}% di Alfresco |",
        f"| In Alfresco ma **NON** in S3 | {n_solo_alf:,} | {pct_solo_alf:.2f}% di Alfresco |",
        f"| In S3 ma **NON** in Alfresco | {n_solo_s3:,} | {pct_solo_s3:.2f}% di S3 |",
        f"",
        f"---",
        f"",
        f"## File in Alfresco ma assenti in S3",
        f"",
        f"**Totale:** {n_solo_alf:,}",
    ]

    if n_solo_alf == 0:
        lines.append("\n_Nessun file: tutti i file Alfresco sono presenti in S3._")
    else:
        if alf_truncated:
            lines.append(
                f"\n> **Nota:** tabella troncata ai primi {MAX_TABLE_ROWS:,} risultati. "
                f"Lista completa in `solo_alfresco_{stem}_{timestamp}.csv`."
            )
        lines.append("")
        lines.append(alf_table)

    lines += [
        f"",
        f"---",
        f"",
        f"## File in S3 ma assenti in Alfresco",
        f"",
        f"**Totale:** {n_solo_s3:,}",
    ]

    if n_solo_s3 == 0:
        lines.append("\n_Nessun file: tutti i file S3 sono presenti in Alfresco._")
    else:
        if s3_truncated:
            lines.append(
                f"\n> **Nota:** tabella troncata ai primi {MAX_TABLE_ROWS:,} risultati. "
                f"Lista completa in `solo_s3_{stem}_{timestamp}.csv`."
            )
        lines.append("")
        lines.append(s3_table)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → Report salvato: {report_path.name}")


def cleanup_tmp() -> None:
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
        print(f"\nChunk temporanei eliminati.")


def main() -> None:
    print("=" * 60)
    print("  Analisi Differenze Alfresco ↔ AWS S3")
    print("=" * 60)

    alfresco_files = sorted(ALFRESCO_DIR.glob("*.csv"))
    if not alfresco_files:
        print(f"Nessun file CSV trovato in {ALFRESCO_DIR}")
        return

    print(f"File Alfresco trovati: {len(alfresco_files)}")
    for f in alfresco_files:
        print(f"  • {f.name}")

    chunk_paths, n_s3_total = split_s3_into_chunks()

    try:
        for alfresco_path in alfresco_files:
            print(f"\n{'─' * 60}")
            analyze_alfresco_file(alfresco_path, chunk_paths, n_s3_total)
    finally:
        cleanup_tmp()

    print(f"\n{'=' * 60}")
    print("  Elaborazione completata.")
    print(f"  Report salvati in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
