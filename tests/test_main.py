"""Unit tests for diff-size analysis functions."""
import io
from pathlib import Path

import pandas as pd
import pytest

from src.main import (
    LIBRETTO_PATTERN,
    AlfrescoSizeResult,
    _fmt_bytes,
    analyze_alfresco_to_s3_with_size,
    analyze_s3_with_size,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alf_csv(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _make_s3_chunk(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# _fmt_bytes
# ---------------------------------------------------------------------------

def test_fmt_bytes_zero():
    assert _fmt_bytes(0) == "0.000 GB"


def test_fmt_bytes_one_gb():
    assert _fmt_bytes(1024**3) == "1.000 GB"


def test_fmt_bytes_fractional():
    result = _fmt_bytes(512 * 1024**2)  # 0.5 GB
    assert "0.500" in result


# ---------------------------------------------------------------------------
# analyze_alfresco_to_s3_with_size
# ---------------------------------------------------------------------------

ALF_ROWS = [
    {"Nome": "a.pdf", "Path": "/alf/a.pdf", "Dimensione (bytes)": "1000", "MIME Type": "application/pdf"},
    {"Nome": "b.pdf", "Path": "/alf/b.pdf", "Dimensione (bytes)": "2000", "MIME Type": "application/pdf"},
    {"Nome": "c.pdf", "Path": "/alf/c.pdf", "Dimensione (bytes)": "3000", "MIME Type": "application/pdf"},
]

S3_ROWS = [
    {"NomeFile": "a.pdf", "Path": "/s3/a.pdf", "Dimensione": "1000", "Data": "2024-01-01"},
    {"NomeFile": "d.pdf", "Path": "/s3/d.pdf", "Dimensione": "5000", "Data": "2024-01-01"},
]


def test_analyze_alfresco_to_s3_with_size_basic(tmp_path):
    alf_path = _make_alf_csv(tmp_path, "alfresco.csv", ALF_ROWS)
    chunk_path = _make_s3_chunk(tmp_path, "s3_chunk.csv", S3_ROWS)
    df_alf = pd.read_csv(alf_path, dtype=str)

    result = analyze_alfresco_to_s3_with_size(alf_path, df_alf, [chunk_path])

    assert isinstance(result, AlfrescoSizeResult)
    assert result.stem == "alfresco"
    assert result.n_alfresco == 3
    assert result.n_in_s3 == 1          # only a.pdf is in S3
    assert result.n_solo_alf == 2        # b.pdf, c.pdf
    assert result.size_in_s3_bytes == pytest.approx(1000.0)
    assert result.size_solo_alf_bytes == pytest.approx(5000.0)  # 2000 + 3000


def test_analyze_alfresco_to_s3_with_size_all_in_s3(tmp_path):
    rows = [
        {"Nome": "x.pdf", "Path": "/x", "Dimensione (bytes)": "100", "MIME Type": ""},
        {"Nome": "y.pdf", "Path": "/y", "Dimensione (bytes)": "200", "MIME Type": ""},
    ]
    s3_rows = [
        {"NomeFile": "x.pdf", "Path": "/s3/x", "Dimensione": "100", "Data": ""},
        {"NomeFile": "y.pdf", "Path": "/s3/y", "Dimensione": "200", "Data": ""},
    ]
    alf_path = _make_alf_csv(tmp_path, "alf.csv", rows)
    chunk_path = _make_s3_chunk(tmp_path, "chunk.csv", s3_rows)
    df_alf = pd.read_csv(alf_path, dtype=str)

    result = analyze_alfresco_to_s3_with_size(alf_path, df_alf, [chunk_path])

    assert result.n_in_s3 == 2
    assert result.n_solo_alf == 0
    assert result.size_in_s3_bytes == pytest.approx(300.0)
    assert result.size_solo_alf_bytes == pytest.approx(0.0)


def test_analyze_alfresco_to_s3_with_size_none_in_s3(tmp_path):
    rows = [
        {"Nome": "z.pdf", "Path": "/z", "Dimensione (bytes)": "500", "MIME Type": ""},
    ]
    s3_rows = [
        {"NomeFile": "other.pdf", "Path": "/s3/other", "Dimensione": "999", "Data": ""},
    ]
    alf_path = _make_alf_csv(tmp_path, "alf.csv", rows)
    chunk_path = _make_s3_chunk(tmp_path, "chunk.csv", s3_rows)
    df_alf = pd.read_csv(alf_path, dtype=str)

    result = analyze_alfresco_to_s3_with_size(alf_path, df_alf, [chunk_path])

    assert result.n_in_s3 == 0
    assert result.n_solo_alf == 1
    assert result.size_in_s3_bytes == pytest.approx(0.0)
    assert result.size_solo_alf_bytes == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# analyze_s3_with_size
# ---------------------------------------------------------------------------

def test_analyze_s3_with_size_basic(tmp_path):
    s3_rows = [
        {"NomeFile": "a.pdf", "Path": "/docs/a.pdf", "Dimensione": "1000", "Data": ""},
        {"NomeFile": "b.pdf", "Path": "/libretto/b.pdf", "Dimensione": "2000", "Data": ""},
        {"NomeFile": "c.pdf", "Path": "/other/c.pdf", "Dimensione": "3000", "Data": ""},
    ]
    chunk_path = _make_s3_chunk(tmp_path, "chunk.csv", s3_rows)
    # a.pdf is in Alfresco; b.pdf and c.pdf are not
    alf_names = {"a.pdf"}

    n_solo, size_solo, size_libretto = analyze_s3_with_size([chunk_path], alf_names, 3)

    assert n_solo == 2                          # b.pdf + c.pdf
    assert size_solo == pytest.approx(5000.0)   # 2000 + 3000
    assert size_libretto == pytest.approx(2000.0)  # only b.pdf has libretto in path


def test_analyze_s3_with_size_libretto_case_insensitive(tmp_path):
    s3_rows = [
        {"NomeFile": "x.pdf", "Path": "/LIBRETTO/x.pdf", "Dimensione": "4000", "Data": ""},
        {"NomeFile": "y.pdf", "Path": "/Libretto/sub/y.pdf", "Dimensione": "1000", "Data": ""},
    ]
    chunk_path = _make_s3_chunk(tmp_path, "chunk.csv", s3_rows)

    _, _, size_libretto = analyze_s3_with_size([chunk_path], set(), 2)

    assert size_libretto == pytest.approx(5000.0)


def test_analyze_s3_with_size_libretti(tmp_path):
    """'libretti' (plural) must also be matched by the pattern."""
    s3_rows = [
        {"NomeFile": "a.pdf", "Path": "/Libretti/a.pdf", "Dimensione": "3000", "Data": ""},
        {"NomeFile": "b.pdf", "Path": "/LIBRETTI/sub/b.pdf", "Dimensione": "2000", "Data": ""},
        {"NomeFile": "c.pdf", "Path": "/other/c.pdf", "Dimensione": "1000", "Data": ""},
    ]
    chunk_path = _make_s3_chunk(tmp_path, "chunk.csv", s3_rows)

    _, _, size_libretto = analyze_s3_with_size([chunk_path], set(), 3)

    assert size_libretto == pytest.approx(5000.0)


def test_libretto_pattern_matches():
    import re
    pat = re.compile(LIBRETTO_PATTERN, re.IGNORECASE)
    assert pat.search("/libretto/doc.pdf")
    assert pat.search("/Libretto/doc.pdf")
    assert pat.search("/LIBRETTO/doc.pdf")
    assert pat.search("/libretti/doc.pdf")
    assert pat.search("/Libretti/doc.pdf")
    assert pat.search("/LIBRETTI/doc.pdf")
    assert not pat.search("/docs/other.pdf")


def test_analyze_s3_with_size_no_solo(tmp_path):
    s3_rows = [
        {"NomeFile": "a.pdf", "Path": "/docs/a.pdf", "Dimensione": "100", "Data": ""},
    ]
    chunk_path = _make_s3_chunk(tmp_path, "chunk.csv", s3_rows)
    alf_names = {"a.pdf"}

    n_solo, size_solo, size_libretto = analyze_s3_with_size([chunk_path], alf_names, 1)

    assert n_solo == 0
    assert size_solo == pytest.approx(0.0)
    assert size_libretto == pytest.approx(0.0)


def test_analyze_s3_with_size_multiple_chunks(tmp_path):
    chunk1 = _make_s3_chunk(tmp_path, "c1.csv", [
        {"NomeFile": "a.pdf", "Path": "/libretto/a.pdf", "Dimensione": "1000", "Data": ""},
    ])
    chunk2 = _make_s3_chunk(tmp_path, "c2.csv", [
        {"NomeFile": "b.pdf", "Path": "/other/b.pdf", "Dimensione": "2000", "Data": ""},
    ])
    alf_names = {"a.pdf"}

    n_solo, size_solo, size_libretto = analyze_s3_with_size([chunk1, chunk2], alf_names, 2)

    assert n_solo == 1
    assert size_solo == pytest.approx(2000.0)
    assert size_libretto == pytest.approx(1000.0)


def test_analyze_s3_with_size_libretto_not_in_filename(tmp_path):
    """libretto in path directory but filename doesn't contain it."""
    s3_rows = [
        {"NomeFile": "doc.pdf", "Path": "/archivio/libretto/2024/doc.pdf", "Dimensione": "500", "Data": ""},
        {"NomeFile": "other.pdf", "Path": "/docs/other.pdf", "Dimensione": "300", "Data": ""},
    ]
    chunk_path = _make_s3_chunk(tmp_path, "chunk.csv", s3_rows)

    _, _, size_libretto = analyze_s3_with_size([chunk_path], set(), 2)

    assert size_libretto == pytest.approx(500.0)
