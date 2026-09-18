"""ファイルを読む — csv / xlsx / parquet / json / sav(SPSS) / dta(Stata)。

**推測したことは必ず `df.attrs["medprep_read"]` に残す。**
文字コードを当てたのも、見出し行が 3 行目だと判断したのも推測である。
黙って推測すると、読み間違えたことに誰も気づかない。

モジュール名について
--------------------
構想では `io.py` だったが、標準ライブラリの `io` と紛らわしいので `loading.py`
にした（`audit.py` → `quality.py`、`split.py` → `splitting.py` と同じ理由）。
"""
from __future__ import annotations

import os

import pandas as pd

#: 日本の医療データで実際に出てくる順に試す。cp932 は Excel が吐く。
ENCODINGS = ["utf-8-sig", "cp932", "utf-8", "euc-jp", "latin-1"]

EXCEL = {".xlsx", ".xlsm", ".xltx", ".xls"}


def read_any(path, *, sheet=0, header: int | str = "auto",
             encoding: str | None = None, **kwargs) -> pd.DataFrame:
    """拡張子で読み方を決めて DataFrame にする。

    Parameters
    ----------
    header : "auto" なら見出し行を探す（表題や空行が上にある表に対応する）。
             整数を渡せばその行を見出しにする。
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ファイルが無い: {path}")
    ext = os.path.splitext(path)[1].lower()
    notes: list[str] = []

    if ext in EXCEL:
        df = _read_excel(path, sheet, header, notes, **kwargs)
    elif ext in (".csv", ".txt", ".tsv"):
        df = _read_csv(path, ext, header, encoding, notes, **kwargs)
    elif ext == ".parquet":
        df = pd.read_parquet(path, **kwargs)
    elif ext in (".json", ".jsonl"):
        df = pd.read_json(path, lines=(ext == ".jsonl"), **kwargs)
    elif ext == ".sav":
        df = _read_spss(path, **kwargs)
    elif ext == ".dta":
        df = pd.read_stata(path, **kwargs)
    else:
        raise ValueError(
            f"読み方が分からない拡張子: {ext!r}。"
            f"csv / xlsx / parquet / json / sav / dta に変換してから渡すこと")

    df = _drop_empty(df, notes)
    _check_duplicate_columns(df, notes)
    notes.insert(0, f"{os.path.basename(path)} を読んだ（{len(df)} 行 × {df.shape[1]} 列）")
    df.attrs["medprep_read"] = notes
    df.attrs["medprep_source"] = os.path.abspath(path)
    return df


# ================================================================== 個別
def _read_excel(path, sheet, header, notes, **kwargs):
    if header == "auto":
        probe = pd.read_excel(path, sheet_name=sheet, header=None, nrows=20)
        k = _header_row(probe)
        if k > 0:
            notes.append(f"★見出しは {k + 1} 行目と判断した★"
                         f"（上の {k} 行は表題か空行）。違うなら header= を指定すること")
        header = k
    return pd.read_excel(path, sheet_name=sheet, header=header, **kwargs)


def _read_csv(path, ext, header, encoding, notes, **kwargs):
    sep = kwargs.pop("sep", "\t" if ext == ".tsv" else ",")
    encodings = [encoding] if encoding else ENCODINGS
    last = None
    for enc in encodings:
        try:
            if header == "auto":
                probe = pd.read_csv(path, sep=sep, header=None, nrows=20,
                                    encoding=enc, engine="python")
                k = _header_row(probe)
                if k > 0:
                    notes.append(f"★見出しは {k + 1} 行目と判断した★"
                                 f"（上の {k} 行は表題か空行）。違うなら header= を指定すること")
            else:
                k = header
            df = pd.read_csv(path, sep=sep, header=k, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError) as e:
            last = e
            continue
        if encoding is None and enc != "utf-8-sig":
            notes.append(f"★文字コードは {enc} と判断した★"
                         f"（utf-8 では読めなかった）。文字化けしていないか必ず目で確かめること")
        return df
    raise UnicodeDecodeError(  # pragma: no cover - 5 通り全滅は事実上起きない
        "medprep", b"", 0, 1,
        f"文字コードを特定できなかった（{', '.join(ENCODINGS)} を試した）: {last}")


def _read_spss(path, **kwargs):
    try:
        return pd.read_spss(path, **kwargs)
    except ImportError as e:
        raise ImportError(
            "SPSS (.sav) を読むには pyreadstat が要る:  uv pip install pyreadstat") from e


# ================================================================== 補助
def _header_row(probe: pd.DataFrame, max_scan: int = 10) -> int:
    """見出しらしい最初の行を返す。

    医療データの Excel は 1 行目が「○○病院 透析患者一覧（2025年3月）」で、
    本当の見出しが 3 行目、ということがよくある。header=0 で読むと
    **列名が全部 `Unnamed: 1` になり、以降のすべてが狂う。**

    「半分以上のセルが埋まっている最初の行」を見出しとみなす。
    表題行はたいてい 1 セルしか埋まっておらず（結合セル）、空行は 0 なので飛ばせる。

    ★『値が重複しない行』という条件にしてはならない。★
    見出しに同じ名前の列が 2 本あるデータ（`Alb, Alb`）で見出し行そのものを
    飛ばしてしまい、**1 行目のデータが列名になる。**
    同名の列は `_check_duplicate_columns()` の側で警告する。

    見つからなければ 0（＝ふつうの表として扱う）。
    """
    ncol = probe.shape[1]
    if ncol == 0:
        return 0
    for i in range(min(max_scan, len(probe))):
        if probe.iloc[i].notna().sum() >= max(2, ncol * 0.5):
            return i
    return 0


def _drop_empty(df: pd.DataFrame, notes: list) -> pd.DataFrame:
    """完全に空の列・行を落とす（Excel の余白がそのまま列になることがある）。"""
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        notes.append(f"中身が空の列を {len(empty_cols)} 本落とした: "
                     + "、".join(str(c) for c in empty_cols[:5])
                     + ("…" if len(empty_cols) > 5 else ""))
    n0 = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    if len(df) < n0:
        notes.append(f"中身が空の行を {n0 - len(df)} 行落とした")
    return df


def _check_duplicate_columns(df: pd.DataFrame, notes: list) -> None:
    """同名の列は pandas が `X.1` に改名してしまう。黙って進めない。"""
    base = [str(c).rsplit(".", 1)[0] for c in df.columns]
    dup = {b for b in base if base.count(b) > 1}
    if dup:
        notes.append("★同じ名前の列がある★（pandas が `.1` を付けて区別している）: "
                     + "、".join(sorted(dup)[:5])
                     + "。どちらが必要な列か確かめること")
