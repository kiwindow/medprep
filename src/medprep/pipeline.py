"""medprep.pipeline — 前処理を 1 つの sklearn 変換器にまとめる。

**このモジュールの目的はリークを構造的に不可能にすることである。**

学習を伴う変換（補完の統計量、スケーラの平均と分散、エンコーダのカテゴリ集合、
Winsorize の閾値）は、すべて `ColumnTransformer` の中に閉じ込める。
`fit` は train にしか呼べず、`transform` は fit 済みのパラメータしか使えない。

    sp   = mp.split(df, sch)                 # 先に分ける
    prep = mp.Preprocessor(sch)
    Xtr  = prep.fit_transform(sp.train)      # fit は train だけ
    Xte  = prep.transform(sp.test)           # test は transform だけ

手作業の前処理では、この順序は**必ず**どこかで破れる。
「中央値で埋めてから分割する」1 行を書いた瞬間にリークが入り、しかも例外は出ない。
だから順序を人の記憶に頼らせない。

3 つの見張り
------------
1. **test に fit しようとしたら止める。**（`split()` が付けた印を見る）
2. **分割を経ていないデータに fit したら警告する。**（全データへの fit は典型的なリーク）
3. **fit に使った行と transform する行が部分的に重なっていたら警告する。**
   train と test が混ざっている合図である。

列名を残す
----------
`set_output(transform="pandas")` と `verbose_feature_names_out=False` の 2 つで、
出力は**列名付きの DataFrame** になる。これを外すと下流の係数プロット・
特徴量重要度・SHAP がすべて `x0, x1, …` になり、教材として成立しない。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from .outliers import NanSafeWinsorizer
from .schema import (
    BINARY,
    DATETIME,
    GROUP,
    NOMINAL,
    NUMERIC,
    ORDINAL,
    Schema,
    _norm,
    binary_output_name,
)
from .splitting import TEST, TRAIN, split_kind

_SCALERS = {"standard": StandardScaler, "robust": RobustScaler,
            "minmax": MinMaxScaler, "none": None}
INFREQUENT = "稀な水準"


class LeakageError(RuntimeError):
    """リークになる操作を止めたときに投げる。"""


class RenameFeatures(BaseEstimator, TransformerMixin):
    """列名を読める形に直すだけの変換器。値は触らない。

    sklearn の `add_indicator` が作る列は `missingindicator_年齢` という名前になる。
    受講者が見るのは最終的な列名なので、`欠損あり_年齢` に直す。
    `get_feature_names_out` もあわせて直すため、下流の係数プロットや SHAP でも
    同じ名前が出る。
    """

    def __init__(self, rules: tuple = (("missingindicator_", "欠損あり_"),)):
        self.rules = rules

    def _rename(self, name: str) -> str:
        s = str(name)
        for old, new in self.rules:
            if s.startswith(old):
                s = new + s[len(old):]
        return s

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X.columns = [self._rename(c) for c in X.columns]
        return X

    def get_feature_names_out(self, input_features=None):
        names = (input_features if input_features is not None
                 else getattr(self, "feature_names_in_", []))
        return np.asarray([self._rename(c) for c in names], dtype=object)


class BinaryMapper(BaseEstimator, TransformerMixin):
    """二値列を 0/1 に直す。**どちらが 1 かを列名で示す。**

    水準が 2 つの列は one-hot にしても列は 1 本にしかならない。問題は
    **どちらの水準が 1 になるか**で、`OneHotEncoder(drop="first")` はそれを
    sklearn の辞書順で決めてしまう。`男/女` は「女」が先なので `性別=男`、
    `M/F` は「F」が先なので `性別=M` と、**同じ意味の列が書き方で別名になる。**

    ここでは schema が見つけた対応表（`ColumnSpec.value_map`）をそのまま使う。
    男性=1・女性=0 と決まり、列名は `男性` になる。0 側が何かは
    `reference_levels()` で確かめられる。

    対応表に無い値（表記ゆれ、想定外の水準）は欠損にして、後段の補完に渡す。
    **学習するものが何も無い**ので、train と test で結果が食い違うことがない。
    """

    def __init__(self, mappings: dict | None = None, names: dict | None = None):
        self.mappings = mappings
        self.names = names

    def _map(self, col):
        return {_norm(k): v for k, v in (self.mappings or {}).get(col, {}).items()}

    def _name(self, col):
        return (self.names or {}).get(col, str(col))

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = {}
        for c in X.columns:
            m = self._map(c)
            col = X[c]
            vals = [np.nan if pd.isna(v) else m.get(_norm(v), np.nan) for v in col]
            out[self._name(c)] = pd.to_numeric(pd.Series(vals, index=col.index),
                                               errors="coerce")
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        names = (input_features if input_features is not None
                 else getattr(self, "feature_names_in_", []))
        return np.asarray([self._name(c) for c in names], dtype=object)


# ============================================== 素の DataFrame に同じ変換を掛ける
def encode_binary_columns(df: pd.DataFrame, schema: Schema) -> tuple[pd.DataFrame, pd.DataFrame]:
    """二値の列を **0/1 に直した DataFrame** と、**何をどう直したかの表**を返す。

    `Preprocessor` の中でやっているのと**同じ変換**を、素の DataFrame にも掛ける。
    「モデルに渡す行列では 0/1 なのに、自分で解析するファイルでは 男/女 のまま」
    という食い違いを無くすためである。

    - 列名は `binary_output_name()` で決める（性別 → **男性**、あり/なし → 元の列名）
    - 元の列は**同じ位置で**置き換える（列の並びは変わらない）
    - 対応表に無い値は欠損にする（**勝手に 0 にしない**）
    """
    out = df.copy()
    rows = []
    for col, sp in schema.columns.items():
        if col not in out.columns or sp.role != BINARY or not sp.value_map:
            continue
        name = binary_output_name(col, sp.value_map)
        if name == col and pd.api.types.is_numeric_dtype(out[col]) \
                and set(pd.unique(out[col].dropna())) <= {0, 1}:
            continue                         # すでに 0/1 で、名前も変わらない
        m = {_norm(k): v for k, v in sp.value_map.items()}
        if name != col and name in out.columns:
            name = col                       # 名前がぶつかるときは元の名前のまま
        src = out[col]
        vals = pd.to_numeric(
            pd.Series([np.nan if pd.isna(v) else m.get(_norm(v), np.nan) for v in src],
                      index=src.index), errors="coerce")
        pos = out.columns.get_loc(col)
        out = out.drop(columns=[col])
        out.insert(pos, name, vals)
        pos1 = next((k for k, v in sp.value_map.items() if v == 1), "")
        pos0 = next((k for k, v in sp.value_map.items() if v == 0), "")
        rows.append({"元の列": col, "元の値": f"{pos1} / {pos0}", "作った列": name,
                     "1 = ": str(pos1), "0 = ": str(pos0),
                     "対応表に無く欠損にした": int(vals.isna().sum() - src.isna().sum())})
    table = pd.DataFrame(rows, columns=["元の列", "元の値", "作った列", "1 = ", "0 = ",
                                        "対応表に無く欠損にした"])
    return out, table


# ================================================================== 設計行列
def _ohe_name(feature, category):
    """one-hot の列名。sklearn の 'infrequent_sklearn' は日本語に直す。"""
    cat = INFREQUENT if str(category) == "infrequent_sklearn" else category
    return f"{feature}={cat}"


def build_preprocessor(
    schema: Schema,
    policy: dict | None = None,
    *,
    columns: list | None = None,
    bounds: dict | None = None,
) -> tuple[ColumnTransformer, dict]:
    """schema と policy から `ColumnTransformer` を組む。

    Returns
    -------
    (ColumnTransformer, 判断の記録 dict)
    """
    pol = {**Schema.DEFAULT_POLICY, **(schema.policy or {}), **(policy or {})}
    feats = columns if columns is not None else schema.features()

    num, cat, ordi, dropped = [], [], [], []
    binmaps, binnames = {}, {}
    for c in feats:
        sp = schema.columns.get(c)
        if sp is None:
            dropped.append((c, "schema に無い"))
            continue
        if sp.role == NUMERIC:
            num.append(c)
        elif sp.role == ORDINAL:
            ordi.append(c)
        elif sp.role == BINARY and sp.value_map:
            # ★対応表のある二値列は one-hot に回さない。★
            #   男性=1・女性=0 のように**意味で**向きを決め、列名もそれに合わせる。
            binmaps[c] = dict(sp.value_map)
            binnames[c] = binary_output_name(c, sp.value_map)
        elif sp.role in (BINARY, NOMINAL, GROUP):
            cat.append(c)
        elif sp.role == DATETIME:
            # ★日付をそのまま特徴量にしない。★
            #   観察開始日を数値として入れると、暦の効果（診療の変化、測定法の変更）を
            #   モデルが学習してしまう。必要なら経過日数などを人が作って numeric にする。
            dropped.append((c, "日付列。暦の効果を学習させないため既定では入れない"))
        else:
            dropped.append((c, f"役割 {sp.role}"))

    # 出力名がぶつかるときは元の列名に戻す（`男性` という列が既にある場合など）
    taken = set(num) | set(ordi) | set(cat)
    for c, nm in list(binnames.items()):
        if nm != c and (nm in taken or list(binnames.values()).count(nm) > 1):
            binnames[c] = str(c)
        taken.add(binnames[c])

    binary = list(binmaps)
    steps, record = [], {"numeric": num, "categorical": cat, "ordinal": ordi,
                         "binary": binary, "binary_names": dict(binnames),
                         "binary_maps": dict(binmaps),
                         "dropped": dropped, "policy": pol}

    if num:
        steps.append(("num", _numeric_pipe(pol, bounds), num))
    if ordi:
        cats = [_ordinal_categories(schema, c) for c in ordi]
        steps.append(("ord", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(categories=cats, handle_unknown="use_encoded_value",
                                   unknown_value=np.nan)),
        ]), ordi))
    if binary:
        steps.append(("bin", Pipeline([
            ("map", BinaryMapper(mappings=dict(binmaps), names=dict(binnames))),
            ("imp", SimpleImputer(strategy="most_frequent")),
        ]), binary))
    if cat:
        enc = pol.get("encode", {})
        steps.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy=enc.get("categorical_impute", "most_frequent"))),
            ("ohe", OneHotEncoder(
                sparse_output=False,
                min_frequency=enc.get("min_frequency", 0.01),
                handle_unknown="infrequent_if_exist",
                # ★既定で基準水準を落とす。★ 落とさないと one-hot 同士が
                #   完全に従属し、線形モデル・Cox の係数が不安定になる。
                #   どの水準を基準にしたかは reference_levels() で確かめられる。
                drop=enc.get("drop", "first"),
                feature_name_combiner=_ohe_name)),
        ]), cat))

    ct = ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct, record


def _numeric_pipe(pol: dict, bounds: dict | None):
    out = pol.get("outlier", {})
    mis = pol.get("missing", {})
    sca = pol.get("scale", {})
    steps = []

    # 1) 外れ値（★補完より前★。補完を先にすると補完値が外れ値に汚染される）
    if out.get("action", "winsorize") == "winsorize":
        steps.append(("win", NanSafeWinsorizer(method=out.get("method", "iqr"),
                                               fold=out.get("fold", 1.5),
                                               bounds=bounds)))
    # 2) 補完（★欠損指示子を既定で残す★）
    strategy = mis.get("numeric", "median")
    if strategy == "iterative":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(random_state=mis.get("random_state", 0),
                               add_indicator=mis.get("add_indicator", True))
    elif strategy == "knn":
        from sklearn.impute import KNNImputer
        imp = KNNImputer(n_neighbors=mis.get("n_neighbors", 5),
                         add_indicator=mis.get("add_indicator", True))
    else:
        imp = SimpleImputer(strategy=strategy,
                            add_indicator=mis.get("add_indicator", True))
    steps.append(("imp", imp))

    # 3) 分布変換（任意）
    if pol.get("transform", {}).get("method") == "yeo-johnson":
        steps.append(("pow", PowerTransformer(method="yeo-johnson", standardize=False)))

    # 4) スケーリング
    scaler = _SCALERS.get(sca.get("method", "standard"))
    if scaler is not None:
        steps.append(("sc", scaler()))
    steps.append(("name", RenameFeatures()))
    return Pipeline(steps)


def _ordinal_categories(schema: Schema, col: str) -> list:
    sp = schema.columns[col]
    if sp.order:
        return [str(x) for x in sp.order]
    return sorted(str(x) for x in (sp.levels or []))


# ================================================================== 包み
@dataclass
class PrepRecord:
    n_fit: int = 0
    fit_kind: str | None = None
    feature_names: list = field(default_factory=list)
    design: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class Preprocessor:
    """`ColumnTransformer` を包み、**fit を train にしか呼べなくする。**

        prep = mp.Preprocessor(schema)
        Xtr  = prep.fit_transform(sp.train)
        Xte  = prep.transform(sp.test)
    """

    def __init__(self, schema: Schema, policy: dict | None = None, *,
                 columns: list | None = None, bounds: dict | None = None,
                 strict: bool = True):
        self.schema = schema
        self.strict = strict
        self.ct, self.design = build_preprocessor(schema, policy, columns=columns,
                                                  bounds=bounds)
        self.record = PrepRecord(design=self.design)
        self._fit_index: set | None = None
        self._train_clip: dict = {}
        self._fitted = False

    # -------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y=None) -> Preprocessor:
        kind = split_kind(X)
        if kind == TEST:
            raise LeakageError(
                "test に fit しようとしている。**これはリークである。**"
                " 前処理の統計量（補完の中央値、スケーラの平均、one-hot のカテゴリ集合、"
                " winsorize の閾値）は train からしか学習してはならない。"
                " prep.fit(split.train) → prep.transform(split.test) の順で呼ぶこと")
        if kind is None:
            msg = ("分割を経ていないデータに fit している。"
                   "全データに fit してから分割すると、test の情報が前処理に漏れる"
                   "（最も多いリークの形）。split() を先に呼ぶこと。"
                   "意図して全データに fit する場合は split.mark_as(df, 'train') を使う")
            if self.strict:
                self.record.warnings.append(msg)
                warnings.warn(msg, stacklevel=2)
            else:
                self.record.notes.append(msg)
        if self._fitted:
            self.record.notes.append("2 回目の fit を行った。1 回目に学習した統計量は捨てられる")

        missing = [c for c in self._needed() if c not in X.columns]
        if missing:
            raise ValueError(f"前処理に必要な列がデータに無い: {missing}")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.ct.fit(X, y)
        for w in caught:
            self.record.warnings.append(f"sklearn からの警告: {w.message}")

        self._fit_index = set(X.index)
        self._fitted = True
        # train 自身が winsorize で何割丸められたかを覚えておく。
        # test の丸め率をこれと比べないと、「分布がずれている」と
        # 「もともと歪んだ分布である」を区別できない。
        self._train_clip = self._clip_rates(X)
        self.record.n_fit = len(X)
        self.record.fit_kind = kind or "（印なし）"
        self.record.feature_names = list(self.ct.get_feature_names_out())
        self._warn_constant(X)
        return self

    def _warn_constant(self, X: pd.DataFrame) -> None:
        """★出来上がった行列に、値が 1 つしかない列が無いか確かめる。★

        前処理の途中で列が潰れても**例外は出ない**。スケーリング後は全例きっかり 0 に
        なり、見た目は正常なまま、その変数だけがモデルから消える。
        実際、IQR = 0 の列を winsorize して潰していた（透析時間は 4.0 が 6 割）。
        直したあとも、同じことが別の経路で起きないよう、ここで見張る。
        """
        try:
            out = self.ct.transform(X)
        except Exception:                                            # noqa: BLE001
            return
        out = pd.DataFrame(out)
        num = out.select_dtypes("number")
        dead = [c for c in num.columns if float(num[c].std(ddof=0)) == 0.0]
        if dead:
            msg = ("前処理のあと、値が 1 つしかない列がある: "
                   + "、".join(map(str, dead[:10]))
                   + "。**この変数はモデルに何も伝えない。**"
                   " 元の列の分布と、外れ値処理の閾値を確かめること")
            self.record.warnings.append(msg)

    # -------------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise LeakageError("まだ fit していない。fit(train) を先に呼ぶこと")
        self._check_overlap(X)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = self.ct.transform(X)
        for w in caught:
            m = str(w.message)
            if "unknown categories" in m:
                self.record.notes.append(
                    "train に無いカテゴリが test にあった。"
                    "handle_unknown='infrequent_if_exist' により『稀な水準』として扱った"
                    "（例外にせず、黙って全 0 にもしない）")
            else:
                self.record.warnings.append(f"sklearn からの警告: {m}")
        self._note_shift(X)
        return out

    def _clip_rates(self, X: pd.DataFrame) -> dict:
        win = self._winsorizer()
        if win is None:
            return {}
        out = {}
        for c in self.design["numeric"]:
            if c not in X.columns:
                continue
            lo, hi = win.limits_.get(c, (float("-inf"), float("inf")))
            v = pd.to_numeric(X[c], errors="coerce")
            n = int(v.notna().sum())
            if n:
                out[c] = float(((v < lo) | (v > hi)).sum()) / n
        return out

    def _note_shift(self, X: pd.DataFrame, *, margin: float = 0.05, ratio: float = 2.0):
        """新しいデータの丸め率が **train より目立って高い** ときだけ知らせる。

        ★train 自身の丸め率と比べる。★ 比べないと、CRP のように
        もともと歪んでいて train でも 1 割丸められる列が、毎回「分布がずれている」
        と言われることになり、本当のずれが埋もれる。
        """
        if not self._train_clip or (self._fit_index and set(X.index) <= self._fit_index):
            return                      # fit に使ったデータ自身は比べる相手がいない
        now = self._clip_rates(X)
        bad = [(c, now[c], self._train_clip.get(c, 0.0)) for c in now
               if now[c] > self._train_clip.get(c, 0.0) + margin
               and now[c] > self._train_clip.get(c, 0.0) * ratio]
        if not bad:
            return
        detail = "、".join(f"{c} {a:.0%}（train では {b:.0%}）" for c, a, b in bad[:5])
        msg = (f"train の範囲の外に出る値が、train 自身より目立って多い列がある（{detail}）。"
               f"winsorize で丸められるため、その分の違いは失われる。"
               f"train と test で分布がずれている合図なので、"
               f"時期・施設・測定法の違いを疑うこと（prep.distribution_shift(X) で一覧）")
        if msg not in self.record.warnings:
            self.record.warnings.append(msg)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # -------------------------------------------------------------- 分布のずれ
    def distribution_shift(self, X: pd.DataFrame, *, warn_at: float = 0.05) -> pd.DataFrame:
        """train で学習した winsorize の上下限に、どれだけの行が張り付くかを見る。

        **train の範囲の外にある test は、丸められて値の違いを失う。**
        極端な場合はその列が定数になり、モデルからは「情報が無い列」に見える。
        これは前処理の不具合ではなく、train と test の分布がずれている合図である
        （時期が違う、施設が違う、測定法が変わった）。
        """
        win = self._winsorizer()
        if win is None or not self._fitted:
            return pd.DataFrame()
        rows = []
        for c in self.design["numeric"]:
            if c not in X.columns:
                continue
            lo, hi = win.limits_.get(c, (float("-inf"), float("inf")))
            v = pd.to_numeric(X[c], errors="coerce")
            n = int(v.notna().sum())
            if not n:
                continue
            clipped = int(((v < lo) | (v > hi)).sum())
            rows.append({"列": c, "train の下限": lo, "train の上限": hi,
                         "丸められた行": clipped, "割合": clipped / n})
        out = pd.DataFrame(rows)
        if len(out):
            out = out[out["割合"] > 0].sort_values("割合", ascending=False)
            out["割合"] = out["割合"].map("{:.1%}".format)
        return out.reset_index(drop=True)

    def _winsorizer(self):
        try:
            return self.ct.named_transformers_["num"].named_steps["win"]
        except (KeyError, AttributeError):
            return None

    def _check_overlap(self, X: pd.DataFrame):
        if self._fit_index is None:
            return
        idx = set(X.index)
        inter = len(idx & self._fit_index)
        if inter == 0 or inter == len(idx):
            return                      # まるごと test か、まるごと train。どちらも正常
        msg = (f"transform しようとしているデータの {inter}/{len(idx)} 行が "
               f"fit に使った行と重なっている。**train と test が混ざっている合図である。**"
               f" 分割をやり直すこと")
        self.record.warnings.append(msg)
        warnings.warn(msg, stacklevel=3)

    def _needed(self) -> list:
        d = self.design
        return (list(d["numeric"]) + list(d["ordinal"])
                + list(d.get("binary", [])) + list(d["categorical"]))

    def input_columns(self) -> list:
        """前処理に**実際に投入した**元の列（変換後の列名ではない）。

        「どの列が特徴量にならなかったか」を数えるのに要る。
        """
        return self._needed()

    # -------------------------------------------------------------- 参照
    def reference_levels(self) -> dict:
        """one-hot で基準にした（＝列を作らなかった）水準。

        **基準が分からなければ係数もオッズ比も読めない。**
        """
        if not self._fitted:
            return {}
        out = {}

        # 二値列（0/1 に直したもの）。0 側が基準である。
        for col, m in (self.design.get("binary_maps") or {}).items():
            neg = next((k for k, v in m.items() if v == 0), None)
            if neg is not None:
                out[col] = str(neg)

        try:
            ohe = self.ct.named_transformers_["cat"].named_steps["ohe"]
        except (KeyError, AttributeError):
            return out
        if getattr(ohe, "drop_idx_", None) is None:
            return out
        for col, cats, idx in zip(self.design["categorical"], ohe.categories_,
                                  ohe.drop_idx_):
            if idx is not None:
                out[col] = str(cats[int(idx)])
        return out

    @property
    def feature_names_(self) -> list:
        if not self._fitted:
            raise LeakageError("まだ fit していない")
        return self.record.feature_names

    def report(self) -> str:
        d = self.design
        lines = [
            f"前処理 {'（fit 済み）' if self._fitted else '（未 fit）'}",
            f"  数値 {len(d['numeric'])} 列 / 二値 {len(d.get('binary', []))} 列 / "
            f"カテゴリ {len(d['categorical'])} 列 / 順序 {len(d['ordinal'])} 列",
        ]
        if d["dropped"]:
            lines.append("  入れなかった列: "
                         + "、".join(f"{c}（{why}）" for c, why in d["dropped"]))
        pol = d["policy"]
        lines.append(f"  方針: 外れ値={pol.get('outlier')} / 補完={pol.get('missing')} / "
                     f"符号化={pol.get('encode')} / スケール={pol.get('scale')}")
        if self._fitted:
            lines.append(f"  fit に使ったデータ: {self.record.n_fit} 例（{self.record.fit_kind}）")
            lines.append(f"  出力 {len(self.record.feature_names)} 列: "
                         + "、".join(self.record.feature_names[:12])
                         + ("…" if len(self.record.feature_names) > 12 else ""))
            refs = self.reference_levels()
            if refs:
                lines.append("  one-hot の基準水準（係数はこの水準との比になる）: "
                             + "、".join(f"{k}='{v}'" for k, v in refs.items()))
        for w in self.record.warnings:
            lines.append(f"[警告] {w}")
        for n in self.record.notes:
            lines.append(f"[注記] {n}")
        return "\n".join(lines)

    def show(self):
        print(self.report())

    def save(self, path):
        """fit 済みの前処理を保存する。`schema.yaml` と対で再現性を担保する。"""
        import joblib
        joblib.dump({"ct": self.ct, "design": self.design,
                     "features": self.record.feature_names}, path)
        return path


# ================================================================== 入口
@dataclass
class Prepared:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None
    preprocessor: Preprocessor | None = None
    notes: list = field(default_factory=list)

    def report(self) -> str:
        lines = [f"学習用の行列  train {self.X_train.shape} / test {self.X_test.shape}"]
        if self.preprocessor is not None:
            lines.append(self.preprocessor.report())
        for n in self.notes:
            lines.append(f"[注記] {n}")
        return "\n".join(lines)


def prepare(split_result, schema: Schema, policy: dict | None = None, *,
            outcome: str | None = None, columns: list | None = None,
            bounds: dict | None = None) -> Prepared:
    """`split()` の結果から、そのまま学習に渡せる行列を作る。

        sp = mp.split(df, sch)
        p  = mp.prepare(sp, sch)
        model.fit(p.X_train, p.y_train)
    """
    outcome = outcome or (schema.target["name"] if schema.target else None)
    prep = Preprocessor(schema, policy, columns=columns, bounds=bounds)
    xtr = prep.fit_transform(split_result.train)
    xte = prep.transform(split_result.test)
    ytr = yte = None
    notes = []
    if outcome:
        if outcome in split_result.train.columns:
            ytr = split_result.train[outcome]
            yte = split_result.test[outcome]
            if ytr.isna().any() or yte.isna().any():
                notes.append(
                    f"★目的変数 '{outcome}' に欠測が残っている"
                    f"（train {int(ytr.isna().sum())} 例 / test {int(yte.isna().sum())} 例）。★ "
                    f"目的変数は補完してはならない。"
                    f"missing.drop_missing_outcome() で先に除くこと")
        else:
            notes.append(f"目的変数 '{outcome}' が分割結果に無い")
    notes.append("前処理は train だけで fit した。test には transform しか当てていない")
    return Prepared(X_train=xtr, X_test=xte, y_train=ytr, y_test=yte,
                    preprocessor=prep, notes=notes)


# ================================================================== 検査
def leak_check(prep: Preprocessor, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """リークの典型を後から検査する。**通ったことの証拠として残せる表。**"""
    rows = []

    def add(name, ok, detail):
        rows.append({"検査": name, "結果": "OK" if ok else "★問題", "内容": detail})

    ktr, kte = split_kind(train), split_kind(test)
    add("train/test の印", ktr == TRAIN and kte == TEST,
        f"train={ktr}、test={kte}")

    inter = set(train.index) & set(test.index)
    add("行の重なり", not inter,
        "重なりなし" if not inter else f"{len(inter)} 行が両方にある")

    fit_ok = prep._fit_index is not None and prep._fit_index <= set(train.index)
    add("fit に使った行", fit_ok,
        "train の行だけで fit している" if fit_ok else "train 以外の行が fit に入っている")

    outcome = prep.schema.target["name"] if prep.schema.target else None
    feats = prep._needed()
    add("目的変数が特徴量に入っていないか", outcome not in feats,
        f"目的変数 = {outcome}" if outcome else "目的変数の指定なし")

    ids = [c for c, s in prep.schema.columns.items() if s.role == "id"]
    add("識別子が特徴量に入っていないか", not (set(ids) & set(feats)),
        f"識別子 = {ids or 'なし'}")

    from .splitting import group_overlap
    gcol = next((c for c, s in prep.schema.columns.items() if s.role == "id"), None)
    if gcol and gcol in train.columns and gcol in test.columns:
        ov = group_overlap(train, test, gcol)
        add("同じ患者が両側に出ていないか", not ov,
            "重なりなし" if not ov else f"{len(ov)} 人が両方にいる（{ov[:5]}…）")
    return pd.DataFrame(rows)
