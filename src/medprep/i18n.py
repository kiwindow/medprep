"""medprep.i18n — 表を英語で出すための対訳。

**論文の Table 1 は英語で要る。** 日本語版だけ作っても、そのあと人が手で
訳し直すことになり、そこで誤訳と写し間違いが入る。だから同じ数値から
日本語版と英語版の**両方を機械が作る**。

対訳が無い列は**訳さずそのまま残す**。★勝手にローマ字にしない。★
訳せなかった列は `missing_english()` で数えられるので、必要ならここに足す。
"""

from __future__ import annotations

import unicodedata


def _norm(s) -> str:
    return unicodedata.normalize("NFKC", str(s)).strip()


#: `dict/ranges_ja.yaml` の項目キー → 英語名
NAME_EN: dict[str, str] = {
    "age": "Age", "height": "Height", "weight": "Body weight",
    "DW": "Dry weight", "BMI": "BMI", "vintage": "Dialysis vintage",
    "SBP": "Systolic blood pressure", "DBP": "Diastolic blood pressure",
    "Td": "Dialysis session length", "weight_gain": "Interdialytic weight gain",
    "interdialytic_days": "Interdialytic interval",
    "BUN_next_pre": "Next pre-dialysis BUN",
    "interdialytic_hours": "Interdialytic time",
    "V_post": "Post-dialysis urea distribution volume",
    "M_removed": "Removed urea mass", "KtV": "spKt/V",
    "TG": "Triglycerides", "BUN": "Blood urea nitrogen", "P": "Phosphorus",
    "Ca": "Calcium", "AMY": "Amylase", "T-Bil": "Total bilirubin",
    "TP": "Total protein", "Alb": "Albumin", "CK": "Creatine kinase",
    "LAP": "Leucine aminopeptidase", "GGT": "γ-glutamyl transferase",
    "AST": "Aspartate aminotransferase", "ALT": "Alanine aminotransferase",
    "Cr": "Creatinine", "UA": "Uric acid", "Fe": "Serum iron",
    "Mg": "Magnesium", "ChE": "Cholinesterase", "Na": "Sodium",
    "K": "Potassium", "Cl": "Chloride",
    "TIBC": "Total iron-binding capacity", "LDL": "LDL cholesterol",
    "CEA": "Carcinoembryonic antigen", "AFP": "α-fetoprotein",
    "GLU": "Glucose", "HDL": "HDL cholesterol", "WBC": "White blood cells",
    "RBC": "Red blood cells", "Hb": "Hemoglobin", "Ht": "Hematocrit",
    "MCV": "MCV", "MCH": "MCH", "MCHC": "MCHC", "Plt": "Platelets",
    "APTT": "APTT", "PT_sec": "Prothrombin time",
    "PT_pct": "Prothrombin activity", "PT_ratio": "PT ratio", "PT_INR": "PT-INR",
    "B2MG": "β2-microglobulin", "Ferritin": "Ferritin",
    "iPTH": "Intact parathyroid hormone", "GA": "Glycated albumin",
    "HbA1c": "HbA1c", "Zn": "Zinc", "hANP": "Human ANP",
    "BNP": "B-type natriuretic peptide", "NT_proBNP": "NT-proBNP",
    "FT3": "Free T3", "FT4": "Free T4", "CRP": "C-reactive protein",
    "Neutro": "Neutrophils", "Band": "Band neutrophils",
    "Seg": "Segmented neutrophils", "Eosino": "Eosinophils",
    "Baso": "Basophils", "Mono": "Monocytes", "Lympho": "Lymphocytes",
    "ALP": "Alkaline phosphatase", "LDH": "Lactate dehydrogenase",
    "GNRI": "Geriatric nutritional risk index",
    "cCa": "Corrected calcium", "cCaxP": "Corrected Ca × P",
    "TSAT": "Transferrin saturation",
    "nPCR": "Normalized protein catabolic rate",
    "pct_CGR": "%Creatinine generation rate",
    "clearspace_AV": "Clear space ratio", "salt": "Salt intake",
    "TAC": "Time-averaged concentration of urea",
}

#: 辞書に載らない列名（属性・派生・日付など）の対訳
COLUMN_EN: dict[str, str] = {
    "施設": "Facility", "施設コード": "Facility code",
    "性別": "Sex", "男性": "Male sex", "女性": "Female sex",
    "年齢": "Age", "透析歴_月": "Dialysis vintage",
    "糖尿病": "Diabetes mellitus", "備考": "Remarks",
    "身長": "Height", "体重": "Body weight",
    "透析開始時刻": "Dialysis start time", "透析終了時刻": "Dialysis end time",
    "透析時間(hr)": "Dialysis session length", "除水量(kg)": "Ultrafiltration volume",
    "URR(%)": "Urea reduction ratio", "spKt/V": "spKt/V",
    "TSAT(%)": "Transferrin saturation", "iCa(mg/dL)": "Corrected calcium",
    "iCa×P": "Corrected Ca × P",
    "転帰": "Outcome", "死亡": "Death", "event": "Event",
    "検体採取日": "Sampling date", "観察開始年月日": "Observation start date",
    "event発生年月日": "Event date", "観察打ち切り年月日": "Censoring date",
    "除外推奨": "Flagged for exclusion", "除外推奨_理由": "Reason for exclusion flag",
    "全体": "Overall", "特性": "Characteristic", "p値": "p-value",
}

#: 採血時点の接頭辞
TIMING_EN: dict[str, str] = {"透析前": "Pre-dialysis", "透析後": "Post-dialysis"}

#: 水準（カテゴリの値）の対訳
LEVEL_EN: dict[str, str] = {
    "男": "Male", "男性": "Male", "女": "Female", "女性": "Female",
    "あり": "Yes", "なし": "No", "有": "Yes", "無": "No",
    "陽性": "Positive", "陰性": "Negative",
    "はい": "Yes", "いいえ": "No",
    "生存": "Alive", "死亡": "Dead",
    "不明": "Unknown", "その他": "Other",
}

#: 単位の対訳。★英語の表に日本語の単位を残さない。★
UNIT_EN: dict[str, str] = {
    "歳": "years", "月": "months", "日": "days", "年": "years",
    "時間": "hours", "hr": "hours", "分": "minutes",
    "回": "times", "件": "n", "例": "n", "人": "n",
    "cm": "cm", "kg": "kg", "L": "L", "mL": "mL",
}


def unit_en(unit: str) -> str:
    u = _norm(unit)
    return UNIT_EN.get(u, u)


#: 検定名の対訳（脚注に使う）
TEST_EN: dict[str, str] = {
    "Welch の t 検定": "Welch's t-test",
    "対応のある t 検定": "Paired t-test",
    "Mann-Whitney U 検定": "Wilcoxon rank-sum test",
    "Wilcoxon 符号付順位検定": "Wilcoxon signed-rank test",
    "一元配置分散分析": "One-way ANOVA",
    "Kruskal-Wallis 検定": "Kruskal-Wallis test",
    "χ² 検定": "Pearson's Chi-squared test",
    "Fisher 正確検定": "Fisher's exact test",
}

_SUFFIX_EN = {"院": "Hospital", "病院": "Hospital", "クリニック": "Clinic"}


def level_en(value) -> str:
    """水準（カテゴリの値）を英語にする。訳が無ければそのまま返す。

    `A院` のように **記号 + 施設種別** の形は `Hospital A` に直す。
    それ以外の未知の値は**そのまま**（★勝手にローマ字にしない★）。
    """
    v = _norm(value)
    if v in LEVEL_EN:
        return LEVEL_EN[v]
    for suf, en in _SUFFIX_EN.items():
        if len(v) > len(suf) and v.endswith(suf):
            head = v[: -len(suf)]
            if head.isascii():
                return f"{en} {head}"
    return v


def column_en(col: str, key: str | None = None, timing: str | None = None) -> str:
    """列の英語名。`key` は `ranges_ja.yaml` の項目キー、`timing` は採血時点。

    優先順位は **列名の直接対訳 → 辞書キーの対訳 → 元の列名のまま**。
    """
    c = _norm(col)
    if c in COLUMN_EN:
        return COLUMN_EN[c]
    base, pre = c, ""
    for ja, en in TIMING_EN.items():
        if c.startswith(ja):
            base, pre = c[len(ja):], en
            break
    if timing in ("pre", "post") and not pre:
        pre = "Pre-dialysis" if timing == "pre" else "Post-dialysis"
    name = None
    if base in COLUMN_EN:
        name = COLUMN_EN[base]
    elif key and key in NAME_EN:
        name = NAME_EN[key]
    if name is None:
        return c
    return f"{pre} {name[0].lower() + name[1:]}" if pre else name


def missing_english(columns) -> list:
    """英語名が用意できなかった列（そのまま出す列）を返す。"""
    return [c for c in columns if column_en(c) == _norm(c) and not _norm(c).isascii()]
