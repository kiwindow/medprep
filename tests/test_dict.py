"""医学領域辞書の機械検査。

この辞書は人が手で育てるものなので、壊れ方も人為的なものになる。
CI で毎回すべての項目を検査し、次を保証する。

  * 範囲の向きが正しい（下限 < 上限）
  * 基準範囲・管理目標が「生理学的にあり得る範囲」の内側にある
  * 管理目標には必ず出典がある（出典の無い目標値を配らない）
  * 別名が項目どうしで衝突していない（列の同定が別の項目に流れない）
  * 派生指標には式か実装の記載がある
"""
import unicodedata

import pytest

from medprep.clean import build_alias_map, load_dict


def _norm(s):
    """build_alias_map と同じ正規化。テスト側で別の規則を使うと検査にならない。"""
    return unicodedata.normalize("NFKC", str(s)).replace(" ", "").lower()

DIC = load_dict()
ITEMS = DIC["items"]
RANGE_KEYS = ("reference", "reference_male", "reference_female")


def test_dictionary_loads_and_is_not_trivially_small():
    assert len(ITEMS) >= 80


@pytest.mark.parametrize("key", sorted(ITEMS))
def test_every_item_has_a_plausible_range_in_the_right_order(key):
    pl = ITEMS[key].get("plausible")
    assert isinstance(pl, list) and len(pl) == 2, f"{key}: plausible が無いか形式が違う"
    assert pl[0] < pl[1], f"{key}: plausible の上下が逆"


@pytest.mark.parametrize("key", sorted(ITEMS))
def test_reference_ranges_sit_inside_the_plausible_range(key):
    spec = ITEMS[key]
    lo, hi = spec["plausible"]
    for rk in RANGE_KEYS:
        r = spec.get(rk)
        if r is None:
            continue
        assert isinstance(r, list) and len(r) == 2 and r[0] <= r[1], f"{key}.{rk}: 形式が不正"
        assert lo <= r[0] and r[1] <= hi, f"{key}.{rk} が plausible の外"


@pytest.mark.parametrize("key", sorted(k for k, v in ITEMS.items() if "target" in v))
def test_targets_are_well_formed_and_cite_a_source(key):
    spec = ITEMS[key]
    t = spec["target"]
    lo_p, hi_p = spec["plausible"]
    assert t.get("source"), f"{key}: 管理目標に出典が無い"
    low, high = t.get("low"), t.get("high")
    assert low is not None or high is not None, f"{key}: 目標の範囲がまったく無い"
    if low is not None:
        assert lo_p <= low <= hi_p, f"{key}: target.low が plausible の外"
    if high is not None:
        assert lo_p <= high <= hi_p, f"{key}: target.high が plausible の外"
    if low is not None and high is not None:
        assert low < high, f"{key}: target の上下が逆"


@pytest.mark.parametrize("key", sorted(k for k, v in ITEMS.items() if "target" in v))
def test_targets_state_whether_bounds_are_inclusive(key):
    """「5.5 未満」と「5.5 以下」を取り違えないよう、明示を必須にする。"""
    t = ITEMS[key]["target"]
    if t.get("low") is not None:
        assert isinstance(t.get("low_inclusive"), bool), f"{key}: low_inclusive が未指定"
    if t.get("high") is not None:
        assert isinstance(t.get("high_inclusive"), bool), f"{key}: high_inclusive が未指定"


def test_no_legacy_dialysis_key_remains():
    """旧 `dialysis:` は開区間を表現できないので target に置き換えた。残していない。"""
    left = [k for k, v in ITEMS.items() if "dialysis" in v]
    assert not left, f"旧キーが残っている: {left}"


def test_alias_map_has_no_collisions_between_items():
    seen, collisions = {}, []
    for key, spec in ITEMS.items():
        for a in [key, spec.get("name_ja", "")] + list(spec.get("aliases") or []):
            if not a:
                continue
            norm = _norm(a)
            if norm in seen and seen[norm] != key:
                collisions.append((norm, seen[norm], key))
            seen[norm] = key
    assert not collisions, f"別名の衝突: {collisions[:5]}"


def test_alias_map_resolves_the_source_list_spellings():
    amap = build_alias_map(DIC)
    for spelling, expected in [
        ("末梢血｜血色素量(Hb)", "Hb"),
        ("C反応性蛋白(CRP)定量", "CRP"),
        ("インタクトPTH(iPTH)", "iPTH"),
        ("アルカリフォスファターゼ(ALP)", "ALP"),
        ("クリアスペース率A/V", "clearspace_AV"),
        ("Ht", "Ht"),          # ★身長に取られないこと★
        ("身長", "height"),
    ]:
        assert amap.get(_norm(spelling)) == expected, f"{spelling} が {expected} に解決されない"


@pytest.mark.parametrize("key", sorted(k for k, v in ITEMS.items() if v.get("derived")))
def test_derived_items_document_how_they_are_computed(key):
    spec = ITEMS[key]
    assert any(spec.get(k) for k in ("formula", "steps", "definition", "methods", "implemented")), \
        f"{key}: 派生指標だが算出方法の記載が無い"


def test_no_item_is_left_marked_as_requiring_review():
    left = [k for k, v in ITEMS.items() if v.get("requires_review")]
    assert not left, f"算出式が未確定の項目が残っている: {left}"


def test_white_cell_differential_composition_uses_neutrophils():
    g = DIC["composition_groups"]["白血球分画"]
    assert g["members"] == ["Neutro", "Eosino", "Baso", "Mono", "Lympho"]
    assert set(g["excluded_members"]) == {"Band", "Seg"}
    assert g["sum_to"] == 100


def test_measurement_changes_record_the_alp_method_switch():
    alp = [c for c in DIC["measurement_changes"] if c["item"] == "ALP"]
    assert alp and "IFCC" in alp[0]["change"]


def test_timing_required_covers_the_pre_post_sensitive_analytes():
    required = {k for k, v in ITEMS.items() if v.get("timing_required")}
    for k in ("BUN", "Cr", "K", "weight"):
        assert k in required, f"{k} に timing_required が立っていない"
