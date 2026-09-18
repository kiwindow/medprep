"""パッケージとしての体裁。インストールされた状態で壊れていないことを確かめる。"""
import importlib

import pytest

import medprep


def test_version_is_exposed():
    assert medprep.__version__.count(".") == 2


def test_dictionary_ships_inside_the_wheel():
    """dict/ranges_ja.yaml が同梱されていないと、インストール後に何も動かない。"""
    d = medprep.load_dict()
    assert "items" in d and len(d["items"]) >= 80


@pytest.mark.parametrize("name", [
    "clean", "dates", "hd", "quality", "schema", "survival_input", "tac", "targets",
    "timing",
])
def test_submodules_import(name):
    importlib.import_module(f"medprep.{name}")


@pytest.mark.parametrize("name", medprep.__all__)
def test_public_api_is_actually_present(name):
    assert hasattr(medprep, name), f"__all__ に {name} があるが実体が無い"


def test_no_heavy_import_at_module_load():
    """import medprep だけで matplotlib のバックエンドを起動しない（CI とサーバで困る）。"""
    import sys
    mods = set(sys.modules)
    importlib.reload(medprep)
    assert "torch" not in set(sys.modules) - mods


# --- スカラーを渡したらスカラーが返ること（float() がそのまま書けること）---
import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.mark.parametrize("call", [
    lambda: medprep.urr(60, 20),
    lambda: medprep.sp_ktv(60, 20, 63, 60, 4),
    lambda: medprep.npcr(1.4, 60, 20),
    lambda: medprep.corrected_ca(8.5, 3.0),
    lambda: medprep.tsat(60, 300),
    lambda: medprep.salt_intake(3.0, 140, 3.0),
    lambda: medprep.bmi(60, 170),
    lambda: medprep.ideal_body_weight(170),
    lambda: medprep.gnri(3.8, 58.0, 165),
    lambda: medprep.clear_space_ratio(15120, 60, 36)["AV_percent"],
    lambda: medprep.removed_mass_from_effluent(12, 126),
    lambda: medprep.clear_space_ratio_from_weight(60, 20, 63, 60, 36),
    lambda: medprep.bun_to_urea_mg_dl(40),
    lambda: medprep.tac_bun_simple(20, 60).value,
    lambda: medprep.percent_cgr(sex="男性", age=60, bun_pre=60, bun_post=20,
                                cr_pre=12, cr_post=4, bw_pre=63, bw_post=60,
                                td_hours=4).percent_CGR,
])
def test_scalar_in_scalar_out(call):
    v = call()
    assert isinstance(v, float), f"スカラーを渡したのに {type(v).__name__} が返った"
    assert float(v) == v


def test_array_in_array_out():
    v = medprep.urr([60, 50], [20, 25])
    assert isinstance(v, np.ndarray) and v.shape == (2,)


def test_cgr_result_fields_are_all_scalars_for_one_case():
    r = medprep.percent_cgr(sex="男性", age=60, bun_pre=60, bun_post=20, cr_pre=12,
                            cr_post=4, bw_pre=63, bw_post=60, td_hours=4)
    for name in ("delta_bw", "R", "spKtV", "nPCR", "Cr_corr", "L", "D", "A",
                 "G_total", "G_ext", "G_int", "G_ref", "percent_CGR"):
        assert isinstance(getattr(r, name), float), f"{name} がスカラーでない"
