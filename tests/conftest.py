"""テスト全体の下ごしらえ。"""
import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    """テストごとに図を閉じる。開きっぱなしにすると matplotlib が警告を出す。"""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")
