"""第3回演習用の合成透析コホートを書き出す（実データは一切含まない）。

中身は `medprep.demo` にある。**教材の同梱版と Colab 版で同じものを使うため、
生成ロジックをここに置かない。**
"""
import sys

sys.path.insert(0, "src")

from medprep import demo  # noqa: E402

path = demo.save("synthetic_dialysis_cohort.xlsx")
df = demo.dialysis_cohort()
print(f"{len(df)} 行 × {df.shape[1]} 列 を {path} に保存")
print(f"真のイベント数: {df.attrs['medprep_demo']['真のイベント数']}"
      "（注入した汚れにより解析対象は減る）")
