import numpy as np
import pandas as pd
from pathlib import Path

MAP = "RDM"  # or "RDM"
SEEDS = [1, 2, 3]
LAST_K = 5
METHODS = {
    "IQL" : "result/iql/{map}_finalrun_s{seed}/global_data.npy",
    "QMIX": "result/qmix/{map}_qm_s{seed}/global_data.npy",
    "MOD" : "result/qmix/{map}_mod_s{seed}/global_data.npy",
    "FED" : "result/qmix/{map}_fed_s{seed}/global_data.npy",
}

HERE = Path(__file__).resolve().parent
rows = []

for method, tpl in METHODS.items():
    for seed in SEEDS:
        fn = HERE / tpl.format(map=MAP, seed=seed)
        if not fn.exists():
            print(f"⚠ Missing: {fn}")
            continue
        data = np.load(fn)[-LAST_K:]
        row_label = f"{method}-s{seed}"
        row = [round(x, 2) for x in data]
        rows.append((row_label, row))

df = pd.DataFrame.from_dict(dict(rows), orient='index',
    columns=[f"Eval-{i}" for i in range(LAST_K, 0, -1)])
df.index.name = "Method-Seed"

# Save & print
df.to_csv(HERE / f"table_last10_{MAP.lower()}_transposed.csv")
print(df.to_markdown())