import requests, io, pandas as pd
from functools import reduce


def fetch_xpt(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return pd.read_sas(io.BytesIO(r.content), format="xport")


# Nowa struktura URL: /Nchs/Data/Nhanes/Public/{rok}/DataFiles/{plik}.XPT
FILES = {
    "2017": {
        "diabetes": "DIQ_J",
        "demo": "DEMO_J",
        "bmi": "BMX_J",
        "glucose": "GLU_J",
        "ghb": "GHB_J",
        "chol": "TCHOL_J",
        "hdl": "HDL_J",
        "bp": "BPX_J",
        "sleep": "SLQ_J",
        "occupation": "OCQ_J",
    },
    "2015": {
        "diabetes": "DIQ_I",
        "demo": "DEMO_I",
        "bmi": "BMX_I",
        "glucose": "GLU_I",
        "ghb": "GHB_I",
        "chol": "TCHOL_I",
        "hdl": "HDL_I",
        "bp": "BPX_I",
        "sleep": "SLQ_I",
        "occupation": "OCQ_I",
    },
    "2013": {
        "diabetes": "DIQ_H",
        "demo": "DEMO_H",
        "bmi": "BMX_H",
        "glucose": "GLU_H",
        "ghb": "GHB_H",
        "chol": "TCHOL_H",
        "hdl": "HDL_H",
        "bp": "BPX_H",
        "sleep": "SLQ_H",
        "occupation": "OCQ_H",
    },
}

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{file}.XPT"

frames = []
for year, files in FILES.items():
    parts = {}
    for name, code in files.items():
        url = BASE.format(year=year, file=code)
        try:
            parts[name] = fetch_xpt(url)
            print(f"OK  {year} {name}: {parts[name].shape}")
        except Exception as e:
            print(f"ERR {year} {name}: {e}")

    if "diabetes" not in parts:
        continue

    df = parts["diabetes"][["SEQN", "DIQ010"]].copy()
    for name, part in parts.items():
        if name == "diabetes":
            continue
        df = df.merge(part, on="SEQN", how="left")
    frames.append(df)

full = pd.concat(frames, ignore_index=True)
full = full[full["DIQ010"].isin([1, 2])]
full["diabetes"] = (full["DIQ010"] == 1).astype(int)
full = full.drop(columns=["DIQ010"])

print(f"\nFinalna próbka: {full.shape}")
print(f"Z cukrzycą:     {full['diabetes'].sum()}")
print(f"Bez cukrzycy:   {(full['diabetes'] == 0).sum()}")

KEEP = [
    "diabetes",
    # istotne (bez LBXGLU)
    "LBXGH",  # HbA1c - zastępuje glukozę
    "BMXBMI",
    "BMXWAIST",
    "RIDAGEYR",
    "LBXTC",
    "LBDHDD",
    "BPXSY1",
    "BPXDI1",
    "RIAGENDR",
    # nieistotne
    "RIDEXMON",
    "DMDEDUC2",
    "DMDHHSIZ",
]

df = full[[c for c in KEEP if c in full.columns]].copy()

for col in df.columns:
    non_null = df.dropna(subset=[col]).shape[0]
    print(f"{col:15} {non_null:6} ({non_null / len(df) * 100:.1f}%)")


df_clean = df.dropna()

print(df_clean.groupby("diabetes")[["RIDEXMON", "DMDEDUC2", "DMDHHSIZ"]].mean())

print(f"Rekordów: {df_clean.shape}")
print(f"Z cukrzycą: {df_clean['diabetes'].sum()}")
print(f"Bez cukrzycy: {(df_clean['diabetes'] == 0).sum()}")


print(f"Przed: {df.shape}")
print(f"Po:    {df_clean.shape}")
print(f"Usunięto: {len(df) - len(df_clean)} wierszy")

print(df.shape)

df_clean.to_csv("data/nhanes_merged.csv", index=False)
