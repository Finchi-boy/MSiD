import pandas as pd

# 1. Wczytanie plików
# (Upewnij się, że ścieżki zgadzają się z układem folderów na Twoim dysku)

year = "11-12"
try:
    df_diq = pd.read_csv(f"D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\{year}\\_def_sick\\DIQ_G.csv")
    df_ghb = pd.read_csv(f"D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\{year}\\_def_sick\\GHB_G.csv")
    df_glu = pd.read_csv(f"D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\{year}\\_def_sick\\GLU_G.csv")
except FileNotFoundError as e:
    print(f"Błąd ścieżki: {e}. Upewnij się, że uruchamiasz skrypt z dobrego folderu.")
    exit()

# 2. Wycięcie tylko potrzebnych kolumn, żeby plik SICK.csv był czytelny
df_diq_subset = df_diq[['SEQN', 'DIQ010']]
df_ghb_subset = df_ghb[['SEQN', 'LBXGH']]
df_glu_subset = df_glu[['SEQN', 'LBXGLU']]

# 3. Łączenie tabel po unikalnym ID pacjenta (SEQN)
# how='outer' gwarantuje, że nie zgubimy pacjenta, jeśli brakuje mu jakiegoś badania
df_merged = df_diq_subset.merge(df_ghb_subset, on='SEQN', how='outer')
df_merged = df_merged.merge(df_glu_subset, on='SEQN', how='outer')

# 4. Aplikowanie reguł diagnostycznych
# (Warunki zwracają True lub False. Jeśli jest NaN, warunek zwraca False)
cond_ankiet = df_merged['DIQ010'] == 1.0
cond_hba1c = df_merged['LBXGH'] >= 6.5
cond_glukoza = df_merged['LBXGLU'] >= 126.0

# 5. Tworzenie kolumny docelowej
# Znak '|' oznacza logiczne "LUB". astype(int) zamienia True/False na 1/0.
df_merged['DIABETES'] = (cond_ankiet | cond_hba1c | cond_glukoza).astype(int)

# 6. Oczyszczanie zbioru (Opcjonalne)
# Usuwamy wiersze, gdzie pacjent nie ma absolutnie żadnej z tych trzech danych
# df_merged = df_merged.dropna(subset=['DIQ010', 'LBXGH', 'LBXGLU'], how='all')

# 7. Zapis do pliku
df_merged.to_csv(f'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\{year}\\_def_sick\\DEF_SICK.csv', index=False)

# Raport w terminalu
print("Pomyślnie utworzono plik DEF_SICK.csv!")
print("\nStatystyki klas:")
print(f"Chorzy (1): {(df_merged['DIABETES'] == 1).sum()}")
print(f"Zdrowi (0): {(df_merged['DIABETES'] == 0).sum()}")