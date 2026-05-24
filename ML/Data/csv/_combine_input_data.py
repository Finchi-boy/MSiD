import pandas as pd
import numpy as np
import os

# 1. Definicja plików wejściowych i kolumn, które chcemy z nich wyciągnąć
# Używamy słownika, żeby kod był czysty i łatwy w modyfikacji

year = "11-12"
pliki_i_kolumny = {
    f'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\{year}\\_input_data\\DEMO_G.csv': ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'INDFMPIR'],
    f'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\{year}\\_input_data\\BMX_G.csv': ['SEQN', 'BMXBMI', 'BMXWAIST', 'BMXWT'],
    f'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\{year}\\_input_data\\BPX_G.csv': ['SEQN', 'BPXOSY1', 'BPXODI1', 'BPXOSY2', 'BPXODI2'],
    f'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\{year}\\_input_data\\TCHOL_G.csv': ['SEQN', 'LBXTC'],
    f'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\{year}\\_input_data\\HDL_G.csv': ['SEQN', 'LBDHDD'],
    f'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\{year}\\_input_data\\TRIGLY_G.csv': ['SEQN', 'LBXTLG', 'LBDLDL'],
    f'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\{year}\\_input_data\\PAQ_G.csv': ['SEQN', 'PAD680'],
    f'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\{year}\\_input_data\\SMQ_G.csv': ['SEQN', 'SMQ020', 'SMQ040'],
    f'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\{year}\\_input_data\\SMQFAM_G.csv': ['SEQN', 'SMD460']
}

# Inicjujemy pusty DataFrame, do którego będziemy dołączać kolejne pliki
df_final = None

# 2. Pętla wczytująca i łącząca (Merge) dane po SEQN
print("Wczytywanie i łączenie plików z cechami (Features)...")
for nazwa_pliku, kolumny in pliki_i_kolumny.items():
    sciezka = os.path.join(".", nazwa_pliku)

    try:
        # Wczytujemy tylko zadeklarowane kolumny
        df_temp = pd.read_csv(sciezka, usecols=kolumny)

        if df_final is None:
            df_final = df_temp
        else:
            # how='outer' zapobiega utracie pacjentów z brakującymi badaniami
            df_final = pd.merge(df_final, df_temp, on='SEQN', how='outer')

    except FileNotFoundError:
        print(f"OSTRZEŻENIE: Nie znaleziono pliku {nazwa_pliku}! Pomijam...")
    except ValueError as e:
        print(f"BŁĄD KOLUMN w pliku {nazwa_pliku}: {e}")

# 3. Czyszczenie wartości "odrzutowych" z ankiet (Zamiana na NaN)
print("Czyszczenie wartości błędnych (Odmowa / Nie wiem)...")

# Czyszczenie SMQ020 i SMQ040 (7, 9 -> NaN)
for col in ['SMQ020', 'SMQ040']:
    if col in df_final.columns:
        df_final[col] = df_final[col].replace([7, 9], -1)  # -1 oznacza "Nie wiem / Odrzucam", co może być użyteczne dla ML (nie NaN!)

# Czyszczenie SMD460 (777, 999 -> NaN)
if 'SMD460' in df_final.columns:
    df_final['SMD460'] = df_final['SMD460'].replace([777, 999], -1)  # -1 oznacza "Nie wiem / Odrzucam"

# 4. Feature Engineering: Uśrednianie ciśnienia (Opcjonalne, ale bardzo zalecane dla ML)
if all(c in df_final.columns for c in ['BPXOSY1', 'BPXOSY2']):
    df_final['BPX_SYSTOLIC_AVG'] = df_final[['BPXOSY1', 'BPXOSY2']].mean(axis=1)
    df_final['BPX_DIASTOLIC_AVG'] = df_final[['BPXODI1', 'BPXODI2']].mean(axis=1)
    # Możemy usunąć oryginalne 4 kolumny z ciśnieniem, zostawiając tylko 2 uśrednione
    df_final = df_final.drop(columns=['BPXOSY1', 'BPXOSY2', 'BPXODI1', 'BPXODI2'])

# 5. Dołączenie zmiennej docelowej (Target - plik SICK.csv)
# try:
#     print("Dołączanie etykiet z pliku SICK.csv...")
#     df_sick = pd.read_csv('SICK.csv', usecols=['SEQN', 'DIABETES'])

#     # Dołączamy etykiety cukrzycy (how='inner' - bierzemy tylko tych pacjentów, których potrafimy ocenić!)
#     df_final = pd.merge(df_final, df_sick, on='SEQN', how='inner')
# except FileNotFoundError:
#     print("OSTRZEŻENIE: Nie znaleziono pliku SICK.csv! Skrypt wygeneruje plik bez kolumny docelowej.")

# 6. Zapis gotowego zbioru treningowego
nazwa_wyjsciowa = f'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\{year}\\_input_data\\ML_DIABETES_DATASET.csv'
df_final.to_csv(nazwa_wyjsciowa, index=False)

print(f"\nGotowe! Zapisano plik: {nazwa_wyjsciowa}")
print(f"Liczba pacjentów w zbiorze: {df_final.shape[0]}")
print(f"Liczba cech (kolumn): {df_final.shape[1]}")