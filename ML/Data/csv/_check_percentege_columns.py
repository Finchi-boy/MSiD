import pandas as pd
import numpy as np
import os

PLIK_CECHY = f'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\_ALL_SETS_DATA\\ML_DIABETES_DATASET_FINAL.csv'
PLIK_ETYKIETY = f'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\_ALL_SETS_DATA\\DEF_SICK_ALL.csv'

def sprawdz_wypelnienie_osobne_pliki():
    print(f"Wczytywanie plików...\nCechy: {PLIK_CECHY}\nEtykiety: {PLIK_ETYKIETY}\n")

    if not os.path.exists(PLIK_CECHY):
        print(f"Błąd: Nie znaleziono pliku cech '{PLIK_CECHY}'.")
        return
    if not os.path.exists(PLIK_ETYKIETY):
        print(f"Błąd: Nie znaleziono pliku etykiet '{PLIK_ETYKIETY}'.")
        return

    # Wczytywanie obu plików
    df_cechy = pd.read_csv(PLIK_CECHY)
    df_etykiety = pd.read_csv(PLIK_ETYKIETY)

    if 'SEQN' not in df_cechy.columns or 'SEQN' not in df_etykiety.columns:
        print("Błąd: W obu plikach musi znajdować się kolumna 'SEQN' do połączenia pacjentów.")
        return

    if 'DIABETES' not in df_etykiety.columns:
        print("Błąd: W pliku etykiet brakuje kolumny 'DIABETES'.")
        return

    # Łączymy pliki po SEQN.
    # how='inner' oznacza, że analizujemy tylko tych pacjentów, którzy występują w OBU plikach.
    df_merged = pd.merge(df_cechy, df_etykiety[['SEQN', 'DIABETES']], on='SEQN', how='inner')

    liczba_wierszy = len(df_merged)
    print(f"Połączono pomyślnie. Liczba pacjentów do analizy: {liczba_wierszy}\n")

    # Zamiana ewentualnych -1 na NaN
    # df_merged = df_merged.replace(-1, np.nan)

    dane_do_raportu = []

    # Iterujemy tylko po kolumnach z pliku cech (pomijamy SEQN, bo to ID)
    kolumny_do_sprawdzenia = [kol for kol in df_cechy.columns if kol != 'SEQN']

    for kolumna in kolumny_do_sprawdzenia:
        # Maska logiczna: True tam, gdzie jest jakaś wartość, False tam, gdzie NaN
        wypelnione_maska = df_merged[kolumna].notna()

        wypelnione = wypelnione_maska.sum()
        braki = liczba_wierszy - wypelnione

        # Zapobieganie błędom dzielenia przez zero, jeśli plik byłby pusty
        procent = (wypelnione / liczba_wierszy * 100) if liczba_wierszy > 0 else 0

        # Zliczamy sumę kolumny DIABETES tylko dla wierszy, które mają wypełnioną daną kolumnę
        chorzy = df_merged.loc[wypelnione_maska, 'DIABETES'].sum()

        dane_do_raportu.append({
            'Kolumna': kolumna,
            'Wypełnione': wypelnione,
            'Braki': braki,
            'Wypełnienie (%)': procent,
            'Z Cukrzycą': int(chorzy)
        })

    # Tworzenie tabeli i sortowanie
    raport = pd.DataFrame(dane_do_raportu)
    raport = raport.sort_values(by='Wypełnienie (%)', ascending=False).reset_index(drop=True)

    # Wyświetlanie raportu
    print("-" * 75)
    header = f"{'KOLUMNA':<20} | {'WYPEŁNIONE':<12} | {'BRAKI':<8} | {'% WYPEŁ.':<9} | {'Z CUKRZYCĄ'}"
    print(header)
    print("-" * 75)

    for _, wiersz in raport.iterrows():
        kol = wiersz['Kolumna']
        wyp = int(wiersz['Wypełnione'])
        braki = int(wiersz['Braki'])
        proc = wiersz['Wypełnienie (%)']
        chorzy_wyp = int(wiersz['Z Cukrzycą'])

        print(f"{kol:<20} | {wyp:<12} | {braki:<8} | {proc:>8.2f}% | {chorzy_wyp}")

    print("-" * 75)

if __name__ == "__main__":
    sprawdz_wypelnienie_osobne_pliki()