import os
import glob
import pandas as pd

# ==========================================
# ⚙️ KONFIGURACJA
# ==========================================
# Folder, w którym znajdują się małe pliki CSV (zgodnie ze zdjęciem)
# FOLDER_WEJSCIOWY = 'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\_ALL_SETS_DATA\\_def_sick'
FOLDER_WEJSCIOWY = 'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\_ALL_SETS_DATA\\_input_data'

# Nazwa głównego, połączonego pliku, który powstanie
PLIK_WYJSCIOWY = 'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\_ALL_SETS_DATA\\_input_data\\ML_DIABETES_DATASET_ALL.csv'

def polacz_pliki_z_diagnoza():
    print(f"Szukam plików CSV w folderze '{FOLDER_WEJSCIOWY}'...\n")

    # Wyszukiwanie wszystkich plików .csv w podanym folderze
    sciezka_szukania = os.path.join(FOLDER_WEJSCIOWY, '*.csv')
    pliki_csv = glob.glob(sciezka_szukania)

    # Sortujemy alfabetycznie (żeby lata szły po kolei: 11_12, 13_14 itd.)
    pliki_csv.sort()

    if not pliki_csv:
        print(f"Błąd: Nie znaleziono żadnych plików CSV w folderze {FOLDER_WEJSCIOWY}.")
        return

    lista_tabel = []
    laczna_liczba_pacjentow = 0

    # Wczytywanie każdego pliku
    for plik in pliki_csv:
        try:
            df = pd.read_csv(plik)
            liczba_wierszy = len(df)
            print(f"Wczytano: {os.path.basename(plik):<25} | Pacjentów: {liczba_wierszy}")

            lista_tabel.append(df)
            laczna_liczba_pacjentow += liczba_wierszy
        except Exception as e:
            print(f"Błąd przy wczytywaniu pliku {plik}: {e}")

    # Łączenie (konkatenacja) wszystkich tabel w jedną dużą
    # ignore_index=True sprawia, że numeracja wierszy utworzy się od nowa, ładnie od 0
    df_final = pd.concat(lista_tabel, ignore_index=True)

    # Zapisujemy do nowego pliku
    df_final.to_csv(PLIK_WYJSCIOWY, index=False)

    print("-" * 55)
    print("ŁĄCZENIE ZAKOŃCZONE SUKCESEM!")
    print(f"Zapisano połączony plik jako: {PLIK_WYJSCIOWY}")

    # Wyświetlamy statystyki, żebyś widział, czym teraz dysponuje model
    print("\n📊 STATYSTYKI ZBIORCZE:")
    print(f"Suma wszystkich pacjentów: {len(df_final)}")

    if 'DIABETES' in df_final.columns:
        chorzy = (df_final['DIABETES'] == 1).sum()
        zdrowi = (df_final['DIABETES'] == 0).sum()
        print(f"Chorzy (1): {chorzy} ({(chorzy/len(df_final))*100:.1f}%)")
        print(f"Zdrowi (0): {zdrowi} ({(zdrowi/len(df_final))*100:.1f}%)")
    else:
        print("Brak kolumny DIABETES do podsumowania.")

if __name__ == "__main__":
    polacz_pliki_z_diagnoza()