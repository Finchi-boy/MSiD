import os
import glob
import pandas as pd

# ==========================================
# ⚙️ KONFIGURACJA
# ==========================================
# Folder, w którym znajdują się pliki CSV do połączenia
FOLDER_WEJSCIOWY = 'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\_ALL_SETS_DATA\\_input_data'

# Nazwa głównego, połączonego pliku, który powstanie
PLIK_WYJSCIOWY = 'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\_ALL_SETS_DATA\\_input_data\\ML_DIABETES_DATASET_ALL.csv'

def polacz_pliki_wejsciowe():
    print(f"Szukam plików CSV w folderze '{FOLDER_WEJSCIOWY}'...\n")

    # Wyszukiwanie wszystkich plików .csv w podanym folderze
    sciezka_szukania = os.path.join(FOLDER_WEJSCIOWY, '*.csv')
    pliki_csv = glob.glob(sciezka_szukania)

    # Sortujemy alfabetycznie, żeby zachować chronologię lat
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
            liczba_kolumn = len(df.columns)

            # Wypisujemy nazwę pliku, liczbę wierszy i kolumn
            nazwa_pliku = os.path.basename(plik)
            print(f"Wczytano: {nazwa_pliku:<35} | Pacjentów: {liczba_wierszy} | Kolumn: {liczba_kolumn}")

            lista_tabel.append(df)
            laczna_liczba_pacjentow += liczba_wierszy
        except Exception as e:
            print(f"Błąd przy wczytywaniu pliku {plik}: {e}")

    # Łączenie (konkatenacja) wszystkich tabel w jedną
    # ignore_index=True nadaje nową, ciągłą numerację wierszy
    print("\nTrwa łączenie danych...")
    df_final = pd.concat(lista_tabel, ignore_index=True)

    # Zapisujemy do nowego pliku
    df_final.to_csv(PLIK_WYJSCIOWY, index=False)

    print("-" * 65)
    print("ŁĄCZENIE ZAKOŃCZONE SUKCESEM!")
    print(f"Zapisano połączony plik jako: {PLIK_WYJSCIOWY}")

    # Wyświetlamy ostateczne statystyki
    print("\n📊 STATYSTYKI ZBIORCZE:")
    print(f"Suma wszystkich pacjentów (wierszy): {len(df_final)}")
    print(f"Liczba cech (kolumn): {len(df_final.columns)}")

    # Wypisujemy listę kolumn, żebyś mógł rzucić okiem, czy wszystko jest na miejscu
    print("\nLista kolumn w finalnym pliku:")
    print(", ".join(df_final.columns.tolist()))

if __name__ == "__main__":
    polacz_pliki_wejsciowe()