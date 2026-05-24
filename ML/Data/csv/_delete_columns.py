import pandas as pd

# ==========================================
# ⚙️ KONFIGURACJA
# ==========================================
PLIK_WEJSCIOWY = 'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\_ALL_SETS_DATA\\ML_DIABETES_DATASET_ALL.csv'
PLIK_WYJSCIOWY = 'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\_ALL_SETS_DATA\\ML_DIABETES_DATASET_FILTERED.csv'

# Wpisz tutaj nazwy kolumn, których chcesz się pozbyć.
# Pamiętaj o cudzysłowach i przecinkach.
KOLUMNY_DO_USUNIECIA = [
    'LBDLDL',
    'LBXTLG',
    'SMQ040'
]

def usun_niepotrzebne_kolumny():
    print(f"Wczytywanie pliku: {PLIK_WEJSCIOWY}...")

    try:
        df = pd.read_csv(PLIK_WEJSCIOWY)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku '{PLIK_WEJSCIOWY}'.")
        return

    liczba_kolumn_start = len(df.columns)
    print(f"Stan początkowy: {liczba_kolumn_start} kolumn.")

    # Usuwanie kolumn.
    # errors='ignore' zapobiega awarii programu, jeśli podanej kolumny nie ma w pliku.
    df = df.drop(columns=KOLUMNY_DO_USUNIECIA, errors='ignore')

    liczba_kolumn_koniec = len(df.columns)
    usunięto = liczba_kolumn_start - liczba_kolumn_koniec

    print(f"Usunięto kolumn: {usunięto}")
    print(f"Stan końcowy: {liczba_kolumn_koniec} kolumn.")

    # Zapisywanie do nowego pliku
    df.to_csv(PLIK_WYJSCIOWY, index=False)
    print(f"\nGotowe! Zapisano odchudzony plik jako: {PLIK_WYJSCIOWY}")

if __name__ == "__main__":
    usun_niepotrzebne_kolumny()