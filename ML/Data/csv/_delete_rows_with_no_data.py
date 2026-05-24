import pandas as pd

# ==========================================
# ⚙️ KONFIGURACJA
# ==========================================
# Jako wejście podajemy odchudzony plik z poprzedniego kroku
PLIK_WEJSCIOWY = 'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\_ALL_SETS_DATA\\ML_DIABETES_DATASET_FILTERED.csv'

# Plik wyjściowy, który będzie już w 100% gotowy do uczenia maszynowego
PLIK_WYJSCIOWY = 'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\_ALL_SETS_DATA\\ML_DIABETES_DATASET_FINAL.csv'

def usun_wiersze_z_nan():
    print(f"Wczytywanie pliku: {PLIK_WEJSCIOWY}...")

    try:
        df = pd.read_csv(PLIK_WEJSCIOWY)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku '{PLIK_WEJSCIOWY}'.")
        return

    liczba_wierszy_start = len(df)
    print(f"Stan początkowy: {liczba_wierszy_start} pacjentów.")

    # Główne czyszczenie.
    # Funkcja dropna() domyślnie usuwa wiersz, jeśli gdziekolwiek (how='any') występuje wartość NaN.
    df_czyste = df.dropna()

    liczba_wierszy_koniec = len(df_czyste)
    usunieto = liczba_wierszy_start - liczba_wierszy_koniec

    # Zabezpieczenie przed błędem dzielenia przez zero
    procent_zostal = (liczba_wierszy_koniec / liczba_wierszy_start) * 100 if liczba_wierszy_start > 0 else 0

    print("-" * 50)
    print(f"Usunięto pacjentów (wierszy): {usunieto}")
    print(f"Stan końcowy: {liczba_wierszy_koniec} pacjentów ({procent_zostal:.1f}% oryginału).")
    print("-" * 50)

    if liczba_wierszy_koniec == 0:
        print("\n⚠️ KRYTYCZNE OSTRZEŻENIE: Usunięto WSZYSTKICH pacjentów!")
        print("Prawdopodobnie każda osoba w bazie miała brak w chociaż jednym badaniu.")
        print("Rozwiązanie: Musisz usunąć więcej 'dziurawych' kolumn w poprzednim skrypcie.")
        return

    # Zapisywanie do nowego pliku
    df_czyste.to_csv(PLIK_WYJSCIOWY, index=False)
    print(f"\nGotowe! Zapisano idealnie czysty plik jako: {PLIK_WYJSCIOWY}")

if __name__ == "__main__":
    usun_wiersze_z_nan()