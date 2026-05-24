import pandas as pd

# ==========================================
# ⚙️ KONFIGURACJA
# ==========================================
PLIK_CECHY = 'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\_ALL_SETS_DATA\\ML_DIABETES_DATASET_FINAL.csv'
PLIK_ETYKIETY = 'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\_ALL_SETS_DATA\\DEF_SICK_ALL.csv'
PLIK_WYJSCIOWY = 'D:\\Users (D)\\amx10\\Desktop (D)\\msid\\MSiD\\ML\\Data\\csv\\_ALL_SETS_DATA\\FINAL_ML_DIABETES_BALANCED.csv'

def stworz_zbalansowany_zbior():
    print("Rozpoczynam tworzenie finalnego, zbalansowanego zbioru...\n")

    # 1. Wczytywanie plików
    try:
        df_cechy = pd.read_csv(PLIK_CECHY)
        # Z pliku etykiet potrzebujemy tylko ID pacjenta i informacji czy jest chory,
        # odrzucamy resztę (LBXGH, GLU itp.), bo nie chcemy ich w ostatecznym modelu
        df_etykiety = pd.read_csv(PLIK_ETYKIETY, usecols=['SEQN', 'DIABETES'])
    except FileNotFoundError as e:
        print(f"BŁĄD: Nie znaleziono pliku. Upewnij się, że pliki istnieją w tym folderze.\nSzczegóły: {e}")
        return

    # 2. Łączenie plików (Inner Merge)
    # Zostawiamy tylko pacjentów, którzy mają zarówno czyste cechy, jak i etykietę DIABETES
    df_merged = pd.merge(df_cechy, df_etykiety, on='SEQN', how='inner')

    print(f"Liczba pacjentów po połączeniu cech i etykiet: {len(df_merged)}")

    # 3. Podział na klasy (Chorzy vs Zdrowi)
    df_chorzy = df_merged[df_merged['DIABETES'] == 1]
    df_zdrowi = df_merged[df_merged['DIABETES'] == 0]

    liczba_chorych = len(df_chorzy)
    liczba_zdrowych = len(df_zdrowi)

    print(f"Dostępni chorzy z pełnymi danymi: {liczba_chorych}")
    print(f"Dostępni zdrowi z pełnymi danymi: {liczba_zdrowych}\n")

    # Zabezpieczenie: sprawdzamy, której grupy jest mniej (zazwyczaj chorych)
    docelowa_liczba = min(liczba_chorych, liczba_zdrowych)

    # 4. Losowanie (Undersampling)
    # Pobieramy wszystkich pacjentów z grupy mniejszej (zazwyczaj chorych)
    df_chorzy_final = df_chorzy.sample(n=docelowa_liczba, random_state=42)
    # Pobieramy losowo TYLE SAMO pacjentów z grupy większej (zazwyczaj zdrowych)
    df_zdrowi_final = df_zdrowi.sample(n=docelowa_liczba, random_state=42)

    # 5. Łączenie w finalny zbiór i przetasowanie (Shuffle)
    # Przetasowanie (frac=1) jest super ważne dla sieci neuronowych, żeby model
    # nie uczył się najpierw samych chorych, a potem samych zdrowych.
    df_final = pd.concat([df_chorzy_final, df_zdrowi_final]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 6. Zapis
    df_final.to_csv(PLIK_WYJSCIOWY, index=False)

    print("-" * 55)
    print("GOTOWE! ZBIÓR ZBALANSOWANY POMYŚLNIE.")
    print(f"Zapisano do pliku: {PLIK_WYJSCIOWY}")
    print(f"Całkowity rozmiar finałowego zbioru: {len(df_final)} wierszy.")
    print("-" * 55)
    print("Rozkład w finalnym pliku:")
    print(f"Chorzy (1): {(df_final['DIABETES'] == 1).sum()}")
    print(f"Zdrowi (0): {(df_final['DIABETES'] == 0).sum()}")

if __name__ == "__main__":
    stworz_zbalansowany_zbior()