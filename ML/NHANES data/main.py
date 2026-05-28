from data_processor import DataProcessor
from constants import BASE_URL, FILES, KEEP


if __name__ == "__main__":
    dd = DataProcessor()
    dd.run(BASE_URL, FILES, KEEP, "data/nhanes_mergedv3.csv")
    print(dd._data)
