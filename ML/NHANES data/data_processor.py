from pathlib import Path
from typing import Self

import requests, io, pandas as pd


class DataProcessor:
    def __init__(self):

        self._data: pd.DataFrame = pd.DataFrame()

    def get_data(self) -> pd.DataFrame:
        if self._data.empty:
            raise Exception(
                "No data downloaded yet. Please call download_data() first."
            )
        return self._data.copy()

    def _fetch_data(self, url: str) -> pd.DataFrame:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return pd.read_sas(io.BytesIO(r.content), format="xport")

    def download_data(self, url: str, files: dict[str, dict[str, str]]) -> pd.DataFrame:
        frames = []
        for year, file_dict in files.items():
            parts: dict[str, pd.DataFrame] = {}
            for name, code in file_dict.items():
                temp_url = url.format(year=year, file=code)
                try:
                    parts[name] = self._fetch_data(temp_url)
                    print(f"OK  {year} {name}: {parts[name].shape}")
                except Exception as e:
                    print(f"ERR {year} {name}: {e}")

            if "diabetes" not in parts:
                continue

            df: pd.DataFrame = parts["diabetes"][["SEQN", "DIQ010"]].copy()
            for name, part in parts.items():
                if name == "diabetes":
                    continue

                print(
                    f"Merging {year} {name} with shape {part.shape} into main dataframe with shape {df.shape}"
                )

                df = df.merge(part, on="SEQN", how="left")

            frames.append(df.drop_duplicates(subset=["SEQN"]).copy())

        return pd.concat(frames, ignore_index=True)

    def filter_data(
        self,
        fields_to_keep: dict[str, str],
        glucose_threshold: float = 6.5,
    ) -> Self:
        if self._data.empty:
            raise Exception("No data to filter. Please call download_data() first.")
        self._data.loc[self._data["DIQ010"] == 2, "DIQ010"] = 0
        print(f"Przed: {self._data.shape}")
        self._remove_unwanted_columns(fields_to_keep)
        print(f"Po:    {self._data.shape}")
        self._remove_invalid_diabetes_status()
        self._remove_missing_values()
        self._remove_high_glucose_level(glucose_threshold)
        self.rename_columns(fields_to_keep)
        return self

    def _remove_unwanted_columns(self, fields_to_keep: dict[str, str]) -> Self:
        self._data = self._data[
            [
                f"{field}"
                for field in fields_to_keep.keys()
                if field in self._data.columns
            ]
        ]

        if set(self._data.columns) != set(["SEQN"] + list(fields_to_keep.keys())):
            missing = set(fields_to_keep.keys()) - set(self._data.columns)
            print(
                f"Warning: The following fields were not found in the data and will be skipped: {missing}"
            )
        return self

    def _remove_high_glucose_level(self, threshold: float = 6.5) -> Self:
        self._data = self._data[
            ~((self._data["DIQ010"] == 0) & (self._data["LBXGH"] > threshold))
        ]
        return self

    def _remove_missing_values(self) -> Self:
        self._data = self._data.dropna()
        return self

    def _remove_invalid_diabetes_status(self) -> Self:
        self._data = self._data[self._data["DIQ010"].isin([0, 1])]
        return self

    def rename_columns(self, fields_to_keep: dict[str, str]) -> Self:
        self._data = self._data.rename(
            columns={
                f"{field}": f"{field} - {desc}"
                for field, desc in fields_to_keep.items()
            }
        )
        return self

    def to_csv(self, filename: str | Path = "data/nhanes_mergedv2.csv"):
        if self._data.empty:
            raise Exception("No data to save. Please call download_data() first.")
        self._data.to_csv(filename, index=False)

    def run(
        self,
        url: str,
        files: dict[str, dict[str, str]],
        fields_to_keep: dict[str, str],
        filename: str | Path = "nhanes_mergedv2.csv",
        glucose_threshold: float = 6.5,
    ):
        self._data = self.download_data(url, files)
        print(self._data.shape)
        self.filter_data(fields_to_keep, glucose_threshold)
        self.to_csv(filename)
