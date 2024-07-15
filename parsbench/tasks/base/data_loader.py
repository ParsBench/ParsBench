import csv
from abc import ABC, abstractmethod
from typing import Any, Callable

import datasets
import jsonlines
import requests
from tqdm import tqdm


def _fetch_text_file(path) -> str:
    if path.startswith("http"):
        headers = {
            "Accept-Encoding": "identity",
            "Accept": "*/*",
        }
        response = requests.get(path, headers=headers, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024
        content = bytearray()

        with tqdm(
            desc="Downloading data",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=chunk_size,
        ) as bar:
            for data in response.iter_content(chunk_size):
                content.extend(data)
                bar.update(len(data))

        assert len(content) == total_size, f"{len(content)} != {total_size}"

        return content.decode()

    with open(path, encoding="utf-8") as f:
        content = f.read()
        return content


class DataLoader(ABC):
    """
    An abstract base class for defining data loaders.

    Attributes:
        data_path (str): The path to the data source.

    Methods:
        load(self) -> list[dict]: Abstract method to be implemented by subclasses for loading data.
    """

    def __init__(self, data_path: str, **kwargs) -> None:
        self.data_path = data_path

    @abstractmethod
    def load(self) -> list[dict]:
        pass


class JSONLineDataLoader(DataLoader):
    """
    A data loader class for loading JSON line data from either a local file or a URL.

    Attributes:
        data_path (str): The path to the JSON line data source.

    Methods:
        load(self) -> list[dict]: Loads the JSON line data from the specified source.
    """

    def load(self) -> list[dict]:
        content = _fetch_text_file(self.data_path)

        reader = jsonlines.Reader(content.split("\n"))
        return list(reader.iter(type=dict, skip_invalid=True, skip_empty=True))


class CSVDataLoader(DataLoader):
    """
    A data loader class for loading CSV line data from either a local file or a URL.

    Attributes:
        data_path (str): The path to the CSV line data source.

    Methods:
        load(self) -> list[dict]: Loads the CSV line data from the specified source.
    """

    def __init__(self, data_path: str, csv_arguments: dict | None = None, **kwargs):
        super().__init__(data_path)
        self.csv_arguments = csv_arguments or {}

    def load(self) -> list[dict]:
        content = _fetch_text_file(self.data_path)

        csv_reader = csv.DictReader(content.split("\n"), **self.csv_arguments)
        return list(csv_reader)


class HuggingFaceDataLoader(DataLoader):
    """
    A data loader class for loading datasets using the Hugging Face library.

    Attributes:
        data_path (str): The path to the data source.
        split (str | None): The split of the dataset to load.

    Methods:
        load(self) -> list[dict]: Loads the dataset from the specified data path and split.
        with_filter(self, func: Callable[..., bool]) -> "HuggingFaceDataLoader": Adds a filter function to apply when loading the dataset.
    """

    def __init__(
        self,
        data_path: str,
        split: str | None = None,
        **optional_parameters: dict[str, Any],
    ) -> None:
        super().__init__(data_path)
        self.split = split
        self.optional_parameters = optional_parameters
        self._filters = []

    def load(self) -> list[dict]:
        dataset = datasets.load_dataset(
            self.data_path, split=self.split, **self.optional_parameters
        )
        if len(self._filters):
            for filter_ in self._filters:
                dataset = dataset.filter(filter_)
        return dataset.to_list()

    def with_filter(self, func: Callable[..., bool]) -> "HuggingFaceDataLoader":
        self._filters.append(func)
        return self
