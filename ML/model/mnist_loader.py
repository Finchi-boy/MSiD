from ast import TypeAlias
from random import randint
from time import sleep
from turtle import ondrag
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import gzip
from pathlib import Path
import struct


def load_raw_data(filename: str, dir: str ):

    def get_numpy_type(type_id: int) -> np.dtype:
        types = {
            8: np.uint8,
            9: np.int8,
            11: np.int16,
            12: np.int32,
            13: np.float32,
            14: np.float64,
        }
        return types.get(type_id, np.uint8)  # Domyślnie uint8, jeśli kod jest dziwny

    path = Path(Path(dir) / filename)
    with gzip.open(path, "rb") as file:
        # odczytujemy typ danych i liczbe wymiarow:
        magic_bytes = file.read(4)

        # typ danych
        data_type: np.dtype = get_numpy_type(magic_bytes[2])

        # liczba wymiarow
        dimensions_num = magic_bytes[3]

        # wczytywanie rozmiarow dla kazdego wymiaru
        dimensions_sizes: list[int] = [0] * dimensions_num

        for i in range(dimensions_num):
            chunk: bytes = file.read(4)
            dimensions_sizes[i] = struct.unpack(">I", chunk)[0]

        # wczytujemy obiekty
        raw_buffer = file.read()

        flat_view: np.ndarray = np.frombuffer(raw_buffer, dtype=data_type)

        objects = flat_view.reshape(dimensions_sizes)
        return objects

def normalise_data(classes_num:int, images_file, labels_file, data_folder_path = str(Path(__file__).parent.parent / 'data' ),):

    data_files =  ['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz']

    #wczytywanie danych
    raw_images = load_raw_data(images_file, data_folder_path)
    raw_labels = load_raw_data(labels_file, data_folder_path)

    #splaszczanie obrazow
    raw_images = raw_images.reshape(raw_images.shape[0], -1)

    #rzutujemy int(0,255) na float(0,1)
    images = raw_images.astype(np.float32)/255.0

    #tworzymy wektory wynikowe

    #tworzymy macierz 60000 na 10
    labels = np.zeros(shape=(raw_labels.shape[0], classes_num), dtype=np.float32)

    # TODO: sprawdzic jak dokladnie dziala arange
    labels[np.arange(raw_labels.shape[0]), raw_labels]=1.0
    return (images, labels)



