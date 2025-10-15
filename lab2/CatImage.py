from typing import Union

import numpy as np

from lab1.implementation import ImageProcessing
from lab1.implementation.custom_image_processing import CustomImageProcessing
from lab1.utils.time_measure import ensure_3_channels


class CatImage:
    """
    Класс для работы с изображениями кошек.

    Инкапсулирует скаченное изображение и его метаданные,
    а также методы обработки изображения.
    """

    def __init__(self, image: np.ndarray, image_url: str, breed: str):
        """
        Инициализация объекта CatImage.

        Args:
            image: numpy-массив с изображением
            image_url: URL изображения
            breed: порода животного
        """
        self._lib_image = None
        self._custom_image = None
        self._image = image
        self._image_url = image_url
        self._breed = breed
        self._lib_image_processor = ImageProcessing()
        self._custom_image_processor = CustomImageProcessing()

    @property
    def image(self) -> np.ndarray:
        """Возвращает изображение как numpy-массив."""
        return self._image

    @property
    def lib_image(self) -> np.ndarray:
        """"""
        return self._lib_image

    @property
    def custom_image(self) -> np.ndarray:
        """"""
        return self._custom_image

    @property
    def image_url(self) -> str:
        """Возвращает URL изображения."""
        return self._image_url

    @property
    def breed(self) -> str:
        """Возвращает породу животного."""
        return self._breed

    def detect_edges_using_library(self) -> np.ndarray:
        """
        Выделение контуров с использованием библиотечного метода

        Returns:
            Изображение с выделенными контурами
        """
        self._lib_image = self._lib_image_processor.edge_detection(self._image)
        return self._lib_image

    def detect_edges_using_custom_method(self) -> np.ndarray:
        """
        Выделение контуров с использованием пользовательского метода (оператор Собеля).

        Returns:
            Изображение с выделенными контурами
        """
        self._custom_image = self._custom_image_processor.edge_detection(self._image)
        return self._custom_image

    def __str__(self) -> str:
        """Строковое представление объекта."""
        return f"CatImage(breed='{self._breed}', url='{self._image_url}', shape={self._image.shape})"

    def __add__(self, other: Union['CatImage', np.ndarray]) -> 'CatImage':
        if isinstance(other, CatImage):
            # Сложение двух CatImage
            other_array = other._image
        elif isinstance(other, np.ndarray):
            # Сложение CatImage с numpy array
            other_array = other
        else:
            raise TypeError(f"Неподдерживаемый тип для сложения: {type(other)}")

        # Проверяем совместимость размеров
        if self._image.shape[:2] != other_array.shape[:2]:
            raise ValueError(
                f"Несовместимые размеры изображений: {self._image.shape} и {other_array.shape}"
            )

        # Преобразуем к float32 для избежания переполнения
        self_3ch = ensure_3_channels(self._image.astype(np.float32))
        other_3ch = ensure_3_channels(other_array.astype(np.float32))

        # Складываем и обрезаем значения
        new_image_float = np.clip(self_3ch + other_3ch, 0, 255)

        # Конвертируем обратно в uint8
        new_image = new_image_float.astype(np.uint8)

        return self.__class__(new_image, image_url=self._image_url, breed=self._breed)

    def __sub__(self, other):
        if not isinstance(other, CatImage):
            new_image = np.clip(self._image.astype(np.int32) - other._image.astype(np.int32),0, 255).astype(np.uint8)

            return self.__class__(new_image, image_url=self._image_url, breed=self._breed)

        raise Exception("Не предназначено для сложения с объектом другого класса")