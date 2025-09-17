"""
Модуль вспомогательных функций для обработки изображений.

Содержит утилиты для работы с ядрами свертки и другими вспомогательными операциями.
"""

from typing import Tuple

import numpy as np


def parse_kernel(kernel_str: str) -> np.ndarray:
    """
    Парсит строку с ядром свертки в массив NumPy.

    Args:
        kernel_str: Строка с значениями ядра, разделенными запятыми

    Returns:
        Ядро свертки как 2D массив NumPy

    Raises:
        ValueError: Если ядро не является квадратным
    """
    values = [float(x) for x in kernel_str.split(',')]
    size = int(len(values) ** 0.5)
    if size * size != len(values):
        raise ValueError("Kernel must be square")

    return np.array(values).reshape((size, size))


def get_sobel_kernels() -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает ядра Собеля для обнаружения границ.

    Returns:
        Кортеж с x и y ядрами Собеля
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    return sobel_x, sobel_y
