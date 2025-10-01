"""
Модуль для обработки изображений с использованием OpenCV и NumPy.

Предоставляет класс ImageProcessor выполнения различных операций обработки изображений:
- свёртка изображения с произвольным ядром
- преобразование RGB-изображения в оттенки серого
- гамма-коррекция
- обнаружение границ (оператор Собеля)
- обнаружение углов (детектор Харриса)
- обнаружение окружностей (преобразование Хафа - не реализовано)

Все операции выполняются с использованием NumPy без готовых функций OpenCV
для основных алгоритмов обработки.
"""

import time
from typing import Any, Callable, NoReturn, Optional

import cv2

import numpy as np

from src.image_processor.utils import get_sobel_kernels


class ImageProcessor:
    """
    Класс для обработки изображений с использованием OpenCV и NumPy.

    Предоставляет методы для выполнения различных операций обработки изображений:
    - свёртка изображения с произвольным ядром
    - преобразование RGB-изображения в оттенки серого
    - гамма-коррекция
    - обнаружение границ (оператор Собеля)
    - обнаружение углов (детектор Харриса)
    - обнаружение окружностей (преобразование Хафа - не реализовано)

    Все операции выполняются с использованием NumPy без готовых функций OpenCV
    для основных алгоритмов обработки.
    """

    def __init__(self: 'ImageProcessor', image_path: str) -> None:
        """
        Инициализация процессора изображений.

        Args:
            image_path: Путь к исходному изображению

        Raises:
            ValueError: Если изображение не может быть загружено
        """
        # Загружаем изображение с помощью OpenCV
        self.image: np.ndarray = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        # Конвертируем BGR в RGB (OpenCV использует BGR по умолчанию)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.result_image: Optional[np.ndarray] = None
        self.execution_time: float = 0.0

    def measure_time(
            self: 'ImageProcessor',
            func: Callable[..., Any],
            *args: Any,
            **kwargs: Any,
    ) -> Any:
        """
        Измеряет время выполнения функции.

        Args:
            func: Функция для выполнения и измерения времени
            *args: Аргументы функции
            **kwargs: Ключевые аргументы функции

        Returns:
            Результат выполнения функции
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        self.execution_time = end_time - start_time
        return result

    def _rgb_to_grayscale(self: 'ImageProcessor', image: np.ndarray) -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.

        Использует стандартную формулу: Y = 0.299*R + 0.587*G + 0.114*B

        Args:
            image: Входное RGB-изображение

        Returns:
            Одноканальное изображение в оттенках серого
        """
        return np.clip(0.299 * image[:, :, 0] + # - ограничивает значения в диапазоне [0, 255]
                       0.587 * image[:, :, 1] +
                       0.114 * image[:, :, 2], 0, 255).astype(np.uint8)

    def _convolution2d(
            self: 'ImageProcessor',
            image: np.ndarray,
            kernel: np.ndarray,
    ) -> np.ndarray:
        """
        Выполняет 2D свертку изображения с ядром.

        Args:
            image: Входное изображение (2D массив)
            kernel: Ядро свертки (2D массив)

        Returns:
            Результат свертки
        """
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2

        # Добавляем padding
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect') # добавляем края (зеркалим)

        # Создаем выходной массив
        output = np.zeros_like(image)

        # Применяем свертку
        for i in range(image.shape[0]):
            for col in range(image.shape[1]):
                region = padded[i:i + k_h, col:col + k_w]
                output[i, col] = np.sum(region * kernel)

        return output

    def to_grayscale(self: 'ImageProcessor') -> None:
        """
        Преобразует цветное изображение в полутоновое.

        Время выполнения операции измеряется и выводится в консоль.
        """
        def _convert() -> None:
            """Внутренняя функция для преобразования в оттенки серого."""
            if len(self.image.shape) == 3:
                self.result_image = self._rgb_to_grayscale(self.image)
            else:
                # Изображение уже в оттенках серого
                self.result_image = self.image.copy()

        self.measure_time(_convert)
        print(f"Grayscale conversion took {self.execution_time:.4f} seconds")

    def gamma_correction(self: 'ImageProcessor', gamma: float) -> None:
        """
        Применяет гамма-коррекцию к изображению.

        Args:
            gamma: Значение гаммы (>1 - осветление, <1 - затемнение)

        Raises:
            ValueError: Если gamma <= 0

        Время выполнения операции измеряется и выводится в консоль.
        """
        if gamma <= 0:
            raise ValueError("Gamma must be greater than 0")

        def _correct() -> None:
            """Внутренняя функция для применения гамма-коррекции."""
            # Нормализуем изображение к диапазону 0-1
            img_float = self.image.astype(np.float32) / 255.0

            # Применяем гамма-коррекцию
            corrected = np.power(img_float, 1.0 / gamma)

            # Масштабируем обратно к диапазону 0-255
            self.result_image = (corrected * 255).astype(np.uint8)

        self.measure_time(_correct)
        print(f"Gamma correction took {self.execution_time:.4f} seconds")

    def apply_convolution(self: 'ImageProcessor', kernel: np.ndarray) -> None:
        """
        Применяет свертку с заданным ядром.

        Args:
            kernel: Ядро свертки (2D массив NumPy)

        Время выполнения операции измеряется и выводится в консоль.
        """
        def _convolve() -> None:
            """Внутренняя функция для применения свертки."""
            # Получаем размеры изображения и ядра
            img_h, img_w = self.image.shape[:2]
            k_h, k_w = kernel.shape

            # Вычисляем padding
            pad_h, pad_w = k_h // 2, k_w // 2

            # Создаем выходной массив
            if len(self.image.shape) == 3:
                output = np.zeros_like(self.image)
            else:
                output = np.zeros((img_h, img_w), dtype=np.float32)

            # Добавляем padding к изображению
            if len(self.image.shape) == 3:
                padded = np.pad(
                    self.image,
                    ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                    mode='reflect',
                )
            else:
                padded = np.pad(
                    self.image,
                    ((pad_h, pad_h), (pad_w, pad_w)),
                    mode='reflect',
                )

            # Применяем свертку
            for i in range(img_h):
                for col in range(img_w):
                    if len(self.image.shape) == 3:
                        for channel in range(3):  # Для каждого канала (R, G, B)
                            region = padded[i:i+k_h, col:col+k_w, channel]
                            output[i, col, channel] = np.sum(region * kernel)
                    else:
                        region = padded[i:i+k_h, col:col+k_w]
                        output[i, col] = np.sum(region * kernel)

            # Обрезаем значения до диапазона 0-255
            output = np.clip(output, 0, 255).astype(np.uint8)
            self.result_image = output

        self.measure_time(_convolve)
        print(f"Convolution took {self.execution_time:.4f} seconds")

    def sobel_edge_detection(self: 'ImageProcessor') -> None:
        """
        Выделяет границы с помощью оператора Собеля.

        Время выполнения операции измеряется и выводится в консоль.
        """
        def _detect_edges() -> None:
            """Внутренняя функция для обнаружения границ Собелем."""
            # Преобразуем в grayscale, если нужно
            if len(self.image.shape) == 3:
                gray = self._rgb_to_grayscale(self.image)
            else:
                gray = self.image.copy()

            gray_float = gray.astype(np.float32)

            # Операторы Собеля
            sobel_x, sobel_y = get_sobel_kernels()

            # Применяем свертки
            grad_x = self._convolution2d(gray_float, sobel_x)
            grad_y = self._convolution2d(gray_float, sobel_y)

            # Вычисляем магнитуду градиента
            magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Нормализуем к диапазону 0-255
            if magnitude.max() > 0:
                magnitude = (magnitude / magnitude.max()) * 255

            self.result_image = magnitude.astype(np.uint8)

        self.measure_time(_detect_edges)
        print(f"Sobel edge detection took {self.execution_time:.4f} seconds")

    def harris_corner_detection(self: 'ImageProcessor') -> None:
        """
        Выполняет обнаружение углов на изображении с помощью детектора Харриса.

        Время выполнения операции измеряется и выводится в консоль.
        """
        def _detect_corners() -> None:
            """Внутренняя функция для обнаружения углов Харрисом."""
            # 1. Конвертируем в оттенки серого
            if len(self.image.shape) == 3:
                gray = self._rgb_to_grayscale(self.image)
            else:
                gray = self.image

            # 2. Вычисляем производные
            sobel_x = np.array(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=np.float32,
            )
            sobel_y = np.array(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=np.float32,
            )

            ix = self._convolution2d(gray, sobel_x)
            iy = self._convolution2d(gray, sobel_y)

            # 3. Вычисляем элементы матрицы структуры
            ix2 = ix * ix
            iy2 = iy * iy
            ixy = ix * iy

            # 4. Гауссово сглаживание
            gaussian_kernel = np.array(
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                dtype=np.float32,
            ) / 16.0

            sx2 = self._convolution2d(ix2, gaussian_kernel)
            sy2 = self._convolution2d(iy2, gaussian_kernel)
            sxy = self._convolution2d(ixy, gaussian_kernel)

            # 5. Вычисляем отклик Харриса
            k_value = 0.04
            det = sx2 * sy2 - sxy * sxy
            trace = sx2 + sy2
            response = det - k_value * trace * trace

            # 6. Нормализация
            response_min = np.min(response)
            response_max = np.max(response)
            response_norm = (response - response_min) / (response_max - response_min)

            # 7. Адаптивный подбор порога
            def find_adaptive_threshold(
                    r_norm: np.ndarray,
                    target_corners: int = 1000,
            ) -> float:
                """
                Автоматически подбирает порог для получения нужного количества углов.

                Args:
                    r_norm: Нормализованный отклик Харриса
                    target_corners: Целевое количество углов

                Returns:
                    Пороговое значение
                """
                sorted_response = np.sort(r_norm.flatten())[::-1]
                if len(sorted_response) > target_corners:
                    threshold = sorted_response[target_corners]

                else:
                    threshold = sorted_response[-1] if len(sorted_response) > 0 else 0.5

                return max(0.1, min(0.9, threshold))

            threshold = find_adaptive_threshold(response_norm, target_corners=1000)
            corner_mask = response_norm > threshold

            # 8. Подавление немаксимумов
            height, width = response.shape
            local_maxima = np.zeros_like(corner_mask, dtype=bool)

            for i in range(1, height - 1):
                for col in range(1, width - 1):
                    if corner_mask[i, col]:
                        neighborhood = response_norm[i - 1:i + 2, col - 1:col + 2]
                        if response_norm[i, col] == np.max(neighborhood):
                            local_maxima[i, col] = True

            # 9. Создаем результат
            result = self.image.copy().astype(np.uint8)

            # 10. Рисуем углы
            corner_coords = np.where(local_maxima)
            if len(corner_coords[0]) > 0:
                for y_coord, x_coord in zip(corner_coords[0], corner_coords[1]):
                    y_start = max(0, y_coord - 1)
                    y_end = min(height, y_coord + 2)
                    x_start = max(0, x_coord - 1)
                    x_end = min(width, x_coord + 2)

                    result[y_start:y_end, x_start:x_end, 0] = 255  # Красный
                    result[y_start:y_end, x_start:x_end, 1] = 0    # Зеленый
                    result[y_start:y_end, x_start:x_end, 2] = 0    # Синий

            self.result_image = result

        self.measure_time(_detect_corners)
        print(f"Harris corner detection took {self.execution_time:.4f} seconds")

    def save_result(self: 'ImageProcessor', output_path: str) -> None:
        """
        Сохраняет обработанное изображение.

        Args:
            output_path: Путь для сохранения

        Raises:
            ValueError: Если нет обработанного изображения для сохранения
        """
        if self.result_image is not None:
            # Конвертируем обратно в BGR для сохранения с помощью OpenCV
            if len(self.result_image.shape) == 3:
                bgr_image = cv2.cvtColor(self.result_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, bgr_image)
            else:
                cv2.imwrite(output_path, self.result_image)
        else:
            raise ValueError("No result image to save")
