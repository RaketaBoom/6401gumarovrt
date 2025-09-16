import time
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image, ImageFilter


class ImageProcessor:
    """Основной класс для обработки изображений."""

    def __init__(self, image_path: str):
        """
        Инициализация процессора изображений.

        Args:
            image_path: Путь к исходному изображению
        """
        self.image = Image.open(image_path)
        self.result_image: Optional[Image.Image] = None
        self.execution_time: float = 0.0

    def measure_time(self, func, *args, **kwargs) -> None:
        """Измеряет время выполнения функции."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        self.execution_time = end_time - start_time
        return result

    def to_grayscale(self) -> None:
        """Преобразует цветное изображение в полутоновое."""
        def _convert():
            if self.image.mode != 'L':
                self.result_image = self.image.convert('L')
            else:
                self.result_image = self.image.copy()

        self.measure_time(_convert)
        print(f"Grayscale conversion took {self.execution_time:.4f} seconds")

    def gamma_correction(self, gamma: float) -> None:
        """
        Применяет гамма-коррекцию к изображению.

        Args:
            gamma: Значение гаммы (>1 - осветление, <1 - затемнение)
        """
        def _correct():
            # Преобразуем изображение в numpy array
            img_array = np.array(self.image, dtype=np.float32) / 255.0

            # Применяем гамма-коррекцию
            corrected = np.power(img_array, gamma)

            # Масштабируем обратно к диапазону 0-255
            corrected = np.uint8(corrected * 255)

            self.result_image = Image.fromarray(corrected)

        self.measure_time(_correct)
        print(f"Gamma correction took {self.execution_time:.4f} seconds")

    def apply_convolution(self, kernel: np.ndarray) -> None:
        """
        Применяет свертку с заданным ядром.

        Args:
            kernel: Ядро свертки (2D массив NumPy)
        """
        def _convolve():
            # Конвертируем изображение в режим RGB
            img = self.image.convert('RGB')
            img_array = np.array(img)

            # Получаем размеры изображения и ядра
            img_h, img_w = img_array.shape[:2]
            k_h, k_w = kernel.shape

            # Вычисляем padding
            pad_h, pad_w = k_h // 2, k_w // 2

            # Создаем выходной массив
            output = np.zeros_like(img_array)

            # Добавляем padding к изображению
            padded = np.pad(img_array, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                            mode='reflect')

            # Применяем свертку
            for i in range(img_h):
                for j in range(img_w):
                    for c in range(3):  # Для каждого канала (R, G, B)
                        region = padded[i:i+k_h, j:j+k_w, c]
                        output[i, j, c] = np.sum(region * kernel)

            # Обрезаем значения до диапазона 0-255
            output = np.clip(output, 0, 255).astype(np.uint8)
            self.result_image = Image.fromarray(output)

        self.measure_time(_convolve)
        print(f"Convolution took {self.execution_time:.4f} seconds")

    def sobel_edge_detection(self) -> None:
        """Выделяет границы с помощью оператора Собеля."""
        def _detect_edges():
            # Преобразуем в grayscale, если нужно
            if self.image.mode != 'L':
                gray_img = self.image.convert('L')
            else:
                gray_img = self.image.copy()

            gray_array = np.array(gray_img, dtype=np.float32)

            # Операторы Собеля
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

            # Применяем свертки
            grad_x = self._convolve2d(gray_array, sobel_x)
            grad_y = self._convolve2d(gray_array, sobel_y)

            # Вычисляем магнитуду градиента
            magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Нормализуем к диапазону 0-255
            magnitude = (magnitude / magnitude.max()) * 255
            magnitude = magnitude.astype(np.uint8)

            self.result_image = Image.fromarray(magnitude)

        self.measure_time(_detect_edges)
        print(f"Sobel edge detection took {self.execution_time:.4f} seconds")

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Вспомогательная функция для 2D свертки.

        Args:
            image: Входное изображение (2D массив)
            kernel: Ядро свертки (2D массив)

        Returns:
            Результат свертки
        """
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2

        # Добавляем padding
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

        # Создаем выходной массив
        output = np.zeros_like(image)

        # Применяем свертку
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+k_h, j:j+k_w]
                output[i, j] = np.sum(region * kernel)

        return output

    def save_result(self, output_path: str) -> None:
        """
        Сохраняет обработанное изображение.

        Args:
            output_path: Путь для сохранения
        """
        if self.result_image is not None:
            self.result_image.save(output_path)
        else:
            raise ValueError("No result image to save")