import os
from typing import List, Dict, Optional

import cv2
import numpy as np

from lab1.utils.time_measure import measure_time
from lab2.CatImage import CatImage
from lab2.CatClient import CatClient
from lab2.CatsResponse import CatsResponse


class CatImageProcessor:
    """
    Класс для обработки изображений кошек.
    Использует CatClient для получения данных и выполняет обработку изображений.
    """

    def __init__(self) -> None:
        """
        Инициализация процессора.
        """
        self._cat_client: CatClient = CatClient()

    @measure_time
    def _build_cat_image(self, image_url: str, breed: str) -> Optional[CatImage]:
        """
        Создает объект CatImage из URL и породы.

        Args:
            image_url: URL изображения кошки
            breed: порода кошки

        Returns:
            Объект CatImage или None при ошибке
        """
        try:
            # Используем CatClient для загрузки изображения
            image: Optional[np.ndarray] = self._cat_client.download_image(image_url)
            if image is None:
                print(f"Не удалось загрузить изображение с URL: {image_url}")
                return None

            cat_image: CatImage = CatImage(image, image_url, breed)
            print(f"Изображение кота создано успешно: {cat_image}")

            return cat_image

        except (KeyError, IndexError) as exception:
            print(f"Ошибка при обработке данных изображения: {exception}")
            return None

    @measure_time
    def get_cat_images(self, limit: int = 1) -> List[CatImage]:
        """
        Получает и преобразует данные API в объекты CatImage.

        Args:
            limit: количество изображений для получения

        Returns:
            Список объектов CatImage
        """
        print("Получение данных о котах через CatClient...")

        # Используем CatClient для получения структурированных данных
        cats_response: CatsResponse = self._cat_client.get_cats(limit)

        print(f"Старт маппинга {cats_response.count} DTO в CatImage...")
        cat_images: List[CatImage] = []

        for index, cat_dto in enumerate(cats_response.images):
            # Определяем породу (берем первую из списка или 'Unknown')
            breed_name: str = "Unknown"
            if cat_dto.breeds:
                breed_name = cat_dto.breeds[0].name

            # Создаем CatImage
            cat_image: Optional[CatImage] = self._build_cat_image(cat_dto.url, breed_name)
            if cat_image is not None:
                cat_images.append(cat_image)

            print(f"Смаплено {index + 1}/{cats_response.count} изображений")

        print(f"Успешно создано {len(cat_images)} объектов CatImage")
        return cat_images

    @measure_time
    def process_images(self, cat_images: List[CatImage]) -> Dict[str, List[np.ndarray]]:
        """
        Обрабатывает изображения (выделение контуров) используя методы CatImage.

        Args:
            cat_images: список объектов CatImage для обработки

        Returns:
            Словарь с тремя списками изображений:
            - 'original': исходные изображения
            - 'lib_edges': библиотечная обработка
            - 'custom_edges': пользовательская обработка
        """
        cat_images_number: int = len(cat_images)
        print(f"Обработка {cat_images_number} изображений...")

        original_images: List[np.ndarray] = []
        lib_edges_images: List[np.ndarray] = []
        custom_edges_images: List[np.ndarray] = []

        for index, cat_image in enumerate(cat_images):
            original_images.append(cat_image.image.copy())
            lib_edges_images.append(cat_image.detect_edges_using_library())
            custom_edges_images.append(cat_image.detect_edges_using_custom_method())

            print(f"Обработано {index + 1}/{cat_images_number} изображений")

        print(f"Обработка {cat_images_number} изображений завершена успешно")

        return {
            'original': original_images,
            'lib_edges': lib_edges_images,
            'custom_edges': custom_edges_images
        }

    @measure_time
    def save_images(self,
                    cat_images: List[CatImage],
                    output_dir: str = "cat_images") -> None:
        """
        Сохраняет изображения в файлы.

        Args:
            cat_images: список объектов CatImage (для получения метаданных)
            processed_data: словарь с обработанными изображениями
            output_dir: директория для сохранения результатов
        """
        if not cat_images:
            print("Нет изображений для сохранения")
            return

        print(f"Сохранение {len(cat_images)} изображений...")
        os.makedirs(output_dir, exist_ok=True)

        for index, cat_image in enumerate(cat_images):
            safe_breed: str = "".join(c if c.isalnum() else "_" for c in cat_image.breed)
            breed_dir: str = self._create_breed_directory(safe_breed, output_dir)

            original_path, lib_edges_path, custom_edges_path, sum_edges_path = self._generate_file_paths(
                breed_dir, safe_breed, index
            )

            edgedCat = cat_image + cat_image.lib_image

            try:
                cv2.imwrite(original_path, cat_image.image)
                cv2.imwrite(lib_edges_path, cat_image.lib_image)
                cv2.imwrite(custom_edges_path, cat_image.custom_image)
                cv2.imwrite(sum_edges_path, edgedCat.image)
                print(f"Сохранено изображение {index + 1}: {cat_image.breed}")

            except Exception as e:
                print(f"Ошибка при сохранении изображения {index + 1}: {e}")

        print(f"Сохранение завершено. Результаты в директории: {output_dir}")

    @measure_time
    def _create_breed_directory(self, safe_breed: str, output_dir: str) -> str:
        """
        Создает безопасную директорию для породы.

        Args:
            safe_breed: название породы
            output_dir: основная директория для сохранения

        Returns:
            Путь к созданной директории
        """
        breed_dir: str = os.path.join(output_dir, safe_breed)
        os.makedirs(breed_dir, exist_ok=True)
        return breed_dir

    @measure_time
    def _generate_file_paths(self, breed_dir: str, safe_breed: str, index: int) -> tuple[str, str, str, str]:
        """
        Генерирует пути для файлов изображений.

        Args:
            breed_dir: директория породы
            safe_breed: безопасное название породы
            index: индекс изображения

        Returns:
            Кортеж путей (original, lib_edges, custom_edges)
        """
        original_path: str = os.path.join(breed_dir, f"{index + 1}_{safe_breed}_original.jpg")
        lib_edges_path: str = os.path.join(breed_dir, f"{index + 1}_{safe_breed}_lib_edges.jpg")
        custom_edges_path: str = os.path.join(breed_dir, f"{index + 1}_{safe_breed}_custom_edges.jpg")
        sum_edges_path: str = os.path.join(breed_dir, f"{index + 1}_{safe_breed}_sum_edges.jpg")

        return original_path, lib_edges_path, custom_edges_path, sum_edges_path