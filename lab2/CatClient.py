import os
from typing import List, Dict, Any, Optional

import requests
import numpy as np
import cv2
from dotenv import load_dotenv

from lab1.utils.time_measure import measure_time
from lab2.CatsResponse import CatsResponse, CatImageDTO, Breed


class CatClient:
    """
    Клиент для работы с Cat API.
    Инкапсулирует всю логику взаимодействия с внешними ресурсами.
    """

    def __init__(self) -> None:
        """
        Инициализация клиента.
        """
        self._base_url: str = "https://api.thecatapi.com/v1/images/search"
        self._api_key: str = self._get_api_key()

    @measure_time
    @staticmethod
    def _get_api_key() -> str:
        """
        Получает API ключ из переменных окружения.

        Returns:
            API ключ

        Raises:
            ValueError: если ключ не найден
        """
        load_dotenv(".env")
        api_key: Optional[str] = os.getenv('API_KEY')
        if not api_key:
            raise ValueError(
                "API_KEY не найден. Добавьте API_KEY в файл .env"
            )
        return api_key

    @measure_time
    def get_cats(self, limit: int = 1) -> CatsResponse:
        """
        Получает данные о котах из API.

        Args:
            limit: количество изображений для получения

        Returns:
            Объект CatsResponse с данными о котах
        """
        print(f"Получение {limit} изображений из API...")

        params: Dict[str, Any] = {
            'limit': limit,
            'has_breeds': 1,
            'api_key': self._api_key
        }

        try:
            response: requests.Response = requests.get(self._base_url, params=params)
            response.raise_for_status()
            json_data: List[Dict[str, Any]] = response.json()

            return self._parse_api_response(json_data)

        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе к API: {e}")
            return CatsResponse(images=[], count=0)

    @measure_time
    def download_image(self, image_url: str) -> Optional[np.ndarray]:
        """
        Загружает изображение по URL.

        Args:
            image_url: URL изображения для загрузки

        Returns:
            numpy-массив с изображением или None при ошибке
        """
        try:
            img_response: requests.Response = requests.get(image_url)
            img_response.raise_for_status()

            img_array: np.ndarray = np.frombuffer(img_response.content, np.uint8)
            image: np.ndarray = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if image is None:
                print(f"Не удалось декодировать изображение с URL: {image_url}")
                return None

            return image

        except requests.exceptions.RequestException as e:
            print(f"Ошибка при загрузке изображения {image_url}: {e}")
            return None
        except Exception as e:
            print(f"Неожиданная ошибка при загрузке изображения {image_url}: {e}")
            return None

    def _parse_api_response(self, api_data: List[Dict[str, Any]]) -> CatsResponse:
        """
        Парсит сырой JSON ответ API в DTO объекты.

        Args:
            api_data: сырые данные из API

        Returns:
            Структурированный ответ с DTO
        """
        cat_images: List[CatImageDTO] = []

        for item in api_data:
            try:
                # Парсим породы
                breeds: List[Breed] = []
                for breed_data in item.get('breeds', []):
                    breed: Breed = Breed(
                        id=breed_data.get('id', ''),
                        name=breed_data.get('name', 'Unknown'),
                        temperament=breed_data.get('temperament'),
                        origin=breed_data.get('origin')
                    )
                    breeds.append(breed)

                # Создаем DTO изображения
                cat_image_dto: CatImageDTO = CatImageDTO(
                    id=item['id'],
                    url=item['url'],
                    width=item['width'],
                    height=item['height'],
                    breeds=breeds
                )
                cat_images.append(cat_image_dto)

            except KeyError as e:
                print(f"Ошибка парсинга элемента API: отсутствует ключ {e}")
                continue

        return CatsResponse(images=cat_images, count=len(cat_images))