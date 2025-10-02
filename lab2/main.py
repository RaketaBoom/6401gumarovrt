from lab2.CatImageProcessor import CatImageProcessor


def main():
    try:
        limit = int(input("Введите количество изображений: "))
        if limit > 100:
            print("Максимальное количество изображений за один запрос - 100. Установлено 100.")
            limit = 100

        processor = CatImageProcessor()

        cat_images = processor.get_cat_images(limit)

        if cat_images:
            processor.process_images(cat_images)

            processor.save_images(cat_images)

            print(f"Успешно обработано и сохранено {len(cat_images)} изображений кошек!")
        else:
            print("Не удалось получить изображения кошек.")

    except ValueError as e:
        print(f"Ошибка ввода: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")


if __name__ == "__main__":
    main()