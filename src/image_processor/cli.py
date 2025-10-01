"""
Модуль командной строки для обработки изображений.

Предоставляет интерфейс командной строки для выполнения различных операций
над изображениями, включая преобразование в градации серого, гамма-коррекцию,
свертку, обнаружение границ и углов.
"""

import argparse
import sys
from typing import NoReturn

from src.image_processor.utils import parse_kernel
from .image_processor import ImageProcessor


def main() -> NoReturn:
    """Основная функция для обработки аргументов командной строки."""
    parser = argparse.ArgumentParser(description='Image Processing Tool')

    # Добавляем аргументы
    parser.add_argument(
        'operation',
        choices=['grayscale', 'gamma', 'convolution', 'sobel', 'harris', 'hough'],
        help='Operation to perform',
    )
    parser.add_argument('input_path', help='Path to input image')
    parser.add_argument(
        'output_path',
        nargs='?',
        default='output.png',
        help='Path to save output image (default: output.png)',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=1.0,
        help='Gamma value for correction (default: 1.0)',
    )
    parser.add_argument(
        '--kernel',
        type=str,
        help='Convolution kernel as comma-separated values',
    )

    # Парсим аргументы
    args = parser.parse_args()

    try:
        # Создаем процессор
        processor = ImageProcessor(args.input_path)

        # Выполняем операцию
        if args.operation == 'grayscale':
            processor.to_grayscale()
        elif args.operation == 'gamma':
            processor.gamma_correction(args.gamma)
        elif args.operation == 'convolution':
            if not args.kernel:
                print("Error: Kernel required for convolution")
                sys.exit(1)
            # Парсим ядро
            kernel = parse_kernel(args.kernel)
            processor.apply_convolution(kernel)
        elif args.operation == 'sobel':
            processor.sobel_edge_detection()
        elif args.operation == 'harris':
            processor.harris_corner_detection()
        elif args.operation == 'hough':
            print("Hough circle detection not implemented yet")
            sys.exit(1)

        # Сохраняем результат
        processor.save_result(args.output_path)
        print(f"Result saved to {args.output_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
