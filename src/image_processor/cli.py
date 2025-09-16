import argparse
import sys
from .image_processor import ImageProcessor
import numpy as np


def main():
    """Основная функция для обработки аргументов командной строки."""
    parser = argparse.ArgumentParser(description='Image Processing Tool')

    # Добавляем аргументы
    parser.add_argument('operation',
                        choices=['grayscale', 'gamma', 'convolution', 'sobel', 'harris', 'hough'],
                        help='Operation to perform')
    parser.add_argument('input_path', help='Path to input image')
    parser.add_argument('output_path', nargs='?', default='output.png',
                        help='Path to save output image (default: output.png)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Gamma value for correction (default: 1.0)')
    parser.add_argument('--kernel', type=str,
                        help='Convolution kernel as comma-separated values')

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
            kernel_values = [float(x) for x in args.kernel.split(',')]
            size = int(len(kernel_values) ** 0.5)
            if size * size != len(kernel_values):
                print("Error: Kernel must be square")
                sys.exit(1)
            kernel = np.array(kernel_values).reshape((size, size))
            processor.apply_convolution(kernel)
        elif args.operation == 'sobel':
            processor.sobel_edge_detection()
        elif args.operation == 'harris':
            print("Harris corner detection not implemented yet")
            sys.exit(1)
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