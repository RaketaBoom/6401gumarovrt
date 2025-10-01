### Преобразование в grayscale
python imageprocessing.py grayscale images/input/photo.jpg images/output/grayscale.jpg

### Гамма-коррекция
python imageprocessing.py gamma images/input/photo.jpg images/output/gamma.jpg --gamma 1.5

### Свертка с ядром размытия
python imageprocessing.py convolution images/input/photo.jpg images/output/blur.jpg --kernel "0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625"

### Обнаружение границ Собелем
python imageprocessing.py sobel images/input/photo.jpg images/output/edges.jpg

# Обнаружение углов Харрисом
python imageprocessing.py harris images/input/photo.jpg images/output/edges.jpg