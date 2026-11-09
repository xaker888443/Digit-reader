# Digit recognizer (CNN, PyTorch)

CNN-классификатор рукописных цифр 0–9 (28×28, оттенки серого).

## Файлы
- `data.py`     — синтетический генератор датасета (сдвиг/поворот/шум)
- `model.py`    — сеть `DigitCNN`
- `train.py`    — обучение + early stopping, чекпойнт `digit_net.pt`
- `recognize.py`— инференс: `--demo` или список путей к изображениям

## Запуск
    pip install -r requirements.txt
    python train.py
    python recognize.py --demo
    python recognize.py path/to/img.png