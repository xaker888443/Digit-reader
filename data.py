"""Синтетический генератор изображений цифр (MNIST-подобный)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

IMG_SIDE = 28


def _render_digit(digit: int, rng: np.random.Generator) -> np.ndarray:
    """Рисует одну цифру на холсте побольше, затем возвращает 28x28 float32."""
    canvas = Image.new("L", (56, 56), color=0)
    draw = ImageDraw.Draw(canvas)

    # грубая сегментная отрисовка цифр 0-9 толстыми линиями
    strokes = {
        0: [(14, 8, 40, 8), (40, 8, 40, 46), (40, 46, 14, 46), (14, 46, 14, 8)],
        1: [(28, 8, 28, 46)],
        2: [(14, 10, 40, 10), (40, 10, 40, 26), (40, 26, 14, 26),
            (14, 26, 14, 44), (14, 44, 40, 44)],
        3: [(14, 10, 40, 10), (40, 10, 40, 44), (40, 44, 14, 44), (20, 27, 40, 27)],
        4: [(14, 10, 14, 30), (14, 30, 40, 30), (40, 10, 40, 46)],
        5: [(40, 10, 14, 10), (14, 10, 14, 27), (14, 27, 40, 27),
            (40, 27, 40, 44), (40, 44, 14, 44)],
        6: [(40, 10, 16, 10), (16, 10, 16, 44), (16, 44, 40, 44), (40, 44, 40, 27), (40, 27, 16, 27)],
        7: [(14, 10, 40, 10), (40, 10, 20, 46)],
        8: [(14, 10, 40, 10), (40, 10, 40, 44), (40, 44, 14, 44), (14, 44, 14, 10),
            (14, 27, 40, 27)],
        9: [(40, 44, 40, 12), (40, 12, 16, 12), (16, 12, 16, 27), (16, 27, 40, 27),
            (16, 12, 16, 10)],
    }
    width = int(rng.integers(2, 5))
    for x1, y1, x2, y2 in strokes[digit]:
        draw.line((x1, y1, x2, y2), fill=255, width=width)

    # лёгкое размытие — сглаживание
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 1.0))))

    # случайный аффинный сдвиг/поворот
    angle = float(rng.uniform(-12, 12))
    canvas = canvas.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    shift = tuple(int(v) for v in rng.integers(-3, 4, size=2))
    canvas = Image.fromarray(np.roll(np.array(canvas), shift, axis=(0, 1)))

    # обрезка/центрирование до 28x28
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    ys, xs = np.nonzero(arr > 0.1)
    if len(xs) == 0:
        return np.zeros((IMG_SIDE, IMG_SIDE), dtype=np.float32)
    cropped = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    pil = Image.fromarray((cropped * 255).astype(np.uint8))
    pil.thumbnail((20, 20), Image.BILINEAR)
    out = Image.new("L", (IMG_SIDE, IMG_SIDE), 0)
    out.paste(pil, ((IMG_SIDE - pil.width) // 2, (IMG_SIDE - pil.height) // 2))
    img = np.asarray(out, dtype=np.float32) / 255.0

    # шум
    img = np.clip(img + rng.normal(0, 0.03, img.shape).astype(np.float32), 0.0, 1.0)
    return img


def generate_dataset(num_per_digit: int = 300, seed: int = 42):
    """Возвращает (X, y): X float32 (N,1,28,28), y int64 (N,)."""
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for digit in range(10):
        for _ in range(num_per_digit):
            images.append(_render_digit(digit, rng))
            labels.append(digit)
    X = np.stack(images).astype(np.float32)[:, None, :, :]
    y = np.asarray(labels, dtype=np.int64)
    return X, y