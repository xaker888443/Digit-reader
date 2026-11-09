"""Распознавание цифр: --demo либо список путей к изображениям."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import generate_dataset
from model import DigitCNN

SIDE = 28
WEIGHTS = Path("digit_net.pt")


def _preprocess(image: Image.Image) -> torch.Tensor:
    gray = image.convert("L").resize((SIDE, SIDE), Image.BILINEAR)
        arr = np.asarray(gray, dtype=np.float32) / 255.0
            if arr.mean() > 0.5:          # тёмная цифра на светлом фоне — инвертируем
                    arr = 1.0 - arr
                        tensor = torch.from_numpy(arr)[None, None, :, :]
                            return tensor


                            def _load_model(device: torch.device) -> DigitCNN:
                                if not WEIGHTS.exists():
                                        raise FileNotFoundError(f"нет весов: {WEIGHTS}. Сначала запустите train.py")
                                            net = DigitCNN().to(device)
                                                net.load_state_dict(torch.load(WEIGHTS, map_location=device))
                                                    net.eval()
                                                        return net


                                                        def _predict(net: DigitCNN, batch: torch.Tensor, device: torch.device) -> np.ndarray:
                                                            with torch.no_grad():
                                                                    logits = net(batch.to(device))
                                                                        return logits.argmax(1).cpu().numpy()


                                                                        def demo(net: DigitCNN, device: torch.device) -> None:
                                                                            X, y = generate_dataset(num_per_digit=5, seed=7)
                                                                                batch = torch.from_numpy(X)
                                                                                    preds = _predict(net, batch, device)
                                                                                        ok = (preds == y).sum()
                                                                                            print(f"demo: {ok}/{len(y)} верно")
                                                                                                for truth, pred in zip(y.tolist(), preds.tolist()):
                                                                                                        mark = "ok" if truth == pred else "!!"
                                                                                                                print(f"  {mark} true={truth} pred={pred}")


                                                                                                                def recognize_files(net: DigitCNN, device: torch.device, paths: list[str]) -> None:
                                                                                                                    for p in paths:
                                                                                                                            img = Image.open(p)
                                                                                                                                    tensor = _preprocess(img)
                                                                                                                                            pred = int(_predict(net, tensor, device)[0])
                                                                                                                                                    print(f"{p}: {pred}")


                                                                                                                                                    def main() -> None:
                                                                                                                                                        parser = argparse.ArgumentParser(description="CNN digit recognizer")
                                                                                                                                                            parser.add_argument("images", nargs="*", help="пути к изображениям")
                                                                                                                                                                parser.add_argument("--demo", action="store_true", help="проверка на синтетике")
                                                                                                                                                                    args = parser.parse_args()

                                                                                                                                                                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                                                                                                                                                            net = _load_model(device)

                                                                                                                                                                                if args.demo:
                                                                                                                                                                                        demo(net, device)
                                                                                                                                                                                            elif args.images:
                                                                                                                                                                                                    recognize_files(net, device, args.images)
                                                                                                                                                                                                        else:
                                                                                                                                                                                                                parser.print_help()


                                                                                                                                                                                                                if __name__ == "__main__":
                                                                                                                                                                                                    