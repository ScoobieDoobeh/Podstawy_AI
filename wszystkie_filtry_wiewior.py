"""Uruchamia wszystkie dzisiejsze filtry i detektory na zdjeciu wiewiorki.

Kazdy rezultat jest zapisywany jako osobny plik PNG. Skrypt nie otwiera okien,
dzieki czemu wszystkie wyniki powstaja po jednym uruchomieniu.
"""

from pathlib import Path

import cv2
import numpy as np


INPUT_PATH = Path(r"C:\Users\anton\Downloads\wiewior.jpg")
OUTPUT_DIR = Path(r"C:\Users\anton\Downloads\wiewior_wyniki")


def konwolucja_valid(image: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """Wykonuje konwolucje bez paddingu i z podanym krokiem."""
    filtered = cv2.filter2D(
        image,
        ddepth=cv2.CV_32F,
        kernel=kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    border = kernel.shape[0] // 2
    valid = filtered[border:-border, border:-border]
    return valid[::stride, ::stride]


def zapisz(output_path: Path, image: np.ndarray) -> None:
    """Zapisuje obraz i jasno sygnalizuje blad zapisu."""
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Nie udalo sie zapisac wyniku: {output_path}")
    print(f"Zapisano: {output_path}")


def main() -> None:
    image = cv2.imread(str(INPUT_PATH))
    if image is None:
        raise FileNotFoundError(f"Nie udalo sie wczytac obrazu: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Filtr Gaussa 5x5.
    gaussian_1d = cv2.getGaussianKernel(5, sigma=0)
    gaussian_kernel = gaussian_1d @ gaussian_1d.T
    gaussian = konwolucja_valid(image, gaussian_kernel)
    zapisz(OUTPUT_DIR / "01_gauss.png", np.clip(gaussian, 0, 255).astype(np.uint8))

    # 2. Filtr konwolucyjny z padding=0 i stride=2.
    convolution_kernel = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        dtype=np.float32,
    )
    convolution = konwolucja_valid(image, convolution_kernel, stride=2)
    zapisz(OUTPUT_DIR / "02_konwolucja_stride_2.png", np.clip(convolution, 0, 255).astype(np.uint8))

    # Obrazy ponizej sa monochromatyczne, tak jak w osobnych skryptach.
    laplace_kernel = np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=np.float32,
    )
    laplace = konwolucja_valid(gray, laplace_kernel)
    zapisz(OUTPUT_DIR / "03_laplace.png", np.clip(np.abs(laplace), 0, 255).astype(np.uint8))

    sobel_x_kernel = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=np.float32,
    )
    sobel_x = konwolucja_valid(gray, sobel_x_kernel)
    zapisz(OUTPUT_DIR / "04_sobel_x.png", np.clip(np.abs(sobel_x), 0, 255).astype(np.uint8))

    sobel_y_kernel = np.array(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=np.float32,
    )
    sobel_y = konwolucja_valid(gray, sobel_y_kernel)
    zapisz(OUTPUT_DIR / "05_sobel_y.png", np.clip(np.abs(sobel_y), 0, 255).astype(np.uint8))

    sharpen_kernel = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        dtype=np.float32,
    )
    sharpened = konwolucja_valid(image, sharpen_kernel)
    zapisz(OUTPUT_DIR / "06_wyostrzanie.png", np.clip(sharpened, 0, 255).astype(np.uint8))

    # 7. Detekcja krawedzi Canny'ego.
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)
    canny = cv2.Canny(blurred_gray, threshold1=50, threshold2=150)
    zapisz(OUTPUT_DIR / "07_canny.png", canny)

    # 8. Probabilistyczna transformacja Hougha na krawedziach Canny'ego.
    hough = image.copy()
    lines = cv2.HoughLinesP(
        canny,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=60,
        maxLineGap=15,
    )
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(hough, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
    zapisz(OUTPUT_DIR / "08_hough_lines.png", hough)

    print(f"Gotowe. Zapisano wyniki w: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
