"""Przyklad wykrywania krawedzi algorytmem Canny'ego w OpenCV."""

from pathlib import Path

import cv2
import numpy as np


def utworz_obraz_testowy() -> np.ndarray:
    """Tworzy obraz demonstracyjny z figurami i delikatnym szumem."""
    image = np.full((500, 800, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (80, 80), (300, 300), (40, 100, 220), -1)
    cv2.circle(image, (560, 220), 120, (50, 180, 80), -1)
    cv2.line(image, (80, 410), (720, 410), (30, 30, 30), 8)

    noise = np.random.default_rng(7).normal(0, 8, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def main() -> None:
    image = utworz_obraz_testowy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Wygladzenie ogranicza wplyw szumu na wykrywanie krawedzi.
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    result = np.hstack(
        (image, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    )
    cv2.putText(result, "oryginal", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(result, "skala szarosci", (815, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(result, "krawedzie Canny", (1615, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    output_path = Path(__file__).with_name("canny_edges_result.png")
    if not cv2.imwrite(str(output_path), result):
        raise IOError(f"Nie udalo sie zapisac wyniku: {output_path}")

    print(f"Wynik zapisano w: {output_path}")
    cv2.imshow("Canny edge detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
