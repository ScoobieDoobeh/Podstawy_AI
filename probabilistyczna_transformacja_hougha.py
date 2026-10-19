"""Przyklad probabilistycznej transformacji Hougha do wykrywania linii."""

from pathlib import Path

import cv2
import numpy as np


def utworz_obraz_testowy() -> np.ndarray:
    """Tworzy prosty obraz z liniami, aby skrypt nie wymagal pliku wejsciowego."""
    image = np.zeros((500, 800, 3), dtype=np.uint8)
    image[:] = (35, 35, 35)

    cv2.line(image, (80, 430), (280, 90), (255, 255, 255), 5)
    cv2.line(image, (280, 90), (500, 430), (255, 255, 255), 5)
    cv2.line(image, (70, 350), (720, 350), (180, 180, 180), 4)
    cv2.line(image, (540, 100), (730, 430), (255, 255, 255), 5)
    cv2.rectangle(image, (100, 120), (230, 250), (120, 120, 120), 3)
    return image


def main() -> None:
    image = utworz_obraz_testowy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

    # HoughLinesP zwraca odcinki, dzieki czemu mozna narysowac je na obrazie.
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=60,
        maxLineGap=15,
    )

    detected = image.copy()
    if lines is not None:
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = line
            cv2.line(detected, (x1, y1), (x2, y2), (0, 0, 255), 3)

    result = np.hstack((image, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), detected))
    cv2.putText(result, "oryginal", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(result, "krawedzie Canny", (815, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(result, "linie Hougha", (1615, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    output_path = Path(__file__).with_name("hough_lines_result.png")
    if not cv2.imwrite(str(output_path), result):
        raise IOError(f"Nie udalo sie zapisac wyniku: {output_path}")

    line_count = 0 if lines is None else int(lines.shape[0])
    print(f"Wykryto odcinkow: {line_count}")
    print(f"Wynik zapisano w: {output_path}")
    cv2.imshow("Transformacja Hougha", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
