import cv2
import numpy as np


def wczytaj_obraz(input_path):
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Nie udało się wczytać obrazu: {input_path}")
    return image


def konwolucja_valid(image, kernel, stride=1):
    """Wykonuje konwolucję bez paddingu i z podanym krokiem."""
    kernel_size = kernel.shape[0]
    border = kernel_size // 2
    filtered = cv2.filter2D(
        image,
        ddepth=cv2.CV_32F,
        kernel=kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    valid = filtered[border:-border, border:-border]
    return valid[::stride, ::stride]


def pokaz_i_zapisz(image, result, window_name, output_path):
    if not cv2.imwrite(output_path, result):
        raise IOError(f"Nie udało się zapisać wyniku: {output_path}")

    cv2.imshow("Oryginalny obraz", image)
    cv2.imshow(window_name, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
