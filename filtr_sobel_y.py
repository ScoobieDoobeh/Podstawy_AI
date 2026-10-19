import numpy as np

from filtr_wspolny import konwolucja_valid, pokaz_i_zapisz, wczytaj_obraz


input_path = r"C:\Users\anton\Downloads\wiewior.jpg"
output_path = r"C:\Users\anton\Downloads\wiewior_sobel_y.jpg"

image = wczytaj_obraz(input_path)
gray = np.dot(image[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)

# Sobel Y wykrywa głównie krawędzie poziome.
kernel = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ],
    dtype=np.float32,
)

result = konwolucja_valid(gray, kernel, stride=1)
result = np.clip(np.abs(result.astype(np.float32)), 0, 255).astype(np.uint8)

pokaz_i_zapisz(gray, result, "Sobel Y", output_path)
