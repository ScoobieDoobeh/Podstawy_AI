import numpy as np

from filtr_wspolny import konwolucja_valid, pokaz_i_zapisz, wczytaj_obraz


input_path = r"C:\Users\anton\Downloads\wiewior.jpg"
output_path = r"C:\Users\anton\Downloads\wiewior_laplace.jpg"

image = wczytaj_obraz(input_path)
gray = np.dot(image[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)

# Laplace wykrywa krawędzie we wszystkich kierunkach.
kernel = np.array(
    [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ],
    dtype=np.float32,
)

result = konwolucja_valid(gray, kernel, stride=1)
result = np.clip(np.abs(result.astype(np.float32)), 0, 255).astype(np.uint8)

pokaz_i_zapisz(gray, result, "Filtr Laplace'a", output_path)
