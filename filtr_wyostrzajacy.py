import numpy as np

from filtr_wspolny import konwolucja_valid, pokaz_i_zapisz, wczytaj_obraz


input_path = r"C:\Users\anton\Downloads\wiewior.jpg"
output_path = r"C:\Users\anton\Downloads\wiewior_wyostrzony.jpg"

image = wczytaj_obraz(input_path)

# Filtr wzmacnia szczegóły i krawędzie obrazu.
kernel = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ],
    dtype=np.float32,
)

result = konwolucja_valid(image, kernel, stride=1)
result = np.clip(result, 0, 255).astype(np.uint8)

pokaz_i_zapisz(image, result, "Filtr wyostrzający", output_path)
