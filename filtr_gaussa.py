import cv2
import numpy as np

from filtr_wspolny import konwolucja_valid, pokaz_i_zapisz, wczytaj_obraz


input_path = r"C:\Users\anton\Downloads\wiewior.jpg"
output_path = r"C:\Users\anton\Downloads\wiewior_gauss.jpg"

image = wczytaj_obraz(input_path)

# Filtr Gaussa wygładza obraz i usuwa drobny szum.
gaussian_1d = cv2.getGaussianKernel(5, sigma=0)
kernel = gaussian_1d @ gaussian_1d.T
result = konwolucja_valid(image, kernel, stride=1)
result = np.clip(result, 0, 255).astype("uint8")

pokaz_i_zapisz(image, result, "Filtr Gaussa", output_path)
