import cv2
import numpy as np


# KROK 1: Ustalamy ścieżkę obrazu wejściowego oraz nazwę pliku wynikowego.
input_path = r"C:\Users\anton\Downloads\wiewior.jpg"
output_path = r"C:\Users\anton\Downloads\wiewior_filtr_konwolucji.jpg"

# KROK 2: Wczytujemy obraz.
# OpenCV zwraca obraz jako tablicę NumPy w kolejności kanałów BGR.
image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError(f"Nie udało się wczytać obrazu: {input_path}")

# KROK 3: Tworzymy jądro (maskę) filtra konwolucyjnego 3x3.
# Podczas konwolucji ta macierz przesuwa się po obrazie.
# W każdym położeniu wartości z maski są mnożone przez wartości pikseli,
# a następnie sumowane. Ten konkretny filtr wyostrza obraz.
# Suma współczynników wynosi 1, więc jasność obrazu pozostaje podobna.
kernel = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ],
    dtype=np.float32,
)

# KROK 4: Ustawiamy parametry konwolucji.
# Rozmiar jądra wynosi 3, więc od każdego brzegu trzeba odrzucić 1 piksel.
# Padding=0 oznacza, że nie dodajemy sztucznych pikseli wokół obrazu.
# Stride=2 oznacza przesuwanie filtra co 2 piksele zamiast co 1 piksel.
kernel_size = kernel.shape[0]
padding = 0
stride = 2

# Sprawdzamy, czy skrypt jest uruchamiany zgodnie z założeniem zadania.
if padding != 0:
    raise ValueError("Ten przykład ma działać bez paddingu, więc padding musi wynosić 0.")

# KROK 5: Wykonujemy konwolucję.
# BORDER_CONSTANT pozwala OpenCV policzyć wartości również na brzegu,
# ale później usuniemy ten brzeg, aby uzyskać wynik bez paddingu.
# CV_32F jest potrzebne, ponieważ filtr zawiera wartości ujemne.
convolved = cv2.filter2D(
    image,
    ddepth=cv2.CV_32F,
    kernel=kernel,
    borderType=cv2.BORDER_CONSTANT,
)

# KROK 6: Usuwamy brzeg obrazu.
# Dla jądra 3x3 odcinamy 1 piksel z każdej strony.
# Pozostała część jest odpowiednikiem padding='valid'.
border = kernel_size // 2
valid = convolved[border:-border, border:-border]

# KROK 7: Stosujemy stride.
# [::2, ::2] wybiera co drugi wiersz i co drugą kolumnę,
# czyli zachowuje wynik co 2 piksele w pionie i poziomie.
result = valid[::stride, ::stride]

# KROK 8: Zamieniamy wartości na zakres poprawny dla obrazu 8-bitowego.
# Wartości poniżej 0 stają się 0, a powyżej 255 stają się 255.
result = np.clip(result, 0, 255).astype(np.uint8)

# KROK 9: Wyświetlamy rozmiary, aby zobaczyć wpływ paddingu i stride.
print(f"Rozmiar obrazu wejściowego: {image.shape[1]} x {image.shape[0]}")
print(f"Rozmiar po konwolucji bez paddingu: {valid.shape[1]} x {valid.shape[0]}")
print(f"Rozmiar wyniku przy stride={stride}: {result.shape[1]} x {result.shape[0]}")

# KROK 10: Zapisujemy przefiltrowany obraz na dysku.
if not cv2.imwrite(output_path, result):
    raise IOError(f"Nie udało się zapisać wyniku: {output_path}")

# KROK 11: Pokazujemy obraz oryginalny i wynik.
cv2.imshow("Oryginalny obraz", image)
cv2.imshow("Konwolucja: padding=0, stride=2", result)

# KROK 12: Czekamy na dowolny klawisz, a potem zamykamy okna.
cv2.waitKey(0)
cv2.destroyAllWindows()
