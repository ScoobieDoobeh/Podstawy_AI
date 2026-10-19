import cv2
import numpy as np

# Ścieżki do pliku wejściowego i pliku wynikowego.
input_path = r"C:\Users\anton\Downloads\wiewior.jpg"
output_path = r"C:\Users\anton\Downloads\wiewior_czarne_tlo.jpg"

# OpenCV wczytuje obraz jako tablicę NumPy w formacie BGR.
image = cv2.imread(input_path)

if image is None:
    raise FileNotFoundError(f"Nie udało się wczytać obrazu: {input_path}")

# image.shape[:2] zawiera kolejno wysokość i szerokość obrazu w pikselach.
# Prostokąt musi obejmować całe zwierzę, łącznie z końcem ogona.
height, width = image.shape[:2]
rect = (400, 100, min(1550, width - 400), min(1000, height - 100))

# Tworzymy maskę GrabCut o jednym elemencie na każdy piksel.
# GrabCut używa czterech wartości:
#   GC_BGD     - na pewno tło
#   GC_PR_BGD  - prawdopodobnie tło
#   GC_FGD     - na pewno pierwszy plan
#   GC_PR_FGD  - prawdopodobnie pierwszy plan
#
# Na początku cały obszar wewnątrz prostokąta jest "prawdopodobnym tłem".
# Później dokładniej wskażemy, gdzie znajduje się wiewiórka.
mask = np.zeros(image.shape[:2], np.uint8)
mask[:] = cv2.GC_PR_BGD

# Wszystko poza prostokątem traktujemy jako pewne tło.
left, top, rect_width, rect_height = rect
mask[:top, :] = cv2.GC_BGD
mask[top + rect_height:, :] = cv2.GC_BGD
mask[:, :left] = cv2.GC_BGD
mask[:, left + rect_width:] = cv2.GC_BGD

# Przybliżony obrys wiewiórki. Każda para [x, y] oznacza punkt obrazu:
# x rośnie w prawo, a y rośnie w dół.
# Obrys nie musi być idealny, ponieważ GrabCut później dopasuje krawędź
# do kolorów i tekstury sierści.
silhouette = np.array(
    [
        [480, 690], [455, 610], [475, 520], [505, 445], [500, 350],
        [525, 165], [575, 360], [635, 300], [690, 430], [820, 470],
        [980, 470], [1110, 550], [1180, 650], [1240, 610],
        [1370, 500], [1530, 510], [1700, 610], [1850, 770],
        [1930, 950], [1900, 1080], [1780, 1010], [1640, 880],
        [1480, 790], [1320, 735], [1190, 760], [1130, 870],
        [1060, 950], [960, 985], [820, 980], [700, 920],
        [630, 825], [550, 790],
    ],
    dtype=np.int32,
)

# Wewnątrz obrysu ustawiamy "prawdopodobny pierwszy plan".
cv2.fillPoly(mask, [silhouette], cv2.GC_PR_FGD)

# Gałąź znajduje się bezpośrednio pod wiewiórką i ma podobne kolory.
# Ten wielokąt oznaczamy jako pewne tło, aby gałąź nie została wycięta
# razem ze zwierzęciem.
branch_background = np.array(
    [
        [400, 1035], [650, 1015], [800, 1010], [950, 1015],
        [1080, 980], [1240, 990], [1360, 1040], [1360, height],
        [400, height],
    ],
    dtype=np.int32,
)
cv2.fillPoly(mask, [branch_background], cv2.GC_BGD)

# Mały fragment tła pod przednimi łapkami i przed brzuchem ma podobny kolor
# do sierści, dlatego GrabCut może błędnie pozostawić go jako pierwszy plan.
# Ten obszar kończy się przed tylną łapką i nie obejmuje samego zwierzęcia.
background_under_paws = np.array(
    [
        [540, 760], [625, 770], [690, 815], [700, 930],
        [680, 1010], [560, 1010],
    ],
    dtype=np.int32,
)
cv2.fillPoly(mask, [background_under_paws], cv2.GC_BGD)

# Tworzymy pewny punkt pierwszego planu wewnątrz obrysu.
# Najpierw rysujemy obrys na osobnej czarno-białej masce:
# 0 oznacza czarne tło, a 255 biały obszar wiewiórki.
seed = np.zeros(mask.shape, np.uint8)
cv2.fillPoly(seed, [silhouette], 255)

# Erozja zmniejsza biały obszar od krawędzi.
# Dzięki temu pewny pierwszy plan leży w środku zwierzęcia, a nie na granicy
# z tłem. Rozmiar (41, 41) określa, jak mocno obszar zostanie zmniejszony.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
sure_foreground = cv2.erode(seed, kernel, iterations=2)
mask[sure_foreground > 0] = cv2.GC_FGD

# GrabCut przechowuje wewnętrzny model kolorów tła i pierwszego planu.
# Tablice muszą mieć rozmiar (1, 65), zgodnie z wymaganiami OpenCV.
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Uruchamiamy GrabCut:
# - None, ponieważ korzystamy z własnej maski, a nie tylko z prostokąta,
# - 8 oznacza liczbę iteracji dopasowywania maski,
# - GC_INIT_WITH_MASK mówi OpenCV, że maska została przygotowana ręcznie.
cv2.grabCut(
    image,
    mask,
    None,
    bgdModel,
    fgdModel,
    8,
    cv2.GC_INIT_WITH_MASK,
)

# Po GrabCut wybieramy zarówno pewny, jak i prawdopodobny pierwszy plan.
# np.where tworzy maskę binarną:
# 1 = zachowaj piksel wiewiórki
# 0 = zamień piksel na czarne tło
foreground = np.where(
    (mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
    0,
    1
).astype("uint8")

# Dodatkowe usunięcie dolnej części gałęzi.
# Współrzędne NumPy mają kolejność [y, x], dlatego:
# [1015:, :1360] oznacza obszar od y=1015 w dół i od x=0 do x=1360.
# Ogon znajduje się dalej na prawo, więc pozostaje nienaruszony.
foreground[1015:, :1360] = 0

# Dodajemy trzeci wymiar do maski, aby można było pomnożyć ją przez obraz BGR.
# Piksele tła stają się czarne, a piksele wiewiórki zachowują oryginalne kolory.
result = image * foreground[:, :, np.newaxis]

# Podgląd obrazu oryginalnego i wyniku.
cv2.imshow("Oryginal", image)
cv2.imshow("Wiewiórka na czarnym tle", result)

# Czekamy na naciśnięcie dowolnego klawisza, a następnie zamykamy okna.
cv2.waitKey(0)
cv2.destroyAllWindows()

# Zapisujemy gotowy obraz i zgłaszamy błąd, jeśli zapis się nie powiedzie.
if not cv2.imwrite(output_path, result):
    raise IOError(f"Nie udało się zapisać wyniku: {output_path}")
