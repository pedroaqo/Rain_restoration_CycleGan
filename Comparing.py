import glob
import cv2
import os
import pandas as pd

index = {}  # Armazenar o nome da imagem e os histogramas
images = {} # Armezar as próprias imagens

# Pegar as imagens na pasta
for imagePath in glob.glob(os.getcwd() + "\*.png"):
    #extrair o nome do arquivo da imagem (considerado único) e
    # carregar a imagem, atualizando o dicionário de imagens
    filename = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
    images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #extrair um histograma de cores RGB da imagem,
    # usando 8 caixas por canal, normalizar e atualizar o índice
    hist = cv2.calcHist([image], [0, 1], None, [8, 8],
                        [0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    index[filename] = hist

# Metodos para calculo do histograma
OPENCV_METHODS = (
    ("Correlation", cv2.HISTCMP_CORREL),
    ("Chi-Squared", cv2.HISTCMP_CHISQR),
    ("Intersection", cv2.HISTCMP_INTERSECT),
    ("Bhattacharyya", cv2.HISTCMP_BHATTACHARYYA))
imagem_analisada = '\\y.png'

lista_resultados = []
lista_methodName = []

for (methodName, method) in OPENCV_METHODS:
    results = {}
    reverse = False
    # se estivermos usando a Correlation ou Intersection
    # classificar os resultados na ordem inversa
    if methodName in ("Correlation", "Intersection"):
        reverse = True
    for (k, hist) in index.items():
        # Calcular a distancia entre os dois histogramas usando os       metodos
        # Atualizar o dicionario de resultados
        d = cv2.compareHist(index[os.getcwd() + imagem_analisada],    hist, method)
        results[k] = d
    # Ordenar os resultados
    #print(methodName)
    lista_methodName.append(methodName)
    results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)
    lista_resultados.append(results)

for i in range(len(lista_methodName)):
   lista_resultados[i] =  pd.DataFrame(lista_resultados[i])
   lista_resultados[i]['Metodo'] = lista_methodName[i]
df = pd.concat(lista_resultados)
print(pd.DataFrame(lista_resultados[2]))
a = 2.400754
b = 1.711870
print(f"Resultado: {b} / {a} * 100% = {b/a*100}%")