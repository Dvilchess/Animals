"""
src/data.py
Carga, preprocesamiento y división del dataset de animales.

Basado en el notebook original: Animales_Final.ipynb
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Tamaño al que se redimensionan todas las imágenes (del notebook original: 100x100)
IMG_SIZE = (100, 100)


def cargar_dataset(dataset_dir: str):
    """
    Recorre el directorio del dataset, carga cada imagen,
    la convierte de BGR a RGB, la redimensiona a 100x100
    y le asigna una etiqueta numérica según su categoría.

    Retorna:
        data      : lista de [img_array, label]
        categories: lista de nombres de clases
    """
    if not os.path.isdir(dataset_dir):
        raise ValueError(
            f"La ruta especificada no es válida: '{dataset_dir}'\n"
            "Asegúrate de que el dataset esté en data/raw/"
        )

    categories = sorted(os.listdir(dataset_dir))  # orden consistente
    data = []

    for category in categories:
        path = os.path.join(dataset_dir, category)
        if not os.path.isdir(path):
            continue  # ignorar archivos sueltos

        archivos = os.listdir(path)
        print(f"  Cargando '{category}' ... {len(archivos)} imágenes")
        label = categories.index(category)

        for img_name in archivos:
            try:
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                img_array = cv2.resize(img_array, IMG_SIZE)
                data.append([img_array, label])
            except Exception:
                pass  # ignorar imágenes corruptas

    print(f"\nTotal imágenes cargadas: {len(data)}")
    return data, categories


def preparar_datos(data, test_size: float = 0.2, random_state: int = 42):
    """
    Separa features y etiquetas, normaliza a [0,1]
    y divide en entrenamiento/prueba (80/20).

    Retorna:
        X_train, X_test, y_train, y_test
    """
    X = []
    y = []

    for features, label in data:
        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Normalizar: dividir entre 255 (del notebook original)
    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Entrenamiento : {X_train.shape[0]} imágenes")
    print(f"Prueba        : {X_test.shape[0]} imágenes")
    print(f"Shape entrada : {X_train.shape[1:]}")

    return X_train, X_test, y_train, y_test
