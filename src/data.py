"""
src/data.py
Carga del dataset usando generador por lotes para no agotar la RAM.
"""

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def cargar_dataset(dataset_dir: str):
    """Retorna los generadores de train y validación."""
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"Ruta inválida: '{dataset_dir}'")

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        subset="training",
        shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        subset="validation",
        shuffle=False,
    )

    categories = list(train_gen.class_indices.keys())
    print(f"Clases encontradas: {categories}")
    print(f"Imágenes entrenamiento: {train_gen.samples}")
    print(f"Imágenes validación   : {val_gen.samples}")

    return train_gen, val_gen, categories