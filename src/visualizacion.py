"""
src/visualizacion.py
Funciones de visualización del notebook original:
  - Mostrar imágenes de ejemplo
  - Curvas de entrenamiento (loss / accuracy)
  - Matriz de confusión
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def mostrar_imagenes(dataset_dir: str, n_categorias: int = 3):
    """
    Muestra una imagen de ejemplo por categoría.
    (del notebook original: show_images)

    Args:
        dataset_dir  : ruta al directorio con subcarpetas por clase
        n_categorias : cuántas categorías mostrar (None = todas)
    """
    categories = sorted(os.listdir(dataset_dir))
    if n_categorias:
        categories = categories[:n_categorias]

    fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), 4))
    if len(categories) == 1:
        axes = [axes]

    for ax, category in zip(axes, categories):
        path = os.path.join(dataset_dir, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img_array = cv2.imread(img_path)
            if img_array is None:
                continue
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            ax.imshow(img_array)
            ax.set_title(category)
            ax.axis("off")
            break

    plt.tight_layout()
    plt.show()


def graficar_entrenamiento(history):
    """
    Curvas de loss y accuracy por época.
    (del notebook original)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history["loss"], label="loss")
    axes[0].plot(history.history["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Curva de Pérdida")
    axes[0].legend()

    # Accuracy
    axes[1].plot(history.history["accuracy"], label="accuracy")
    axes[1].plot(history.history["val_accuracy"], label="val_accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Curva de Precisión")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def graficar_matriz_confusion(model, X_test, y_test, categories):
    """
    Genera y grafica la matriz de confusión.
    (del notebook original)
    """
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    conf_matrix = confusion_matrix(y_test, y_pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return conf_matrix
