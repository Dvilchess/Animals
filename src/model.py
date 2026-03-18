"""
src/model.py
Arquitectura CNN, compilación y entrenamiento.

Basado en el notebook original: Animales_Final.ipynb
"""

import tensorflow as tff
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def construir_modelo(input_shape, num_clases: int) -> tf.keras.Model:
    """
    CNN del notebook original:
      - Conv2D(64) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(64) → Dense(n_clases)

    Args:
        input_shape : forma de entrada, ej: (100, 100, 3)
        num_clases  : cantidad de clases (10 animales)

    Retorna:
        modelo sin compilar
    """
    model = Sequential([
        Conv2D(64, (3, 3), input_shape=input_shape, activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(num_clases, activation="softmax"),
    ])
    return model


def compilar_modelo(model: tf.keras.Model) -> tf.keras.Model:
    """
    Compilación del notebook original:
      - Optimizador : Adam
      - Loss        : sparse_categorical_crossentropy
      - Métrica     : accuracy
    """
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def entrenar(model, X_train, y_train, X_test, y_test, epochs: int = 10):
    """
    Entrena el modelo tal como en el notebook original (10 épocas).

    Retorna:
        history : objeto con las curvas de loss y accuracy
    """
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
    )
    return history
