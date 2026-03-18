"""
src/model.py
Arquitectura con EfficientNetB0 (Transfer Learning)
"""
"""se cambio el entrenamiento con imagenes locales para utilizar efficientnet"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


def construir_modelo(input_shape, num_clases: int) -> tf.keras.Model:
    # Base preentrenada de Google (sin la última capa)
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base.trainable = False  # congelar pesos de EfficientNet

    # Tu clasificador encima
    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_clases, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)


def compilar_modelo(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def entrenar(model, train_gen, val_gen, epochs: int = 10):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    return model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
    )