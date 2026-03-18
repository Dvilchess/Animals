"""
scripts/train.py
Pipeline de entrenamiento usando generadores (bajo consumo de RAM).
"""

import argparse
import os

from src.data import cargar_dataset
from src.model import compilar_modelo, construir_modelo
from src.visualizacion import graficar_entrenamiento, graficar_matriz_confusion
import numpy as np


def main(data_dir: str = "data/raw", epochs: int = 10):

    print("=" * 50)
    print("  Clasificador de Animales — CNN")
    print("=" * 50)

    # 1. Cargar datos con generador
    print("\n📂 Cargando dataset...")
    train_gen, val_gen, categories = cargar_dataset(data_dir)

    # 2. Construir modelo
    print("\n🧠 Construyendo modelo CNN...")
    model = construir_modelo(
        input_shape=(224, 224, 3),
        num_clases=len(categories),
    )
    model = compilar_modelo(model)
    model.summary()

    # 3. Entrenar
    print(f"\n🚀 Entrenando por {epochs} épocas...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
    )

    # 4. Guardar modelo
    os.makedirs("models", exist_ok=True)
    model.save("models/modelo_animales.keras")
    print("\n✅ Modelo guardado en: models/modelo_animales.keras")

    # 5. Resultado final
    loss, acc = model.evaluate(val_gen, verbose=0)
    print(f"\n🎯 Accuracy final en validación: {acc:.4f} ({acc*100:.1f}%)")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args.data_dir, args.epochs)