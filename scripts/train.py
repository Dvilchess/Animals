"""
scripts/train.py
Pipeline completo de entrenamiento — replica exactamente el notebook original.

Uso:
    python scripts/train.py
    python scripts/train.py --data-dir data/raw --epochs 10
"""

import argparse
import os

from src.data import cargar_dataset, preparar_datos
from src.model import compilar_modelo, construir_modelo, entrenar
from src.visualizacion import graficar_entrenamiento, graficar_matriz_confusion


def main(data_dir: str = "data/raw", epochs: int = 10, guardar_modelo: bool = True):

    print("=" * 50)
    print("  Clasificador de Animales — CNN")
    print("=" * 50)

    # ─── 1. Cargar dataset ────────────────────────────────
    print("\n📂 Cargando dataset...")
    data, categories = cargar_dataset(data_dir)
    print(f"Clases encontradas: {categories}")

    # ─── 2. Preparar datos ────────────────────────────────
    print("\n🔀 Dividiendo en entrenamiento/prueba (80/20)...")
    X_train, X_test, y_train, y_test = preparar_datos(data)

    # ─── 3. Construir modelo ──────────────────────────────
    print("\n🧠 Construyendo modelo CNN...")
    model = construir_modelo(
        input_shape=X_train.shape[1:],
        num_clases=len(categories),
    )
    model = compilar_modelo(model)
    model.summary()

    # ─── 4. Entrenar ──────────────────────────────────────
    print(f"\n🚀 Entrenando por {epochs} épocas...")
    history = entrenar(model, X_train, y_train, X_test, y_test, epochs=epochs)

    # ─── 5. Visualizar resultados ─────────────────────────
    print("\n📊 Generando gráficos...")
    graficar_entrenamiento(history)
    graficar_matriz_confusion(model, X_test, y_test, categories)

    # ─── 6. Guardar modelo ────────────────────────────────
    if guardar_modelo:
        os.makedirs("models", exist_ok=True)
        model.save("models/modelo_animales.keras")
        print("\n✅ Modelo guardado en: models/modelo_animales.keras")

    # ─── 7. Resultado final ───────────────────────────────
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n🎯 Accuracy final en test: {acc:.4f} ({acc*100:.1f}%)")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar clasificador de animales")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--no-guardar", action="store_true")
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        epochs=args.epochs,
        guardar_modelo=not args.no_guardar,
    )
