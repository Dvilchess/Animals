# 🐾 Clasificador de Animales — CNN

Clasificador de imágenes de 10 tipos de animales usando CNN con TensorFlow/Keras.

**Integrantes:** Matías Soto · José Donoso · Dante Vilches · Matías Navarrete

---

## 🐾 Clases

`cane` · `cavallo` · `elefante` · `farfalla` · `gallina` · `gatto` · `mucca` · `pecora` · `ragno` · `scoiattolo`

---

## 📁 Estructura

```
animales-classifier/
├── src/
│   ├── data.py          # Carga y preprocesamiento
│   ├── model.py         # Arquitectura CNN
│   └── visualizacion.py # Gráficos y métricas
├── scripts/
│   └── train.py         # Pipeline completo
├── data/
│   └── raw/             # Dataset (no incluido en git)
├── models/              # Modelos entrenados
└── requirements.txt
```

---

## 🚀 Uso rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Colocar el dataset en data/raw/
#    (una carpeta por clase: data/raw/cane/, data/raw/gatto/, etc.)

# 3. Entrenar
python scripts/train.py

# 4. Con opciones
python scripts/train.py --data-dir data/raw --epochs 10
```

---

## 🧠 Modelo (notebook original)

```
Input (100×100×3)
  Conv2D(64, 3×3, relu)
  MaxPooling2D(2×2)
  Conv2D(64, 3×3, relu)
  MaxPooling2D(2×2)
  Flatten
  Dense(64, relu)
  Dense(10, softmax)
```

- **Optimizador:** Adam
- **Loss:** sparse_categorical_crossentropy
- **Épocas:** 10
- **Split:** 80% train / 20% test
