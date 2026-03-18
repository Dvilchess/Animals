#!/usr/bin/env python3
"""
scripts/cargar_dataset.py
Descomprime el ZIP del dataset en data/raw/

Uso:
    python scripts/cargar_dataset.py --zip data/animales.zip
    python scripts/cargar_dataset.py --zip data/animales.zip --dest data/raw
"""

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path

CLASES_ESPERADAS = [
    "cane", "cavallo", "elefante", "farfalla", "gallina",
    "gatto", "mucca", "pecora", "ragno", "scoiattolo"
]


def verificar_si_ya_existe(dest: Path) -> bool:
    """Revisa si el dataset ya está descomprimido."""
    clases_encontradas = [c for c in CLASES_ESPERADAS if (dest / c).is_dir()]
    if len(clases_encontradas) == 10:
        total = sum(len(list((dest / c).glob("*"))) for c in CLASES_ESPERADAS)
        print(f"✅ Dataset ya presente: 10/10 clases, {total:,} imágenes")
        return True
    if clases_encontradas:
        print(f"⚠️  Dataset incompleto: {len(clases_encontradas)}/10 clases")
    return False


def encontrar_raiz_clases(base: Path) -> Path:
    """
    Detecta la carpeta que contiene las subcarpetas de clases,
    sin importar cómo venga estructurado el ZIP.

    Ej: zip puede tener raw-img/cane/ o directo cane/
    """
    for candidato in [base, *sorted(base.rglob("*"))]:
        if candidato.is_dir():
            subdirs = {d.name for d in candidato.iterdir() if d.is_dir()}
            if subdirs & set(CLASES_ESPERADAS):
                return candidato
    return None


def descomprimir(zip_path: str, dest_dir: str = "data/raw"):
    zip_file = Path(zip_path)
    dest = Path(dest_dir)

    # ─── Validar que el ZIP existe ────────────────────────
    if not zip_file.exists():
        print(f"❌ No se encontró el archivo: {zip_path}")
        print()
        print("¿Cómo subir el ZIP al Codespace?")
        print("  1. En el panel izquierdo de VS Code, abre la carpeta 'data/'")
        print("  2. Arrastra tu ZIP desde tu PC a esa carpeta")
        print("  3. Vuelve a correr este script")
        sys.exit(1)

    # ─── Verificar si ya está descomprimido ───────────────
    dest.mkdir(parents=True, exist_ok=True)
    if verificar_si_ya_existe(dest):
        return

    # ─── Descomprimir en carpeta temporal ─────────────────
    tmp = Path("data/.tmp")
    tmp.mkdir(parents=True, exist_ok=True)

    size_mb = zip_file.stat().st_size / 1_000_000
    print(f"📦 Descomprimiendo '{zip_file.name}' ({size_mb:.0f} MB)...")

    with zipfile.ZipFile(zip_file, "r") as zf:
        total = len(zf.namelist())
        for i, member in enumerate(zf.namelist(), 1):
            zf.extract(member, tmp)
            if i % 2000 == 0 or i == total:
                pct = i / total * 100
                print(f"  {pct:.0f}% ({i:,}/{total:,} archivos)", end="\r")

    print(f"\n✅ Extraído")

    # ─── Detectar estructura interna del ZIP ──────────────
    raiz = encontrar_raiz_clases(tmp)
    if not raiz:
        print("❌ No se encontraron las carpetas de clases dentro del ZIP.")
        print(f"   Se esperaban subcarpetas como: {CLASES_ESPERADAS[:3]}...")
        print()
        print("   Contenido encontrado:")
        for p in sorted(tmp.rglob("*"))[:20]:
            print(f"     {p.relative_to(tmp)}")
        shutil.rmtree(tmp)
        sys.exit(1)

    # ─── Mover cada clase a data/raw/ ─────────────────────
    print(f"\n📁 Organizando clases en '{dest_dir}/':")
    for clase_dir in sorted(raiz.iterdir()):
        if clase_dir.is_dir() and clase_dir.name in CLASES_ESPERADAS:
            destino = dest / clase_dir.name
            if not destino.exists():
                shutil.move(str(clase_dir), str(destino))
            count = len(list((dest / clase_dir.name).glob("*")))
            print(f"  ✅ {clase_dir.name:<14} {count:>5} imágenes")

    # ─── Limpiar tmp ──────────────────────────────────────
    shutil.rmtree(tmp, ignore_errors=True)

    # ─── Resumen ──────────────────────────────────────────
    encontradas = [c for c in CLASES_ESPERADAS if (dest / c).is_dir()]
    total_imgs = sum(len(list((dest / c).glob("*"))) for c in encontradas)

    print()
    print(f"🎉 Dataset listo: {len(encontradas)}/10 clases — {total_imgs:,} imágenes")
    print(f"   Ubicación: {dest.resolve()}")

    if len(encontradas) < 10:
        faltantes = set(CLASES_ESPERADAS) - set(encontradas)
        print(f"⚠️  Clases faltantes: {faltantes}")
        sys.exit(1)

    print()
    print("Próximo paso → entrenar el modelo:")
    print("  python scripts/train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descomprime el dataset de animales en data/raw/"
    )
    parser.add_argument(
        "--zip",
        required=True,
        help="Ruta al ZIP (ej: data/animales.zip)",
    )
    parser.add_argument(
        "--dest",
        default="data/raw",
        help="Carpeta destino (default: data/raw)",
    )
    args = parser.parse_args()
    descomprimir(args.zip, args.dest)
