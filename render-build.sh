#!/usr/bin/env bash

# Habilitar errores para depurar mejor
set -o errexit
set -o nounset

# Instalar Tesseract
apt-get update
apt-get install -y tesseract-ocr

# Instalar dependencias de Python
pip install -r requirements.txt
