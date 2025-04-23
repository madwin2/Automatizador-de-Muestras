FROM python:3.11-slim

# Instalar Tesseract, libGL y otras dependencias necesarias
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crear carpeta de trabajo
WORKDIR /app

# Copiar todos los archivos al contenedor
COPY . .

# Instalar las dependencias Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exponer puerto de Render
EXPOSE 10000

# Ejecutar la app con Gunicorn (ajustá si tu archivo o variable se llaman diferente)
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
