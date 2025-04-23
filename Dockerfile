# Imagen base liviana con Python
FROM python:3.11-slim

# Instalar tesseract y dependencias necesarias
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crear carpeta de trabajo
WORKDIR /app

# Copiar todos los archivos del proyecto al contenedor
COPY . .

# Instalar las dependencias de Python desde requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exponer el puerto en el que corre tu app (Render usa el 10000 por defecto)
EXPOSE 10000

# Comando para ejecutar la app con Gunicorn (ajustá si usás otra cosa)
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
