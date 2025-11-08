# 1. Imagen Base (Python 3.9, la misma que Colab)
FROM python:3.10-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar el archivo de requerimientos primero (para caché de Docker)
COPY requirements.txt .

# 4. Instalar las librerías
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto de los archivos del proyecto (.py, .pkl)
COPY . .

# 6. Exponer el puerto que Gunicorn usará
EXPOSE 5000

# 7. Comando para iniciar el servidor web Gunicorn (producción)
# Esto ejecuta el 'app' (la variable Flask) dentro de tu archivo 'app.py'
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]