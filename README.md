# Proyecto Detección de Spam (AA3 - MLOps)

Este proyecto implementa un pipeline completo de Machine Learning (MLOps) para un modelo de clasificación de Spam. El modelo (un Árbol de Decisión) se entrena en un notebook, se serializa y luego se despliega como una API REST dentro de un contenedor Docker.

## 📚 Componentes del Proyecto

* `/model_final.pkl`: El modelo de Árbol de Decisión entrenado (post-SMOTE y GridSearch).
* `/vectorizer.pkl`: El objeto `TfidfVectorizer` entrenado.
* `/app.py`: El script de la API REST (Flask) que sirve el modelo.
* `/requirements.txt`: Las librerías de Python necesarias.
* `/Dockerfile`: Las instrucciones para construir la imagen de Docker.

---

## ⚙️ Prerrequisitos

Para ejecutar este proyecto localmente, necesitas:

1.  **Git** (para clonar el repositorio).
2.  **Docker Desktop** (asegúrate de que esté instalado y corriendo en tu PC).
3.  **Postman** (o `curl`) (para probar la API).

---

## 🚀 Instalación y Ejecución Local

Sigue estos pasos en tu terminal para levantar el servicio:

Instalación y Ejecución Local
Sigue estos pasos en tu terminal para levantar el servicio:

## 1. Clonar el Repositorio
Bash

# Reemplaza la URL por la de tu repositorio
git clone https://github.com/jhuertasn/proyecto-spam-aa3.git
cd proyecto-spam-aa3
## 2. Construir la Imagen Docker
Este comando lee el Dockerfile e instala todas las dependencias.

Bash

docker build -t spam-api:v1 .
## 3. Ejecutar el Contenedor
Este comando inicia la API y mapea el puerto 8080 de tu PC al puerto 5000 del contenedor.

Bash

docker run -d -p 8080:5000 --name mi-api-spam spam-api:v1
## 4. Detener el Contenedor (Opcional)
Si necesitas detener y eliminar el contenedor, puedes usar:

Bash

docker stop mi-api-spam
docker rm mi-api-spam

## Cómo Probar la API (Usando Postman)
Abrir Postman y crear una nueva solicitud.

Método: POST

URL: http://localhost:8080/predict

Pestaña "Body" -> seleccionar "raw" -> seleccionar "JSON".

Prueba de NO SPAM (Ham)
Pega esto en el body de Postman:

JSON

{
    "email_text": "Hey, are you coming to the meeting tomorrow?"
}
Respuesta Esperada:

JSON

{
    "prediccion_spam": 0,
    "status": "success"
}
Prueba de SPAM
Pega esto en el body de Postman:

JSON

{
    "email_text": "WINNER!! You have been selected to receive a $1000 prize. Click this link to claim now!"
}
Respuesta Esperada:

JSON

{
    "prediccion_spam": 1,
    "status": "success"
}