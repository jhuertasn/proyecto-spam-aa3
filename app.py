import pickle
from flask import Flask, request, jsonify
import numpy as np
# Importamos las clases exactas para que pickle funcione sin problemas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

# --- 1. INICIALIZAR LA APP FLASK ---
app = Flask(__name__)

# --- 2. CARGAR LOS ARTEFACTOS (Modelo y Vectorizador) ---
try:
    with open('model_final.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
    
    with open('vectorizer.pkl', 'rb') as f_vec:
        vectorizer = pickle.load(f_vec)
    
    print("¡Modelo y Vectorizador cargados exitosamente!")

except FileNotFoundError:
    print("Error: No se encontraron los archivos .pkl. Asegúrate de que estén en la misma carpeta.")
    model = None
    vectorizer = None

# --- 3. DEFINIR EL ENDPOINT DE PREDICCIÓN ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({'status': 'error', 'message': 'Modelo no cargado'}), 500

    try:
        # Obtener los datos JSON de la solicitud (enviados desde Postman)
        data = request.get_json(force=True)
        
        # Extraer el texto del SMS
        # Lo ponemos en una lista porque TfidfVectorizer espera una lista de documentos
        sms_text = [str(data['email_text'])]
        
        # Preprocesar el texto con el vectorizador que entrenamos
        X_nuevo = vectorizer.transform(sms_text)
        
        # Realizar la predicción
        prediccion = model.predict(X_nuevo)
        
        # Convertir la predicción (que es numpy) a un int nativo de Python
        resultado = int(prediccion[0])
        
        # Retornar la respuesta en formato JSON
        return jsonify({
            'status': 'success',
            'prediccion_spam': resultado # 0 (No Spam) o 1 (Spam)
        })

    except Exception as e:
        # Capturar cualquier otro error
        return jsonify({'status': 'error', 'message': str(e)}), 400

# --- 4. COMANDO DE INICIO ---
if __name__ == '__main__':
    # Esto permite ejecutar 'python app.py' para pruebas locales
    app.run(host='0.0.0.0', port=5000, debug=True)