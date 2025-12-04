import pickle
from flask import Flask, request, jsonify
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
# --- NUEVO: Importar LIME ---
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)

# --- CARGAR ARTEFACTOS ---
try:
    with open('model_final.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
    
    with open('vectorizer.pkl', 'rb') as f_vec:
        vectorizer = pickle.load(f_vec)
    
    print("¡Modelo y Vectorizador cargados exitosamente!")

    # --- NUEVO: Inicializar el Explicador LIME ---
    # Le decimos que las clases son 'No Spam' y 'Spam'
    explainer = LimeTextExplainer(class_names=['No Spam', 'Spam'])

except FileNotFoundError:
    print("Error: No se encontraron los archivos .pkl.")
    model = None
    vectorizer = None

# --- NUEVO: Función Puente para LIME ---
# LIME necesita una función que reciba TEXTO PURO y devuelva PROBABILIDADES.
# Nuestro modelo necesita VECTORES. Esta función conecta ambos mundos.
def predict_proba_pipeline(texts):
    # 1. Convertir texto a vector
    vec = vectorizer.transform(texts)
    # 2. Devolver probabilidad (DecisionTree tiene predict_proba)
    return model.predict_proba(vec)

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({'status': 'error', 'message': 'Modelo no cargado'}), 500

    try:
        data = request.get_json(force=True)
        texto_original = str(data['email_text']) # Texto puro
        
        # 1. Predicción Normal (como antes)
        texto_lista = [texto_original]
        X_nuevo = vectorizer.transform(texto_lista)
        prediccion = model.predict(X_nuevo)[0]
        
        # --- NUEVO: Generar Explicación con LIME ---
        # Pedimos a LIME que explique ESTA instancia específica
        # num_features=5 significa "dame las 5 palabras más importantes"
        exp = explainer.explain_instance(
            texto_original, 
            predict_proba_pipeline, 
            num_features=5
        )
        
        # Convertimos la explicación a una lista fácil de leer (JSON)
        # Ejemplo de salida: [('winner', 0.85), ('free', 0.50)]
        explicacion_lista = exp.as_list()

        return jsonify({
            'status': 'success',
            'prediccion_spam': int(prediccion), # 0 o 1
            'explicacion': explicacion_lista    # ¡La justificación!
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)