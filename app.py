import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)

# --- CARGAR ARTEFACTOS CON MANEJO DE ERRORES ---
print("Inicializando sistema...")
try:
    with open('model_final.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
    with open('vectorizer.pkl', 'rb') as f_vec:
        vectorizer = pickle.load(f_vec)
    
    # Inicializar LIME una sola vez
    explainer = LimeTextExplainer(class_names=['No Spam', 'Spam'])
    print("✅ Sistema cargado correctamente.")

except Exception as e:
    print(f"❌ ERROR CRÍTICO: {e}")
    model = None
    vectorizer = None

# Función auxiliar para LIME
def predict_proba_pipeline(texts):
    vec = vectorizer.transform(texts)
    return model.predict_proba(vec)

# --- RUTAS ---

@app.route('/')
def home():
    # Sirve el archivo index.html
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'status': 'error', 'message': 'El modelo no está cargado en el servidor.'}), 500

    try:
        data = request.get_json(force=True)
        texto = str(data.get('email_text', '')).strip()

        if not texto:
            return jsonify({'status': 'error', 'message': 'El texto del correo está vacío.'}), 400

        # 1. Predecir Clase
        vec = vectorizer.transform([texto])
        prediccion = int(model.predict(vec)[0])

        # 2. Generar Explicación LIME
        # num_features=6 para mostrar las 6 palabras más influyentes
        exp = explainer.explain_instance(texto, predict_proba_pipeline, num_features=6)
        
        return jsonify({
            'status': 'success',
            'prediccion_spam': prediccion,
            'explicacion': exp.as_list()
        })

    except Exception as e:
        print(f"Error en predicción: {e}")
        return jsonify({'status': 'error', 'message': 'Ocurrió un error interno al procesar.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)