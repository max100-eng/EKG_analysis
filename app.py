import gradio as gr
from transformers import pipeline

# 1. Cargar el modelo de Hugging Face
# IMPORTANTE: Reemplaza "nombre-de-usuario/nombre-del-modelo" con el modelo real que uses.
# El modelo debe estar diseñado para clasificar imágenes.
try:
    ecg_classifier = pipeline("image-classification", model="adi9-48/ecg_classification_model")
except Exception as e:
    # Esto es por si el modelo no existe o hay un error al cargarlo
    print(f"Error al cargar el modelo: {e}")
    # Puedes usar un modelo de demostración o un modelo predeterminado
    ecg_classifier = lambda x: [{"label": "Clasificación de demostración", "score": 1.0}]

def analyze_ecg(image):
    """
    Esta función analiza la imagen del ECG y devuelve una clasificación.
    """
    # La función `pipeline` de Transformers maneja todo el pre-procesamiento
    # y la inferencia por ti.
    predictions = ecg_classifier(image)
    
    # Formatear las predicciones para que Gradio pueda mostrarlas en una etiqueta
    results = {p["label"]: p["score"] for p in predictions}
    return results

# 2. Crear la interfaz de usuario con Gradio
# gr.Interface conecta la función de análisis con los componentes de entrada y salida
demo = gr.Interface(
    fn=analyze_ecg,
    inputs=gr.Image(type="pil", label="Sube una imagen de ECG"),
    outputs=gr.Label(num_top_classes=3),
    title="Análisis de ECG con IA",
    description="Sube una imagen de tu electrocardiograma para obtener una clasificación."
)

# 3. Lanzar la aplicación
demo.launch()