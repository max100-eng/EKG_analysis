title: "Análisis de ECG con DenseNet (8 Clases)"
emoji: "🩺"
colorFrom: "gray"
SDK: "gradio"
app_file: "app.py"
🩺 Análisis de ECG con DenseNet201

Aplicación web (Gradio) para clasificar imágenes de ECG en 8 condiciones cardíacas usando un modelo basado en DenseNet201.

Permite subir una imagen de ECG, preprocesarla y obtener la predicción con probabilidades por clase.

🚀 Características
Interfaz web sencilla con Gradio.
Preprocesado estándar de imágenes (rescalado, normalización).
Carga de modelo (PyTorch / TensorFlow).
Salida: clase más probable + probabilidades por clase.
Ejemplo de clases:
Normal
Fibrilación auricular
Taquicardia supraventricular
Bloqueo AV
Bloqueo de rama
Extrasístole
Infarto agudo
Otra arritmia

⚠️ Ajusta la lista según tu dataset real.

📦 Requisitos

Archivo requirements.txt mínimo:

gradio
torch
torchvision
Pillow
numpy
opencv-python

📂 Estructura recomendada
mi-ecg-app/
├── app.py              # Gradio app (interfaz)
├── model.py            # Funciones de carga/modelo
├── preprocess.py       # Funciones de preprocesado
├── weights/
│   └── densenet_ecg.pth
├── README.md
└── requirements.txt

▶️ Uso

Ejecutar la aplicación:

python app.py


Se abrirá en tu navegador (por defecto en http://127.0.0.1:7860).

⚙️ Ejemplo mínimo de app.py
import io
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import gradio as gr
from torchvision import models

CLASSES = [
    "Normal", "Fibrilación auricular", "Taquicardia supraventricular",
    "Bloqueo AV", "Bloqueo de rama", "Extrasístole",
    "Infarto agudo", "Otra arritmia"
]

def load_model(weights_path="weights/densenet_ecg.pth", device="cpu"):
    model = models.densenet201(pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, len(CLASSES))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

DEVICE = "cpu"
MODEL = load_model()

def preprocess_image(img: Image.Image):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    if img.mode != "RGB":
        img = img.convert("RGB")
    return transform(img).unsqueeze(0)

def predict_ecg(image):
    x = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

iface = gr.Interface(
    fn=predict_ecg,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Análisis de ECG con DenseNet201",
    description="Sube una imagen de ECG y obtén la clasificación en 8 clases."
)

if __name__ == "__main__":
    iface.launch()

📊 Consejos
Normalizar siempre con los mismos parámetros del entrenamiento.
Si las imágenes incluyen varias derivaciones, considera recortes o segmentación.
Añade métricas de validación (accuracy, F1, matriz de confusión).
⚖️ Aviso

Esta app es solo con fines educativos. No debe usarse para diagnóstico clínico sin validación médica y aprobación regulatoria.
