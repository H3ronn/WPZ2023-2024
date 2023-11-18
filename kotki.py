import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.preprocessing import image

# Ścieżka do wytrenowanego modelu (np. MobileNetV2)
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = tf.keras.Sequential([hub.KerasLayer(model_url, input_shape=(224,224,3))])

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Tworzy batch składający się z jednego obrazu
    img_array /= 255.0  # Normalizacja do zakresu 0-1

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])

    return predicted_index

# Przykład użycia
img_path = 'pobrane.jpg'
predicted_index = classify_image(img_path)

# Uzyskaj nazwę rasy używając mapy
rasa_mapa = {
    285: 'Syjamski',
    288: 'Maine Coon',
    282: 'Brytyjski',
    284: 'Pers'
    # Dodaj tutaj inne ras
}

predicted_rasa = rasa_mapa.get(predicted_index, "Nieznana rasa")
print(f'Przewidywana rasa kota: {predicted_rasa}')