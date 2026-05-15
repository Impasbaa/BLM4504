import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

def classify_image(image_path):
    print("ResNet50 Sınıflandırma Modeli Yükleniyor...")
    # Hazır ResNet50 modeli
    model = ResNet50(weights='imagenet')
    
    img = image.load_img(image_path, target_size=(224, 224))
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    x = preprocess_input(x)

    preds = model.predict(x)
    
    results = decode_predictions(preds, top=3)[0]

    for i, (imagenet_id, label, probability) in enumerate(results):
        print(f"{i+1}. Tahmin: {label.upper()} - Olasılık: %{probability * 100:.2f}")

if __name__ == "__main__":
    test_fotografi = "yapay_zeka_sonucu_106.jpg" 
    # test_fotografi = "data/train/low/106.png"
    classify_image(test_fotografi)