import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import cv2
import matplotlib.pyplot as plt

# DCE-Net Mimarisini Tanımlama
def build_dce_net():
    input_img = layers.Input(shape=(None, None, 3))
    conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_img)
    conv2 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    conv4 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)

    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
    conv5 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(int_con1)

    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(int_con2)

    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    x_r = layers.Conv2D(24, (3, 3), strides=(1, 1), padding='same', activation='tanh')(int_con3)

    return Model(inputs=input_img, outputs=x_r, name="DCE-Net")

class ZeroDCE_Inference(tf.keras.Model):
    def __init__(self, model, **kwargs):
        super(ZeroDCE_Inference, self).__init__(**kwargs)
        self.dce_model = model

    def get_enhanced_image(self, data, output):
        x = data
        for i in range(0, 24, 3):
            alpha = output[:, :, :, i:i+3]
            x = x + alpha * x * (1.0 - x)
        return x
        
    def call(self, inputs):
        alpha_maps = self.dce_model(inputs)
        return self.get_enhanced_image(inputs, alpha_maps)

def main():
    print("Model mimarisi kuruluyor...")
    
    dce_model = build_dce_net()

    inference_model = ZeroDCE_Inference(dce_model)
    
    inference_model.build(tf.TensorShape([None, None, None, 3]))
    
    print("Ağırlıklar yükleniyor...")

    inference_model.load_weights("weights/zero_dce_best_weights.weights.h5")
    print("Ağırlıklar başarıyla yüklendi.")

    # Test Görüntüsünü Oku 
    image_path = "data/test_images/1.jpeg" 
    
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Hata: {image_path} bulunamadı!")
        return
        
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    img_tensor = tf.convert_to_tensor(original_img, dtype=tf.float32) / 255.0
    img_tensor = tf.expand_dims(img_tensor, axis=0) 

    enhanced_tensor = inference_model(img_tensor)
    
    enhanced_img = np.squeeze(enhanced_tensor.numpy())
    enhanced_img = np.clip(enhanced_img, 0.0, 1.0)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(original_img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Zero-DCE ile Netleştirilmiş Görüntü")
    plt.imshow(enhanced_img)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()