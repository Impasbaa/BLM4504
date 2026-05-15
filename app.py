import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from PIL import Image

def build_dce_net():
    input_img = layers.Input(shape=(None, None, 3))
    conv1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_img)
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
    conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(conv2)
    conv4 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(conv3)
    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
    conv5 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(int_con1)
    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(int_con2)
    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    x_r = layers.Conv2D(24, (3, 3), padding='same', activation='tanh')(int_con3)
    return Model(inputs=input_img, outputs=x_r)

class ZeroDCE_Inference(tf.keras.Model):
    def __init__(self, model):
        super(ZeroDCE_Inference, self).__init__()
        self.dce_model = model
    def call(self, inputs):
        alpha_maps = self.dce_model(inputs)
        x = inputs
        for i in range(0, 24, 3):
            alpha = alpha_maps[:, :, :, i:i+3]
            x = x + alpha * x * (1.0 - x)
        return x

# Modeli Cache Mekanizması ile Yükleme
@st.cache_resource
def load_trained_model():
    dce_model = build_dce_net()
    dce_model.build(tf.TensorShape([None, None, None, 3]))
    
    dce_model.load_weights("weights/zero_dce_200_epochs.weights.h5")
    
    inference_model = ZeroDCE_Inference(dce_model)
    inference_model.build(tf.TensorShape([None, None, None, 3]))
    return inference_model

# Streamlit Arayüzü
st.set_page_config(page_title="Görüntü İyileştirici", layout="wide")
st.title("Görüntü İyileştirme (Zero-DCE)")

with st.spinner("Yapay zeka hazırlanıyor..."):
    model = load_trained_model()

uploaded_file = st.file_uploader("Karanlık bir fotoğraf seçin...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    # İşlemi hızlandırmak için resmi 512 piksele küçültüyoruz
    img.thumbnail((512, 512)) 
    
    st.subheader("İşlem Başlatılıyor...")
    
    # user_gamma = st.slider(
    #     "Aydınlatma Gücü Ayarı", 
    #     min_value=0.4, 
    #     max_value=1.5, 
    #     value=0.8, 
    #     step=0.1
    # )

    user_brightness = st.slider(
        "Parlaklık Seviyesi", 
        min_value=0.5,
        max_value=2.0,
        value=1.2,
        step=0.1
    )

    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Orijinal Fotoğraf", use_container_width=True)
    
    if st.button("Aydınlat"):
        with st.spinner("Yapay zeka pikselleri analiz ediyor..."):
            img_tensor = tf.convert_to_tensor(np.array(img), dtype=tf.float32) / 255.0
            img_tensor = tf.expand_dims(img_tensor, axis=0)
            
            # Model Tahmini
            enhanced_tensor = model(img_tensor)
            enhanced_img = np.squeeze(enhanced_tensor.numpy())
            enhanced_img = np.clip(enhanced_img, 0.0, 1.0)
            
            black_point = 0.05
            enhanced_img = np.clip((enhanced_img - black_point) / (1.0 - black_point), 0.0, 1.0)
            real_gamma = 1.0 / user_brightness
            enhanced_img = np.power(enhanced_img, real_gamma)
            
            with col2:
                st.image(enhanced_img, caption="AI Sonucu", use_container_width=True)
                st.success("İşlem tamamlandı!")