import cv2
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

def extract_spatial_features(img_channel):
    blur = cv2.GaussianBlur(img_channel, (5, 5), 0) # Etrafındaki ışık ortalaması
    sobelx = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, ksize=3) # Yatay kenarlar
    sobely = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=3) # Dikey kenarlar
    
    features = np.stack([
        img_channel.ravel(),
        blur.ravel(),
        np.abs(sobelx).ravel(),
        np.abs(sobely).ravel()
    ], axis=1)
    return features

def main():
    print("Klasik Makine Öğrenmesi (Öznitelik + Ridge Regression) Eğitiliyor...")
    
    # Eğitim için karanlık ve aydınlık fotoğraf (LOL Dataset)
    dark_img = cv2.imread("data/train/low/10.png") 
    bright_img = cv2.imread("data/train/high/10.png")
    
    dark_hsv = cv2.cvtColor(dark_img, cv2.COLOR_BGR2HSV)
    bright_hsv = cv2.cvtColor(bright_img, cv2.COLOR_BGR2HSV)
    
    dark_v = dark_hsv[:, :, 2] / 255.0
    bright_v = bright_hsv[:, :, 2] / 255.0

    X_train = extract_spatial_features(dark_v)
    y_train = bright_v.ravel()

    # Model: Ridge Regresyon
    ml_model = Ridge(alpha=1.0)
    ml_model.fit(X_train, y_train)
    print("Eğitim Tamamlandı!")

    test_img = cv2.imread("data/test_images/12.jpeg")
    test_hsv = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)
    test_v = test_hsv[:, :, 2] / 255.0
    
    print("Test görüntüsü aydınlatılıyor...")
    X_test = extract_spatial_features(test_v)
    y_pred = ml_model.predict(X_test)
    
    enhanced_v = y_pred.reshape(test_v.shape)
    enhanced_v = np.clip(enhanced_v, 0.0, 1.0) * 255.0
    
    test_hsv[:, :, 2] = enhanced_v.astype(np.uint8)
    enhanced_img = cv2.cvtColor(test_hsv, cv2.COLOR_HSV2RGB)
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite("klasik_ml_sonucu.jpg", cv2.cvtColor(enhanced_img.astype(np.uint8), cv2.COLOR_RGB2BGR))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Orijinal Karanlık Test")
    plt.imshow(test_img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Klasik ML Sonucu (Gaussian/Sobel + Ridge)")
    plt.imshow(enhanced_img)
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    main()