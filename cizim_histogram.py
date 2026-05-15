import cv2
import matplotlib.pyplot as plt

def histogramlari_karsilastir(karanlik_yol, ai_sonucu_yol):
    dark_img = cv2.imread(karanlik_yol, cv2.IMREAD_GRAYSCALE)
    enhanced_img = cv2.imread(ai_sonucu_yol, cv2.IMREAD_GRAYSCALE)

    if dark_img is None or enhanced_img is None:
        print("Görüntüler bulunamadı!")
        return

    plt.figure(figsize=(10, 5))
   
    plt.hist(dark_img.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.6, label='Karanlık (Orijinal)')
    
    plt.hist(enhanced_img.ravel(), bins=256, range=[0, 256], color='orange', alpha=0.6, label='Zero-DCE (Aydınlatılmış)')
    
    plt.title("Piksel Parlaklık Dağılımı")
    plt.xlabel("Piksel Değeri (0: Siyah, 255: Beyaz)")
    plt.ylabel("Piksel Sayısı")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig("histogram_karsilastirma.png", dpi=300)
    plt.show()

histogramlari_karsilastir("data/train/low/10.png", "yapay_zeka_sonucu_106.jpg")