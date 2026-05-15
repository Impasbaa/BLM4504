import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def metrikleri_hesapla(gercek_aydinlik_yol, ai_sonucu_yol):
    img_true = cv2.imread(gercek_aydinlik_yol)
    img_ai = cv2.imread(ai_sonucu_yol)

    if img_true is None or img_ai is None:
        print("Görüntüler bulunamadı!")
        return

    if img_true.shape != img_ai.shape:
        img_ai = cv2.resize(img_ai, (img_true.shape[1], img_true.shape[0]))

    # Metriklerin Hesaplanması
    # PSNR Hesaplama
    psnr_degeri = psnr(img_true, img_ai)
    
    # SSIM Hesaplama
    ssim_degeri = ssim(img_true, img_ai, channel_axis=2)

    print(f"PSNR (Tepe Sinyal Gürültü Oranı) : {psnr_degeri:.2f} dB")
    print(f"SSIM (Yapısal Benzerlik İndeksi) : {ssim_degeri:.4f}")

metrikleri_hesapla("data/test_images/12.jpeg", "yapay_zeka_sonucu.jpg")
# metrikleri_hesapla("data/test_images/12.jpeg", "klasik_ml_sonucu.jpg")