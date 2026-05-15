# BLM4504 Sayısal Görüntü İşleme

# Düşük Işık Koşullarında Görüntü Netleştirme ve Sınıflandırma

Bu proje, düşük ışık koşullarında çekilen görüntülerin **Zero-DCE (Zero-Reference Deep Curve Estimation)** derin öğrenme mimarisi ile gerçek zamanlı olarak iyileştirilmesi ve nesne tespiti (ResNet-50) algoritmalarının başarımının artırılması amacıyla geliştirilmiştir.

## Proje Özellikleri
- **Zero-DCE Entegrasyonu:** Referans bir aydınlık görüntüye ihtiyaç duymadan (zero-reference) kendi kendini eğiten, ışık eğrisi tahmini (curve estimation) yapan CNN mimarisi.
- **Etkileşimli Web Arayüzü (Streamlit):** Kullanıcıların yükledikleri fotoğrafları anında aydınlatabildikleri, Gama düzeltme içeren UX odaklı arayüz.
- **Akademik Karşılaştırma:** Geliştirilen derin öğrenme modelinin, klasik makine öğrenmesi algoritmalarıyla (Ridge Regression + Sobel/Gaussian Özellik Çıkarımı) kıyaslanması.
- **Kalite Metrikleri:** Başarımın PSNR ve SSIM metrikleriyle analizi.

## Kullanılan Teknolojiler
- **Derin Öğrenme:** TensorFlow, Keras, ResNet-50
- **Görüntü İşleme:** OpenCV, Scikit-Image
- **Arayüz:** Streamlit
- **Makine Öğrenmesi:** Scikit-Learn

## Kurulum ve Çalıştırma

1. Projeyi klonlayın:
   ```bash
   git clone [https://github.com/Impasbaa/BLM4504.git](https://github.com/Impasbaa/BLM4504.git)
