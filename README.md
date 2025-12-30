# 🐦 BirdDetect AI - Yapay Zeka Destekli Kuş Türü Tanıma

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%26%20HuggingFace-orange)
![Interface](https://img.shields.io/badge/Interface-Streamlit-red)

**BirdDetect AI**, son teknoloji **Vision Transformer (ViT)** mimarisini kullanarak doğadaki kuş türlerini fotoğraflarından yüksek doğrulukla tespit eden derin öğrenme tabanlı bir görüntü sınıflandırma projesidir.

---

## 🎯 Proje Hakkında

Bu proje, karmaşık görsel verileri işleyebilen modern bir yapay zeka modeli eğitmek ve bu modeli herkesin kullanabileceği pratik bir web arayüzüne dönüştürmek amacıyla geliştirilmiştir. Geleneksel Evrişimli Sinir Ağları (CNN) yerine, görüntüleri birer kelime dizisi gibi işleyen ve global bağlamı yakalayan **Google Vision Transformer (ViT)** mimarisi üzerine inşa edilmiştir.

### ✨ Temel Özellikler
* **Geniş Tür Yelpazesi:** 220 farklı kuş türü üzerinde özelleştirilmiş eğitim süreci gerçekleştirilmiştir.
* **Transformer Gücü:** `google/vit-base-patch16-224` modeli üzerinde Fine-Tuning (İnce Ayar) yapılmıştır.
* **Hızlı ve İnteraktif Arayüz:** Streamlit kütüphanesi ile güçlendirilmiş, anlık tahmin yapan kullanıcı dostu panel.
* **Derinlemesine Analiz:** Sadece tek bir tahmin değil, **Top-3 Olasılık Dağılımı** ve **Güven Skoru** sunumu.
* **Akıllı Ön İşleme:** Görüntüleri modelin eğitim formatına uygun hale getiren otomatik `ViTImageProcessor` entegrasyonu.

---

## 📂 Veri Seti Detayları

Proje kapsamında [Kaggle Bird Species Classification (220 Categories)](https://www.kaggle.com/datasets/kedarsai/bird-species-classification-220-categories) veri seti kullanılmıştır.

* **Sınıf Sayısı:** 220 Farklı Kuş Türü.
* **Eğitim/Test Ayrımı:** Veri seti %80 Eğitim ve %20 Test (Validation) olacak şekilde rastgele bölünmüştür.
* **Ön İşleme:** Görüntüler model gereksinimlerine göre normalize edilmiş ve 224x224 boyutuna getirilmiştir.

---

## 🛠️ Kullanılan Teknolojiler

| Alan | Teknoloji / Kütüphane | Açıklama |
| :--- | :--- | :--- |
| **Dil** | Python 3.10 | Ana programlama dili |
| **Model** | Hugging Face Transformers | ViT model mimarisi ve ön-eğitimli ağırlıklar |
| **Framework** | PyTorch (CUDA) | GPU tabanlı model eğitimi ve çıkarım işlemleri |
| **Arayüz** | Streamlit | Web tabanlı interaktif kullanıcı arayüzü |
| **Veri Analizi** | Scikit-learn & Matplotlib | Başarı metrikleri ve eğitim grafiklerinin oluşturulması |

---

## 📊 Model Performansı ve Eğitim Süreci

[EĞİTİLMİŞ MODEL BAĞLANTISI](https://drive.google.com/file/d/1EYG2B_fZh8yPFqwTWlu7HXdwGAO1tt13/view?usp=sharing)

Model eğitimi, CUDA destekli bir GPU üzerinde 10 epoch boyunca sürdürülmüştür. Eğitim sırasında `learning_rate=2e-5` ve `weight_decay=0.01` optimizasyon parametreleri uygulanmıştır.

### Sonuçlar:
* **Doğruluk (Validation Accuracy):** Eğitim sonucunda yaklaşık **%85** doğruluk oranına ulaşılmıştır.
* **Kayıp (Loss):** Eğitim kaybı (Training Loss) istikrarlı bir şekilde azalırken, doğruluk grafiği modelin başarıyla genelleme yaptığını göstermektedir.

![Eğitim Grafikleri](egitim_sonuclari.png)

---

## 🚀 Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

[Uygulama Demo Videosu ](https://drive.google.com/file/d/1L-KYh08mN-cVR2RWgAfrJaOK41goTuE-/view?usp=sharing)

### 📦 Depoyu Klonlayın
```bash
git clone https://github.com/yigitkalkan/BirdDetect.git
cd BirdDetect-AI 
```

## 📖 Gerekli Kütüphaneleri Yükleyin
```bash
pip install -r requirements.txt
```

## 🚀 Uygulamayı Başlatın
```bash
streamlit run mainbird.py
```
Terminalde aşağıdaki gibi bir çıktı alırsınız:
```bash
Local URL: http://localhost:8501
```
## 💻 Kullanım Rehberi

1. **🖼️ Görüntü Yükleme**  
   Web arayüzünden bir kuş fotoğrafı yükleyin.

2. **⚙️ Otomatik İşleme**  
   Yüklenen görüntü sistem tarafından otomatik olarak işlenir.

3. **📊 Model Çıktıları**  
   Model aşağıdaki bilgileri kullanıcıya sunar:
   - Tahmin edilen kuş türü
   - Güven oranı (%)
   - En olası ilk 3 tahmin











