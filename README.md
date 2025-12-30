# 🐦 Bird Species Classification with Vision Transformer

Bu proje, Kaggle’daki **220 sınıflı kuş türü veri seti** kullanılarak **Vision Transformer (ViT)** ile görüntü sınıflandırma problemi çözmek için hazırlanmış bir ödev/projedir.Model eğitimi **Hugging Face Transformers + PyTorch** ile yapılmış, ayrıca **Streamlit** ile basit bir tahmin arayüzü geliştirilmiştir.

- Veri seti: https://www.kaggle.com/datasets/kedarsai/bird-species-classification-220-categories?resource=download
- Kullanılan model: `google/vit-base-patch16-224`
- Eğitim çıktıları: `egitim_sonuclari.png`

---

## 📁 Klasör Yapısı

Aşağıdaki yapı, proje klasörünüzle uyumludur:

bird/
├─ archive/ # Kaggle veri seti (imagefolder formatında)
├─ bird_model/ # Eğitilmiş model + processor çıktıları (Trainer save_model)
├─ mybird/ # (Opsiyonel) Python venv klasörü
├─ birdd.py # Model eğitimi (ViT + Trainer)
├─ mainbird.py # Streamlit tahmin uygulaması
├─ egitim_sonuclari.png # Eğitim süreci grafikleri (loss/accuracy)
├─ requirements.txt # Bağımlılıklar
└─ README.md # Bu dosya

yaml
Kodu kopyala

> Not: `archive/` klasörü içeriği, `datasets` kütüphanesinin `imagefolder` yapısına uygun olmalıdır (sınıf klasörleri altında görseller).

---

## ✅ Kullanılan Teknolojiler

- Python
- PyTorch (CUDA destekli)
- Hugging Face Transformers (`ViTForImageClassification`, `ViTImageProcessor`)
- Hugging Face Datasets (`load_dataset("imagefolder")`)
- Scikit-learn (accuracy)
- Matplotlib (eğitim grafikleri)
- Streamlit (arayüz)

---

## ⚙️ Kurulum

### 1) Sanal ortam (önerilir)

```powershell
python -m venv mybird
.\mybird\Scripts\activate
PowerShell “running scripts is disabled” hatası alırsanız:

powershell
Kodu kopyala
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Sonra tekrar:

powershell
Kodu kopyala
.\mybird\Scripts\activate

2) Bağımlılıkları yükleme
bash
Kodu kopyala
pip install -r requirements.txt
requirements.txt içinde CUDA 11.8 için PyTorch index adresi tanımlıdır. (GPU kullanacaksanız uygundur.)

📦 Veri Seti Hazırlığı
Kaggle linkinden veri setini indir:

https://www.kaggle.com/datasets/kedarsai/bird-species-classification-220-categories?resource=download

Dosyaları proje içindeki archive/ klasörüne çıkartın.


📈 Eğitim Sonuçları
Eğitim sırasında loss düşüşü ve validation accuracy değişimi egitim_sonuclari.png dosyasına kaydedilir.

Validation accuracy grafiğinde doğruluk hızlı yükselip ~0.85 civarında dengelenmektedir.

Training loss düşerken validation loss daha yavaş düşerek belli bir seviyede stabil kalmaktadır (normal bir genelleme davranışı).


## 📌 Kaynak

* Kaggle veri seti: [https://www.kaggle.com/datasets/kedarsai/bird-species-classification-220-categories?resource=download](https://www.kaggle.com/datasets/kedarsai/bird-species-classification-220-categories?resource=download)
* ViT: [https://huggingface.co/google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
