import streamlit as st
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import torch.nn.functional as F

# --- AYARLAR ---
# Eğitilmiş modelin bulunduğu klasör yolu
MODEL_PATH = r"C:\Users\THERMALTAKE\Desktop\bird\bird_model"

# Sayfa başlığı ve ikonu
st.set_page_config(page_title="Kuş Türü Tanıma", page_icon="🐦")

# --- MODEL YÜKLEME (Önbellek kullanarak hızlandırıyoruz) ---
@st.cache_resource
def load_model():
    try:
        # Cihaz seçimi (GPU varsa kullan, yoksa CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Modeli ve işlemciyi (processor) yükle
        model = ViTForImageClassification.from_pretrained(MODEL_PATH)
        processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
        
        model.to(device)
        model.eval() # Değerlendirme modu
        return model, processor, device
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None, None, None

# Modeli yükle
model, processor, device = load_model()

# --- ARAYÜZ TASARIMI ---
st.title("🐦 Kuş Türü Tahmin Uygulaması")
st.write("Eğitilmiş ViT modelini kullanarak kuş türlerini sınıflandırın.")

# Dosya yükleme alanı
uploaded_file = st.file_uploader("Bir kuş fotoğrafı yükleyin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Resmi göster
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption='Yüklenen Fotoğraf', width=350)

    # --- TAHMİN İŞLEMİ ---
    with st.spinner('Kuş türü analiz ediliyor...'):
        # Resmi modelin anlayacağı formata getir
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Tahmin yap
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Olasılıkları hesapla (Softmax)
        probs = F.softmax(logits, dim=1)
        
        # En yüksek olasılıklı sınıfı bul
        top_prob, top_class_idx = probs.topk(1, dim=1)
        
        # Etiketi al (id2label config dosyasından gelir)
        predicted_label = model.config.id2label[top_class_idx.item()]
        confidence = top_prob.item() * 100

    # --- SONUÇLARI GÖSTER ---
    with col2:
        st.success(f"Tahmin: **{predicted_label}**")
        st.metric(label="Doğruluk Oranı (Güven)", value=f"%{confidence:.2f}")
        
        st.markdown("---")
        st.write("🔎 **En Olası 3 Tahmin:**")
        
        # En yüksek 3 tahmini göster
        top3_probs, top3_indices = probs.topk(3, dim=1)
        
        for i in range(3):
            label = model.config.id2label[top3_indices[0][i].item()]
            prob = top3_probs[0][i].item() * 100
            st.write(f"**{i+1}. {label}**")
            st.progress(int(prob)) # İlerleme çubuğu