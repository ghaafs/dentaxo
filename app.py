import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import time

# ========== CONFIG ========== #
st.set_page_config(
    page_title="DENTAXO - Dental Classification App",
    page_icon="🦷",
    layout="centered"
)

# ========== LOAD MODEL ========== #
@st.cache_resource
def load_dl_model():
    return load_model('ICDAS-80-part5.h5')

model = load_dl_model()
class_names = ['Advanced', 'Early', 'Healthy']

# ========== CUSTOM STYLE ========== #
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Prompt', sans-serif;
            background: linear-gradient(135deg, #e0f7fa, #ffffff);
        }

        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: #5D3FD3;
        }

        .stButton > button {
            background-color: #5D3FD3;
            color: white;
            padding: 10px 24px;
            font-weight: 600;
            border-radius: 8px;
        }

        .stRadio > div {
            gap: 10px;
        }

        .prediction-card {
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(93, 63, 211, 0.2);
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ========== SIDEBAR MENU ========== #
menu = st.sidebar.radio("📁 Navigasi Menu", [
    "🏠 Dashboard",
    "📘 Petunjuk Penggunaan",
    "🧠 Deteksi Gigi",
    "ℹ️ Tentang DENTAXO"
])

# ========== HALAMAN: DASHBOARD ========== #
if menu == "🏠 Dashboard":
    st.markdown("## 🦷 Selamat Datang di DENTAXO")
    st.markdown("""
    **DENTAXO** adalah sistem klasifikasi citra gigi berbasis **Deep Learning** yang membantu deteksi dini masalah gigi.
    
    ### Kategori Klasifikasi:
    - 🔴 **Advanced** – Kerusakan parah, perlu penanganan medis segera
    - 🟡 **Early** – Tanda-tanda awal kerusakan
    - 🟢 **Healthy** – Gigi sehat
    
    > Teknologi kami mendukung langkah awal pencegahan dan konsultasi digital untuk kesehatan gigi masyarakat.
    """)

# ========== HALAMAN: PETUNJUK ========== #
elif menu == "📘 Petunjuk Penggunaan":
    st.markdown("## 📘 Panduan Penggunaan")
    st.markdown("""
    1. Siapkan gambar gigi dengan pencahayaan baik dan fokus.
    2. Buka menu **🧠 Deteksi Gigi**.
    3. Upload gambar dan tunggu hasil analisis.
    4. Baca interpretasi hasil dan saran tindakan.

    ⚠️ Hindari gambar yang buram atau terlalu gelap.
    """)

# ========== HALAMAN: DETEKSI GIGI ========== #
elif menu == "🧠 Deteksi Gigi":
    st.markdown("## 📤 Upload Gambar Gigi")
    uploaded_file = st.file_uploader("Unggah gambar gigi Anda (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Gambar yang diupload", use_container_width=True)

        # Preprocessing
        try:
            image = image.convert("RGB").resize((128, 128))
            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0).astype('float32') / 255.0

            with st.spinner('🔍 Menganalisis kondisi gigi...'):
                time.sleep(2)  # efek loading
                pred = model.predict(img_array)[0]

            predicted_index = np.argmax(pred)
            predicted_label = class_names[predicted_index]
            confidence = pred[predicted_index] * 100

            # Hasil prediksi
            st.markdown("### 🧪 Hasil Deteksi")
            if predicted_label == 'Advanced':
                st.markdown(f"""
                <div class="prediction-card" style="background-color:#ffebee;">
                    <h3>🔴 Kerusakan Parah</h3>
                    <p>Probabilitas: <strong>{confidence:.2f}%</strong></p>
                    <p>Gigi menunjukkan kerusakan berat. Disarankan segera konsultasi ke dokter gigi.</p>
                </div>
                """, unsafe_allow_html=True)

            elif predicted_label == 'Early':
                st.markdown(f"""
                <div class="prediction-card" style="background-color:#fff8e1;">
                    <h3>🟡 Tahap Awal</h3>
                    <p>Probabilitas: <strong>{confidence:.2f}%</strong></p>
                    <p>Ada tanda-tanda awal kerusakan. Perlu pemeriksaan lebih lanjut.</p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="prediction-card" style="background-color:#e8f5e9;">
                    <h3>🟢 Gigi Sehat</h3>
                    <p>Probabilitas: <strong>{confidence:.2f}%</strong></p>
                    <p>Gigi Anda dalam kondisi baik. Pertahankan kebiasaan kebersihan gigi!</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📊 Rincian Probabilitas"):
                for i, prob in enumerate(pred):
                    st.progress(float(prob), f"{class_names[i]}: {prob:.2%}")

        except Exception as e:
            st.error("⚠️ Terjadi kesalahan saat memproses gambar.")
            st.exception(e)

# ========== HALAMAN: TENTANG DENTAXO ========== #
elif menu == "ℹ️ Tentang DENTAXO":
    st.markdown("## ℹ️ Tentang Aplikasi")
    st.markdown("""
    **DENTAXO** adalah platform AI interaktif untuk deteksi kondisi gigi berbasis gambar.

    ### Teknologi:
    - Model CNN dengan akurasi tinggi
    - Dataset gigi dari berbagai tingkat kerusakan
    - Inferensi cepat dan berbasis cloud

    Aplikasi ini bertujuan untuk mempercepat diagnosis awal serta memperluas akses masyarakat terhadap informasi kesehatan gigi yang akurat.
    """)
