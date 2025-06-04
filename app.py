import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ========== CONFIG ========== #
st.set_page_config(
    page_title="DENTAXO - Dental Classification App",
    page_icon="🦷",
    layout="centered"
)

# Load Model
model = load_model('ICDAS-80-part5.h5')

# Kelas
class_names = ['Advanced', 'Early', 'Healthy']

# ========== CUSTOM STYLING ========== #
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #f0f4ff, #ffffff);
        }
        .main {
            background-color: transparent;
        }
        h1, h2, h3 {
            color: #3C3C3C;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton > button {
            background-color: #5D3FD3;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: bold;
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            font-size: 0.85em;
            color: #888;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ========== SIDEBAR MENU ========== #
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2909/2909779.png", width=100)
st.sidebar.markdown("## **DENTAXO**")
menu = st.sidebar.radio("📁 Navigasi", [
    "🏠 Dashboard",
    "📘 Petunjuk Penggunaan",
    "ℹ️ Tentang DENTAXO",
    "📤 Upload & Deteksi"
])

# ========== HALAMAN: DASHBOARD ========== #
if menu == "🏠 Dashboard":
    st.title("🦷 DENTAXO - Dental Classification AI")
    st.markdown("""
    Selamat datang di **DENTAXO**, aplikasi klasifikasi kondisi gigi berbasis deep learning yang mampu mengenali tiga kategori:
    
    - 🟥 **Advanced** (Kerusakan Parah)
    - 🟨 **Early** (Tahap Awal)
    - 🟩 **Healthy** (Sehat)

    Dibuat untuk membantu masyarakat dan profesional dalam deteksi dini kesehatan gigi dari gambar secara cepat, praktis, dan akurat.
    """)

# ========== HALAMAN: PETUNJUK ========== #
elif menu == "📘 Petunjuk Penggunaan":
    st.header("📘 Petunjuk Penggunaan")
    st.markdown("""
    1. Siapkan gambar gigi yang jelas, fokus, dan pencahayaan cukup.
    2. Masuk ke menu **Upload & Deteksi**.
    3. Klik **Upload** dan pilih gambar dari perangkat Anda.
    4. Lihat hasil prediksi dan informasi klasifikasi.
    
    ⚠️ Disarankan untuk menggunakan gambar dari kamera gigi/dokter untuk hasil lebih akurat.
    """)

# ========== HALAMAN: TENTANG ========== #
elif menu == "ℹ️ Tentang DENTAXO":
    st.header("ℹ️ Tentang Aplikasi DENTAXO")
    st.markdown("""
    **DENTAXO** adalah sistem klasifikasi citra gigi berbasis **Deep Learning (CNN)** yang dilatih untuk mengenali pola visual kondisi gigi.
    
    **Kategori yang dideteksi:**
    - 🟥 **Advanced**: Kerusakan berat seperti lubang besar, infeksi, gigi hancur.
    - 🟨 **Early**: Plak, karies ringan, atau perubahan warna.
    - 🟩 **Healthy**: Tidak ada indikasi kerusakan signifikan.

    Dibuat sebagai solusi inovatif dalam bidang kesehatan gigi.
    """)

# ========== HALAMAN: UPLOAD & DETEKSI ========== #
elif menu == "📤 Upload & Deteksi":
    st.header("📤 Upload Gambar Gigi")
    uploaded_file = st.file_uploader("Silakan upload gambar gigi (JPG/PNG):", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='🖼️ Gambar yang Diupload', use_container_width=True)

        # --- Preprocessing ---
        try:
            image = image.convert("RGB")
            image = image.resize((128, 128))
            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype('float32') / 255.0

            # --- Predict ---
            pred = model.predict(img_array)[0]
            predicted_index = np.argmax(pred)
            predicted_label = class_names[predicted_index]

            # --- Display Result ---
            st.subheader("🧠 Hasil Prediksi")
            if predicted_label == "Advanced":
                st.markdown(f"<div class='prediction-box' style='background-color:#FFE6E6;'><h3>🟥 Kerusakan Parah</h3><p>Segera konsultasi ke dokter gigi. Gigi menunjukkan kerusakan berat.</p></div>", unsafe_allow_html=True)
            elif predicted_label == "Early":
                st.markdown(f"<div class='prediction-box' style='background-color:#FFFBE6;'><h3>🟨 Tahap Awal</h3><p>Gigi menunjukkan gejala awal kerusakan. Lakukan perawatan lebih lanjut.</p></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='prediction-box' style='background-color:#E6FFEA;'><h3>🟩 Sehat</h3><p>Gigi Anda dalam kondisi baik. Tetap jaga kebersihan mulut!</p></div>", unsafe_allow_html=True)

            # Probabilities
            st.markdown("#### 📊 Probabilitas Kelas:")
            for i, prob in enumerate(pred):
                st.write(f"- **{class_names[i]}**: {prob:.2%}")

        except Exception as e:
            st.error("⚠️ Terjadi kesalahan saat memproses gambar.")
            st.exception(e)

# ========== FOOTER ========== #
st.markdown("""
    <div class="footer">
        Aplikasi ini dikembangkan oleh <b>Ghany Fitriamara Suci</b><br>
        Program Studi Fisika, Universitas Islam Negeri (UIN) Jakarta<br>
        Tahun 2025
    </div>
""", unsafe_allow_html=True)
