import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import time

# ========== CONFIG ========== #
st.set_page_config(
    page_title="DENTAXO - Deteksi Kondisi Gigi",
    page_icon="🦷",
    layout="centered"
)

# Load model
model = load_model('ICDAS-80-part5.h5')
class_names = ['Advanced', 'Early', 'Healthy']

# Session State
if "prediction_count" not in st.session_state:
    st.session_state["prediction_count"] = 0

# ========== STYLING ========== #
st.markdown("""
    <style>
        body { background-color: #ffffff; }
        .main { background-color: #ffffff; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: #5D3FD3; }
        .stButton>button {
            background-color: #5D3FD3;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .footer {
            margin-top: 4rem;
            text-align: center;
            font-size: 0.9rem;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

# ========== SIDEBAR ========== #
menu = st.sidebar.radio("📂 Menu Navigasi", [
    "🏠 Beranda",
    "📘 Panduan Penggunaan",
    "📤 Upload & Deteksi",
    "ℹ️ Tentang DENTAXO"
])
st.sidebar.markdown("---")
st.sidebar.markdown(f"🧮 Total Prediksi Sesi Ini: **{st.session_state['prediction_count']}**")

# ========== BERANDA ========== #
if menu == "🏠 Beranda":
    st.title("🦷 DENTAXO")
    st.markdown("""
    **DENTAXO** adalah aplikasi cerdas berbasis AI untuk mengklasifikasikan kondisi gigi manusia secara otomatis melalui gambar.

    **Kemampuan:**
    - Mendeteksi gigi **Sehat**, **Tahap Awal Kerusakan**, dan **Kerusakan Parah**
    - Memberikan hasil cepat dan mudah dipahami
    - Dibangun menggunakan teknologi Deep Learning terkini

    Cocok digunakan sebagai alat bantu pemeriksaan awal sebelum ke dokter gigi.
    """)

# ========== PANDUAN ========== #
elif menu == "📘 Panduan Penggunaan":
    st.header("📘 Panduan Penggunaan")
    st.markdown("""
    **🛠️ DENTAXO masih dalam tahap pengembangan.**  
    Untuk akurasi terbaik, unggah gambar close-up **1–3 gigi saja** dalam 1 gambar.

    ### ✅ Contoh Gambar yang Direkomendasikan:
    - Gambar close-up 1 gigi berlubang
    - Dua gigi depan dengan plak atau perubahan warna
    - Foto fokus ke bagian gigi yang mengalami masalah

    ---

    ### 🔍 Cara Penggunaan:
    1. Masuk ke menu **Upload & Deteksi**
    2. Klik tombol **Upload**, pilih gambar dari perangkat
    3. Tunggu proses analisis
    4. Lihat hasil prediksi & rekomendasi

    ⚠️ Hindari gambar gelap, blur, atau terlalu banyak gigi sekaligus.
    """)

# ========== UPLOAD & DETEKSI ========== #
elif menu == "📤 Upload & Deteksi":
    st.header("📤 Upload Gambar Gigi")
    uploaded_file = st.file_uploader("Unggah gambar (JPG/PNG, maksimal 1–3 gigi per gambar):", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Gambar yang Diunggah", use_column_width=True)

        try:
            with st.spinner("🔎 Menganalisis gambar..."):
                image = image.convert("RGB")
                image = image.resize((128, 128))
                img_array = img_to_array(image)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array.astype('float32') / 255.0
                time.sleep(1.5)  # simulasi loading

                pred = model.predict(img_array)[0]
                predicted_index = np.argmax(pred)
                predicted_label = class_names[predicted_index]

                st.session_state["prediction_count"] += 1

                st.success(f"### 🧠 Prediksi: {predicted_label}")
                label_id = predicted_label.lower()

                if label_id == "advanced":
                    st.error("**🔴 Kerusakan Parah**\n\nTerdapat indikasi kerusakan gigi yang serius. Segera konsultasikan ke dokter gigi.")
                elif label_id == "early":
                    st.warning("**🟡 Tahap Awal Kerusakan**\n\nTerdapat gejala awal kerusakan. Disarankan kontrol ke dokter gigi secepatnya.")
                else:
                    st.success("**🟢 Sehat**\n\nGigi tampak sehat. Pertahankan kebersihan mulut secara rutin!")

                st.markdown("---")
                st.subheader("📈 Rincian Probabilitas:")
                for i, prob in enumerate(pred):
                    st.write(f"- {class_names[i]}: **{prob:.2%}**")

        except Exception as e:
            st.error("⚠️ Terjadi kesalahan saat memproses gambar.")
            st.exception(e)

# ========== TENTANG ========== #
elif menu == "ℹ️ Tentang DENTAXO":
    st.header("ℹ️ Tentang Aplikasi DENTAXO")
    st.markdown("""
    **DENTAXO** (Dental Image Classifier) merupakan proyek pengembangan AI untuk membantu mendeteksi kondisi gigi melalui gambar secara otomatis.

    ### Model & Teknologi:
    - Model: CNN (Convolutional Neural Network)
    - Dataset: Gambar gigi berdasarkan standar ICDAS
    - Kategori:
        - **Advanced**: Kerusakan parah, lubang besar
        - **Early**: Plak, lubang awal, perubahan warna
        - **Healthy**: Tidak ada indikasi kerusakan

    DENTAXO bukan alat diagnosis medis, tetapi dapat membantu edukasi dan deteksi awal.

    ---
    """)

# ========== FOOTER ========== #
st.markdown("""
    <div class="footer">
        Aplikasi ini dikembangkan oleh <b>Ghany Fitriamara Suci</b><br>
        Program Studi Fisika, Universitas Islam Negeri (UIN) Jakarta<br>
        Tahun 2025
    </div>
""", unsafe_allow_html=True)
