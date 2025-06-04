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
class_names = ['Advanced', 'Early', 'Healthy']
session_counter = st.session_state.setdefault("predictions", 0)

# ========== STYLE ========== #
st.markdown("""
    <style>
        body, .main, .block-container { background-color: #fdfdff; }
        h1, h2, h3, .stMarkdown { color: #5D3FD3; }
        .stButton>button {
            background-color: #5D3FD3;
            color: white;
            border-radius: 8px;
            font-weight: bold;
        }
        .footer {
            margin-top: 4em;
            padding: 1em;
            text-align: center;
            font-size: 0.9em;
            color: #666;
        }
    </style>
""", unsafe_allow_html=True)

# ========== SIDEBAR MENU ========== #
menu = st.sidebar.radio("🔍 Menu", [
    "🏠 Dashboard",
    "📘 Petunjuk Penggunaan",
    "ℹ️ Tentang DENTAXO",
    "📤 Upload & Deteksi"
])

# ========== HALAMAN: DASHBOARD ========== #
if menu == "🏠 Dashboard":
    st.title("🦷 DENTAXO")
    st.markdown("""
    **DENTAXO** adalah aplikasi klasifikasi kondisi gigi berbasis AI (Deep Learning CNN) untuk mengenali tiga kategori:
    
    - 🟥 **Advanced**: Kerusakan gigi parah
    - 🟨 **Early**: Kerusakan tahap awal
    - 🟩 **Healthy**: Gigi sehat
    
    Didesain untuk membantu masyarakat dan profesional kesehatan dalam mendeteksi masalah gigi melalui citra digital.
    """)
    st.info(f"📈 Jumlah prediksi selama sesi ini: **{session_counter} kali**")

# ========== HALAMAN: PETUNJUK ========== #
elif menu == "📘 Petunjuk Penggunaan":
    st.header("📘 Cara Menggunakan DENTAXO")
    st.markdown("""
    1. Siapkan gambar gigi yang **jelas, terang, dan fokus**, khusus untuk **1-3 gigi saja per gambar**.
    2. Masuk ke menu **Upload & Deteksi**.
    3. Klik **Upload**, lalu pilih file gambar dalam format JPG/PNG.
    4. Tunggu hasil prediksi muncul di layar.

    ⚠️ Gambar yang terlalu buram, gelap, atau menampilkan seluruh gigi sekaligus bisa menyebabkan hasil prediksi tidak akurat.

    #### 📷 Contoh Gambar Ideal:
    - Close-up 1 gigi berlubang
    - 2 gigi bagian depan dengan karies
    - 3 gigi kiri atas yang tampak jelas
    """)

# ========== HALAMAN: TENTANG ========== #
elif menu == "ℹ️ Tentang DENTAXO":
    st.header("ℹ️ Tentang DENTAXO")
    st.markdown("""
    DENTAXO adalah sistem klasifikasi citra berbasis AI untuk deteksi dini kondisi gigi.

    **Kategori Klasifikasi:**
    - **Advanced**: Lubang besar, infeksi, gigi hancur
    - **Early**: Plak, karies ringan, perubahan warna
    - **Healthy**: Gigi utuh tanpa kerusakan

    Aplikasi ini masih dalam tahap pengembangan dan tidak menggantikan diagnosis medis profesional.
    """)

# ========== HALAMAN: UPLOAD & DETEKSI ========== #
elif menu == "📤 Upload & Deteksi":
    st.header("📤 Upload Gambar Gigi")
    uploaded_file = st.file_uploader("Silakan upload gambar gigi (JPG/PNG):", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Gambar yang Diupload', use_container_width=True)

        # Preprocess
        image = image.resize((128, 128))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0).astype('float32') / 255.0

        # Prediction
        pred = model.predict(img_array)[0]
        predicted_index = np.argmax(pred)
        predicted_label = class_names[predicted_index]
        session_counter += 1
        st.session_state.predictions = session_counter

        # Result
        st.success(f"### 🧠 Prediksi (EN): {predicted_label}")
        if predicted_label == "Advanced":
            st.error("**🦷 Prediksi (ID): Kerusakan Parah**\n\nSegera konsultasikan ke dokter gigi.")
        elif predicted_label == "Early":
            st.warning("**🦷 Prediksi (ID): Tahap Awal**\n\nLakukan pemeriksaan dan perawatan ringan.")
        else:
            st.success("**🦷 Prediksi (ID): Sehat**\n\nGigi dalam kondisi baik. Tetap jaga kebersihan!")

        st.markdown("#### 🔢 Probabilitas Kelas:")
        for i, prob in enumerate(pred):
            st.write(f"- {class_names[i]}: **{prob:.2%}**")

# ========== FOOTER ========== #
st.markdown("""
    <div class="footer">
        Aplikasi ini dikembangkan oleh <b>Ghany Fitriamara Suci</b><br>
        Program Studi Fisika, Universitas Islam Negeri (UIN) Jakarta<br>
        Tahun 2025
    </div>
""", unsafe_allow_html=True)
