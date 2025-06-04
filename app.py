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
model = load_model('ICDAS-80-part5.h5')  # ganti dengan path model kamu

# Kelas
class_names = ['Advanced', 'Early', 'Healthy']

# ========== STYLE ========== #
st.markdown("""
    <style>
        body {
            background-color: #ffffff;
        }
        .main {
            background-color: #ffffff;
        }
        .css-1d391kg {background-color: #ffffff;}
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #5D3FD3; /* ungu */
        }
        .stButton>button {
            background-color: #5D3FD3;
            color: white;
        }
        .st-bf {
            background-color: #ccf2d1; /* hijau muda */
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
    **DENTAXO** adalah aplikasi klasifikasi kondisi gigi berbasis deep learning yang dapat mengenali tiga kategori:
    - 🟥 Advanced (Kerusakan Parah)
    - 🟨 Early (Tahap Awal)
    - 🟩 Healthy (Sehat)

    Dirancang untuk membantu masyarakat dan profesional kesehatan dalam mengenali kondisi gigi dari gambar secara cepat dan akurat.
    """)

# ========== HALAMAN: PETUNJUK ========== #
elif menu == "📘 Petunjuk Penggunaan":
    st.header("📘 Petunjuk Penggunaan")
    st.markdown("""
    1. Siapkan gambar gigi yang jelas, fokus, dan dalam posisi yang baik.
    2. Masuk ke menu **Upload & Deteksi**.
    3. Klik **Upload** dan pilih gambar dari perangkat Anda.
    4. Tunggu hasil prediksi muncul di layar.

    ⚠️ Gambar yang buram atau terlalu gelap bisa menghasilkan prediksi yang tidak akurat.
    """)

# ========== HALAMAN: TENTANG DENTAXO ========== #
elif menu == "ℹ️ Tentang DENTAXO":
    st.header("ℹ️ Tentang DENTAXO")
    st.markdown("""
    DENTAXO adalah sistem klasifikasi citra berbasis AI untuk mendeteksi kondisi gigi manusia.
    
    **Kategori Klasifikasi:**
    - **Advanced**: Tanda kerusakan gigi tingkat lanjut seperti lubang besar, infeksi, atau gigi hancur.
    - **Early**: Indikasi awal kerusakan gigi seperti plak, karies ringan, atau perubahan warna.
    - **Healthy**: Struktur gigi masih sehat, tidak ditemukan tanda kerusakan yang signifikan.

    Dibangun dengan teknologi **Deep Learning CNN**, sistem ini dilatih dengan ribuan data gambar untuk mengenali pola visual dari berbagai kondisi gigi.
    """)

# ========== HALAMAN: UPLOAD & DETEKSI ========== #
elif menu == "📤 Upload & Deteksi":
    st.header("📤 Upload Gambar Gigi")
    uploaded_file = st.file_uploader("Silakan upload gambar gigi dalam format JPG/PNG:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Tampilkan gambar
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang Diupload', use_container_width=True)

        # Preprocessing
        image = image.convert("RGB")  
        image = image.resize((128, 128)) 
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = img_array.astype('float32') / 255.0  

        st.write("Model input shape:", model.input_shape)
        st.write("Image array shape:", img_array.shape)

        # Prediksi
        pred = model.predict(img_array)[0]
        predicted_index = np.argmax(pred)
        predicted_label = class_names[predicted_index]

        # Tampilkan hasil
        st.success(f"### 🧠 Prediction (EN): {predicted_label}")
        
        label_id = predicted_label.lower()
        if label_id == "advanced":
            st.error("**🦷 Prediksi (ID): Kerusakan Parah**\n\nGigi menunjukkan kerusakan berat dan sebaiknya segera ditangani dokter.")
        elif label_id == "early":
            st.warning("**🦷 Prediksi (ID): Tahap Awal**\n\nGigi menunjukkan tanda-tanda awal kerusakan. Perlu perawatan dini.")
        else:
            st.success("**🦷 Prediksi (ID): Sehat**\n\nGigi dalam kondisi sehat. Pertahankan kebiasaan menjaga kebersihan mulut!")

        st.markdown("---")
        st.write("Probabilitas Kelas:")
        for i, prob in enumerate(pred):
            st.write(f"- {class_names[i]}: {prob:.2%}")
