import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ========== CONFIG ========== #
st.set_page_config(
    page_title="DENTAXO - Dental AI",
    page_icon="🦷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load model
model = load_model('ICDAS-80-part5.h5')
class_names = ['Advanced', 'Early', 'Healthy']

# Style CSS untuk tema ungu + hijau modern dan minimalis
st.markdown("""
<style>
    /* Background gradient putih ke hijau muda */
    .main, body, .block-container {
        background: linear-gradient(135deg, #f7f9fc, #d8f1e4);
        color: #3b2e5a; /* ungu tua */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Judul besar ungu lembut */
    h1, h2, h3 {
        color: #6c5b7b; /* soft purple */
        font-weight: 700;
        margin-bottom: 0.3em;
    }

    /* Sidebar background hijau muda lembut */
    .css-1d391kg {
        background-color: #c8e6c9 !important; /* soft green */
        color: #2e3d27;
    }

    /* Sidebar teks ungu gelap */
    .css-1d391kg span, .css-1d391kg label {
        color: #3b2e5a !important;
        font-weight: 600;
    }

    /* Tombol utama hijau dengan efek hover */
    div.stButton > button {
        background-color: #4caf50; /* hijau segar */
        color: white;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(76, 175, 80, 0.3);
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #388e3c;
        cursor: pointer;
    }

    /* File uploader styling */
    .css-1d0p7cg {
        border: 3px dashed #6c5b7b !important; /* soft purple */
        border-radius: 16px !important;
        padding: 1.3rem !important;
        background-color: #f0f4f8;
    }

    /* Prediction box */
    .stAlert {
        border-radius: 14px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        font-style: italic;
        color: #555;
        padding: 1rem 0;
        border-top: 1px solid #d3d3d3;
        margin-top: 3rem;
    }

    /* Probabilities list */
    div.stMarkdown > ul > li {
        font-weight: 600;
        color: #4a3f61; /* purple shade */
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar menu
menu = st.sidebar.radio(
    "Menu DENTAXO",
    ["🏠 Dashboard", "📘 Cara Pakai", "ℹ️ Tentang", "📤 Upload Gambar"]
)

# ======= Dashboard =======
if menu == "🏠 Dashboard":
    st.title("🦷 DENTAXO")
    st.markdown("""
    Selamat datang di **DENTAXO** — aplikasi deteksi karies gigi berbasis AI yang membantu mengklasifikasikan kondisi gigi Anda menjadi tiga kategori:
    
    - 🔴 **Advanced** (Kerusakan berat)
    - 🟠 **Early** (Karies tahap awal)
    - 🟢 **Healthy** (Gigi sehat)
    
    Aplikasi ini dirancang untuk mendukung deteksi dini kerusakan gigi dengan citra digital.
    """)
    count = st.session_state.get("predictions", 0)
    st.info(f"Jumlah prediksi selama sesi ini: **{count}** kali")

# ======= Cara Pakai =======
elif menu == "📘 Cara Pakai":
    st.header("📘 Cara Menggunakan DENTAXO")
    st.markdown("""
    1. Siapkan gambar gigi dengan pencahayaan dan fokus yang baik.
    2. Gunakan menu **Upload Gambar** untuk mengunggah foto gigi (format JPG/PNG).
    3. Tunggu hasil prediksi muncul di layar.
    4. Gunakan hasil prediksi untuk konsultasi lebih lanjut dengan dokter gigi.
    
    ⚠️ Pastikan gambar hanya memuat 1-3 gigi agar hasil lebih akurat.
    """)

# ======= Tentang =======
elif menu == "ℹ️ Tentang":
    st.header("ℹ️ Tentang DENTAXO")
    st.markdown("""
    DENTAXO adalah aplikasi berbasis AI yang menggunakan model CNN untuk mengklasifikasikan citra gigi ke dalam:
    
    - **Advanced**: Kerusakan berat yang memerlukan penanganan segera.
    - **Early**: Kerusakan awal yang masih bisa diatasi dengan perawatan ringan.
    - **Healthy**: Gigi sehat tanpa tanda kerusakan.
    
    Aplikasi ini sebagai alat bantu dan tidak menggantikan pemeriksaan dokter.
    """)

# ======= Upload & Prediction =======
elif menu == "📤 Upload Gambar":
    st.header("📤 Unggah Gambar Gigi untuk Prediksi")
    uploaded_file = st.file_uploader("Pilih file gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

        # Preprocess
        img_resized = image.resize((128, 128))
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0).astype('float32') / 255.0

        # Predict
        prediction = model.predict(img_array)[0]
        pred_idx = np.argmax(prediction)
        pred_label = class_names[pred_idx]

        # Update session counter
        st.session_state["predictions"] = st.session_state.get("predictions", 0) + 1

        # Show result
        if pred_label == "Advanced":
            st.error("🦷 Prediksi: Kerusakan Parah (Advanced)\nSegera konsultasi ke dokter gigi.")
        elif pred_label == "Early":
            st.warning("🦷 Prediksi: Tahap Awal (Early)\nPerawatan ringan disarankan.")
        else:
            st.success("🦷 Prediksi: Gigi Sehat (Healthy)\nJaga kebersihan gigi Anda!")

        # Probabilities
        st.markdown("### 🔢 Probabilitas Tiap Kelas:")
        for i, prob in enumerate(prediction):
            st.write(f"- **{class_names[i]}**: {prob*100:.2f}%")

# ======= Footer =======
st.markdown("""
<div class="footer">
    &copy; 2025 DENTAXO - Dikembangkan oleh Ghany Fitriamara Suci<br>
    Program Studi Fisika, UIN Jakarta
</div>
""", unsafe_allow_html=True)
