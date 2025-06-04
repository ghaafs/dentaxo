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

# ========== SESSION STATE ========== #
if "prediction_count" not in st.session_state:
    st.session_state["prediction_count"] = 0

# ========== STYLE ========== #
st.markdown("""
    <style>
        body { background-color: #ffffff; }
        .main { background-color: #ffffff; }
        .css-1d391kg {background-color: #ffffff;}
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: #5D3FD3; }
        .stButton>button {
            background-color: #5D3FD3;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .footer {
            margin-top: 3rem;
            text-align: center;
            font-size: 0.9rem;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

# ========== SIDEBAR MENU ========== #
menu = st.sidebar.radio("📁 Navigasi", [
    "🏠 Dashboard",
    "📘 Petunjuk Penggunaan",
    "ℹ️ Tentang DENTAXO",
    "📤 Upload & Deteksi",
    "📊 Grafik Akurasi"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"🧮 Total Prediksi Sesi Ini: **{st.session_state['prediction_count']}**")

# ========== DASHBOARD ========== #
if menu == "🏠 Dashboard":
    st.title("🦷 DENTAXO")
    st.markdown("""
    **DENTAXO** adalah aplikasi klasifikasi kondisi gigi berbasis deep learning yang dapat mengenali tiga kategori:
    
    - 🟥 Advanced (Kerusakan Parah)
    - 🟨 Early (Tahap Awal)
    - 🟩 Healthy (Sehat)

    Dirancang untuk membantu masyarakat dan profesional kesehatan dalam mengenali kondisi gigi dari gambar secara cepat dan akurat.
    """)

# ========== PETUNJUK ========== #
elif menu == "📘 Petunjuk Penggunaan":
    st.header("📘 Petunjuk Penggunaan")
    st.markdown("""
    **🛠️ DENTAXO masih dalam tahap pengembangan awal.**  
    Untuk akurasi terbaik, silakan unggah gambar **1–3 gigi saja**, bukan seluruh gigi.

    **Contoh Gambar yang Direkomendasikan:**
    - 1 gigi berlubang (misal geraham belakang)
    - 2 gigi depan dengan perubahan warna
    - Foto close-up dari 1 bagian gigi rusak
    
    ---

    **Langkah-langkah:**
    1. Siapkan gambar gigi yang jelas dan fokus.
    2. Masuk ke menu **Upload & Deteksi**.
    3. Klik **Upload**, pilih gambar dari perangkat Anda.
    4. Lihat hasil prediksi dan probabilitas.

    ⚠️ Gambar buram atau terlalu gelap dapat menghasilkan prediksi tidak akurat.
    """)

# ========== TENTANG DENTAXO ========== #
elif menu == "ℹ️ Tentang DENTAXO":
    st.header("ℹ️ Tentang DENTAXO")
    st.markdown("""
    **DENTAXO** adalah sistem klasifikasi citra berbasis AI untuk mendeteksi kondisi gigi manusia.

    **Kategori Klasifikasi:**
    - **Advanced**: Lubang besar, infeksi, gigi rusak berat
    - **Early**: Tanda awal karies, plak, perubahan warna ringan
    - **Healthy**: Tidak ditemukan tanda kerusakan signifikan

    Dibangun menggunakan teknologi **Deep Learning CNN**, dan dilatih dengan ribuan gambar gigi untuk mengenali pola visual dari berbagai kondisi.

    ---

    <div class="footer">
        Aplikasi ini dikembangkan oleh <b>Ghany Fitriamara Suci</b><br>
        Program Studi Fisika, Universitas Islam Negeri (UIN) Jakarta<br>
        Tahun 2025
    </div>
    """, unsafe_allow_html=True)

# ========== UPLOAD & DETEKSI ========== #
elif menu == "📤 Upload & Deteksi":
    st.header("📤 Upload Gambar Gigi")
    uploaded_file = st.file_uploader("Silakan upload gambar gigi (format JPG/PNG, maksimal 1–3 gigi per gambar):", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='📷 Gambar yang Diupload', use_container_width=True)

        try:
            image = image.convert("RGB")
            image = image.resize((128, 128))
            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype('float32') / 255.0

            pred = model.predict(img_array)[0]
            predicted_index = np.argmax(pred)
            predicted_label = class_names[predicted_index]

            st.session_state["prediction_count"] += 1

            st.success(f"### 🧠 Prediction (EN): {predicted_label}")

            label_id = predicted_label.lower()
            if label_id == "advanced":
                st.error("**🦷 Prediksi (ID): Kerusakan Parah**\n\nGigi menunjukkan kerusakan berat. Segera konsultasikan ke dokter.")
            elif label_id == "early":
                st.warning("**🦷 Prediksi (ID): Tahap Awal**\n\nAda indikasi awal kerusakan. Lakukan perawatan dini.")
            else:
                st.success("**🦷 Prediksi (ID): Sehat**\n\nGigi dalam kondisi sehat. Pertahankan kebersihan gigi secara rutin!")

            st.markdown("---")
            st.subheader("📈 Probabilitas Kelas:")
            for i, prob in enumerate(pred):
                st.write(f"- {class_names[i]}: {prob:.2%}")

        except Exception as e:
            st.error("⚠️ Terjadi kesalahan saat memproses gambar.")
            st.exception(e)

# ========== GRAFIK AKURASI ========== #
elif menu == "📊 Grafik Akurasi":
    st.header("📊 Grafik Akurasi Model DENTAXO")
    try:
        acc = np.load("accuracy.npy")
        val_acc = np.load("val_accuracy.npy")

        st.line_chart({
            "Training Accuracy": acc,
            "Validation Accuracy": val_acc
        })

        st.markdown("""
        Grafik ini menunjukkan peningkatan akurasi model selama proses training.
        - 📘 **Training Accuracy**: Akurasi saat model belajar dari data training.
        - 🧪 **Validation Accuracy**: Akurasi saat diuji pada data validasi.
        """)
    except Exception:
        st.warning("📁 File `accuracy.npy` atau `val_accuracy.npy` belum tersedia. Silakan unggah file akurasi terlebih dahulu.")
