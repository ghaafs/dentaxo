import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Daftar nama kelas ICDAS
class_names = ['Advanced', 'Early', 'Healthy']

# Load model
model = tf.keras.models.load_model('ICDAS-90-part5.h5')

# Set ukuran input gambar
img_size = (128, 128)

# UI Streamlit
st.title("🦷 DENTAXO 🦷")
st.write("Upload gambar gigi untuk mendeteksi tingkat keparahan karies berdasarkan ICDAS.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar gigi...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file)
    st.image(img, caption='Gambar Diupload', use_container_width=True)

    # Preprocessing
    img = img.resize(img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    pred = model.predict(img_array)[0]
    predicted_label = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    # Output
    st.markdown(f"### 🧠 Prediksi: **{predicted_label}**")
    st.markdown(f"Confidence: `{confidence:.2f}%`")

    # Display hasil per kelas
    st.write("### 🔎 Detail Probabilitas Kelas:")
    for idx, prob in enumerate(pred):
        st.write(f"{class_names[idx]}: `{prob * 100:.2f}%`")
    
    # Notifikasi berdasarkan prediksi
    if predicted_label == "Healthy":
        st.success("✅ Tidak terdeteksi karies. Tetap jaga kebersihan gigi ya!")
    elif predicted_label == "Early":
        st.warning("⚠️ Terdeteksi karies ringan. Disarankan untuk menjaga kebersihan dan periksa rutin.")
    elif predicted_label == "Advanced":
        st.error("🚨 Karies terdeteksi cukup parah. SEGERA periksa ke dokter gigi untuk penanganan lebih lanjut.")


st.markdown("""
<hr style="border:1px solid gray;margin-top:20px;">
<p style='text-align: center; color: gray; font-size: small;'>
     Made with ❤️ by Ghany Fitriamara
</p>
""", unsafe_allow_html=True)


