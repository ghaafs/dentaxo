import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model sekali saja
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('final_karies_model_baru.h5')
    return model

model = load_model()

# Judul web
st.title("🦷 Deteksi Karies Gigi dengan Transfer Learning")

st.write("Upload gambar gigi Anda untuk prediksi karies atau sehat.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar gigi...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file)
    st.image(img, caption='Gambar Diupload', use_column_width=True)

    # Preprocessing
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    pred = model.predict(img_array)[0]
    classes = ['Karies', 'Sehat']
    predicted_label = classes[np.argmax(pred)]
    confidence = np.max(pred)

    # Output
    st.markdown(f"### 🧠 Prediksi: **{predicted_label}**")
    st.markdown(f"Confidence: `{confidence:.2f}`")

    if predicted_label == "Karies":
        st.error("⚠️ Hasil menunjukkan adanya kemungkinan karies. Disarankan periksa ke dokter.")
    else:
        st.success("✅ Tidak terdeteksi karies. Tetap jaga kebersihan gigi ya!")