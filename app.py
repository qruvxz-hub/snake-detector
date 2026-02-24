import tf_keras
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="ระบบคัดแยกชนิดงู เชียงราย", page_icon="🐍")

st.title("🐍 ระบบคัดแยกชนิดของงู (เชียงราย)")
st.write("Mini Project โดย ด.ช.ภัทชภณ และ ด.ญ.ณกัญญา")

# --- 1. โหลดโมเดล (โหลดครั้งเดียว) ---
@st.cache_resource
def load_my_model():
    model = tf_keras.models.load_model("keras_model.h5", compile=False) # ใช้ tf_keras ตรงนี้
    with open("labels.txt", "r", encoding="utf-8") as f:
        class_names = f.readlines()
    return model, class_names

try:
    model, class_names = load_my_model()
except Exception as e:
    st.error(f"หาไฟล์โมเดลไม่เจอ! ตรวจสอบว่ามีไฟล์ keras_model.h5 และ labels.txt หรือยัง? \n error: {e}")
    st.stop()

# --- 2. ส่วนแสดงตารางข้อมูล (Intro) ---
with st.expander("📊 ดูตารางรายชื่อและประเภทงู"):
    st.markdown("""
    | สัญลักษณ์ | ประเภทงู | รายชื่อ |
    | :---: | :--- | :--- |
    | ❤️ | มีพิษ (อันตราย) | งูสามเหลี่ยม, งูจงอาง, งูเห่า, งูเขียวหางไหม้, งูทับสมิงคลา, งูลายสาบคอแดง, งูหางแห้มภูเขา, งูปล้องหวายหัวดำ, งูลายสาบเขียวขวั้นดำ |
    | 💚 | ไม่มีพิษ | งูลายสอ, งูเขียวพระอินทร์, งูทางมะพร้าว, งูสิงหางดำ, งูสิงหางลาย, งูสิงตาโต, งูหมอก, งูปากจิ้งจก, งูงอด, งูทางมะพร้าวแดง, งูทางมะพร้าวเขียว, งูเหลือม, งูหลาม, งูปล้องฉนวน, งูปล้องทอง, งูกินทากเกล็ดสันเมืองเหนือ, งูกินทากลายจุด, งูกินทากหัวโหนก, งูปี่แก้วลายแต้ม, งูปี่แก้วลายจุด, งูปี่แก้วลายหัวใจ, งูดิน, งูปลิงสองสี, งูแส้หางม้าเทา, งูลายสาบดอกหญ้า, งูแม่ตะงาวรังนก, งูคอควั่นดำ, งูลายสาบภูเขา, งูหัวลายลูกศร |
    | 🤍 | ไม่ใช่รูปงู | มนุษย์ / สิ่งของ |
    """)

# --- 3. ส่วนเปิดกล้อง ---
img_file = st.camera_input("ถ่ายรูปงูเพื่อวิเคราะห์")

if img_file:
    # --- 4. เตรียมรูปภาพ ---
    image = Image.open(img_file).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # แปลงรูปเป็น array ที่ Model เข้าใจ
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # --- 5. พยากรณ์ผล ---
    with st.spinner('กำลังวิเคราะห์...'):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

    # --- 6. แสดงผลลัพธ์ ---
    st.subheader(f"🔍 ผลวิเคราะห์: {class_name}")
    st.write(f"📊 ความมั่นใจ: {confidence_score:.2%}")

    if "❤️" in class_name:
        st.error("🚨 อันตราย: ตรวจพบงูมีพิษ! โปรดถอยห่าง")
    elif "💚" in class_name:
        st.success("✅ ข้อมูล: งูไม่มีพิษร้ายแรง แต่ควรระวัง")
    elif "🤍" in class_name:
        st.info("👤 สถานะ: ไม่พบงู หรือเป็นมนุษย์")
    else:
        st.warning("⚠️ ไม่สามารถระบุได้ชัดเจน")
