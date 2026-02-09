import streamlit as st
import cv2
import os
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np

# ================= SESSION STATE =================
if "stop_attendance" not in st.session_state:
    st.session_state.stop_attendance = False

# ================= CONFIG =================
CASCADE_PATH = "haarcascade_frontalface_default.xml"
DATASET_DIR = "dataset"
DB_NAME = "attendance.db"
TOTAL_IMAGES = 50
FACE_SIZE = (200, 200)
LABEL_MAP_FILE = "label_map.npy"

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS classes(
        class_name TEXT PRIMARY KEY,
        max_students INTEGER)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS students(
        id INTEGER PRIMARY KEY,
        name TEXT,
        class_name TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS attendance(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        date TEXT,
        time TEXT)""")
    conn.commit()
    conn.close()

def insert_class(cname, limit):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO classes VALUES (?,?)",(cname,limit))
    conn.commit()
    conn.close()

def delete_class(cname):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM classes WHERE class_name=?", (cname,))
    cur.execute("DELETE FROM students WHERE class_name=?", (cname,))
    conn.commit()
    conn.close()

def delete_student_completely(sid):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM students WHERE id=?", (sid,))
    cur.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
    conn.commit()
    conn.close()
    if os.path.exists(DATASET_DIR):
        for f in os.listdir(DATASET_DIR):
            if f.startswith(f"user.{sid}."):
                os.remove(os.path.join(DATASET_DIR, f))

def get_student_details(sid):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT name, class_name FROM students WHERE id=?", (sid,))
    r = cur.fetchone()
    conn.close()
    return r if r else ("Unknown", "Unknown")

def mark_attendance(sid):
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM attendance WHERE student_id=? AND date=?", (sid, today))
    if not cur.fetchone():
        cur.execute("INSERT INTO attendance(student_id,date,time) VALUES(?,?,?)", (sid, today, now))
    conn.commit()
    conn.close()

# ================= TRAIN MODEL =================
def train_model():
    if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
        return None, None
    faces, labels, label_map = [], [], {}
    current_label = 0
    for file in os.listdir(DATASET_DIR):
        if file.endswith(".jpg"):
            try:
                sid = int(file.split('.')[1])
                if sid not in label_map.values():
                    label_map[current_label] = sid
                    current_label += 1
                lbl = [k for k, v in label_map.items() if v == sid][0]
                img = cv2.imread(os.path.join(DATASET_DIR, file), cv2.IMREAD_GRAYSCALE)
                faces.append(cv2.resize(img, FACE_SIZE))
                labels.append(lbl)
            except: continue
    np.save(LABEL_MAP_FILE, label_map)
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.train(faces, np.array(labels))
    return rec, label_map

# ================= ENROLL WITH FACE DUPLICATE CHECK =================
def capture_and_enroll(sid, sname, cname):
    os.makedirs(DATASET_DIR, exist_ok=True)
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    rec, label_map = train_model()
    
    count, duplicate_found = 0, False
    frame_box = st.empty()
    bar = st.progress(0)

    while count < TOTAL_IMAGES:
        ret, frame = cam.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            face_img = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
            if rec is not None:
                lab, conf = rec.predict(face_img)
                if conf < 50:
                    existing_sid = label_map.get(lab, -1)
                    ex_name, _ = get_student_details(existing_sid)
                    st.error(f"❌ Face already enrolled as: **{ex_name}** (ID: {existing_sid})")
                    duplicate_found = True
                    break

            count += 1
            cv2.imwrite(f"{DATASET_DIR}/user.{sid}.{count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        if duplicate_found: break
        bar.progress(count / TOTAL_IMAGES)
        frame_box.image(frame, channels="BGR")

    cam.release()
    if not duplicate_found and count >= TOTAL_IMAGES:
        conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO students VALUES (?,?,?)", (sid, sname, cname))
        conn.commit(); conn.close()
        st.success(f"✅ Student {sname} enrolled successfully!")

# ================= RECOGNITION & ATTENDANCE =================
def recognize_and_mark():
    rec, label_map = train_model()
    if rec is None:
        st.error("No trained data found."); return
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    frame_box = st.empty()
    status_box = st.empty()

    while not st.session_state.stop_attendance:
        ret, frame = cam.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = detector.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in detected:
            roi = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
            lab, conf = rec.predict(roi)
            sid = label_map.get(lab, -1)
            
            if conf < 65 and sid != -1:
                name, cls = get_student_details(sid)
                mark_attendance(sid)
                status_box.success(f"📌 **Recognized:** {name} | **ID:** {sid} | **Class:** {cls}")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "UNKNOWN", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        frame_box.image(frame, channels="BGR")
    cam.release()

# ================= MAIN UI =================
st.set_page_config("Smart Attendance", "🎓", layout="wide")
st.title("🎓 Smart Attendance System")
init_db()

menu = st.sidebar.radio("📌 Menu", [
    "📚 Classes Overview", 
    "🏫 Create Class", 
    "👤 Enroll Student", 
    "📸 Mark Attendance", 
    "📊 View Attendance", 
    "🗑️ Delete Student"
])

if menu == "📚 Classes Overview":
    conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
    cur.execute("SELECT class_name, max_students FROM classes")
    classes = cur.fetchall(); conn.close()
    if not classes:
        st.info("No classes found.")
    for cname, limit in classes:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.subheader(f"🏫 Class: {cname} (Max: {limit})")
            if c2.button("🗑️ Delete Class", key=f"del_class_{cname}"):
                delete_class(cname)
                st.rerun()
            st.dataframe(pd.read_sql(f"SELECT id as ID, name as Name FROM students WHERE class_name='{cname}'", sqlite3.connect(DB_NAME)), use_container_width=True)

elif menu == "🏫 Create Class":
    cname, limit = st.text_input("Class Name"), st.number_input("Max Students", 1)
    if st.button("Create"):
        insert_class(cname, limit); st.success("Created!")

elif menu == "👤 Enroll Student":
    cname, sid, sname = st.text_input("Class Name"), st.number_input("ID", 1), st.text_input("Name")
    if st.button("Enroll"): capture_and_enroll(sid, sname, cname)

elif menu == "📸 Mark Attendance":
    col1, col2 = st.columns(2)
    if col1.button("▶ Start"): 
        st.session_state.stop_attendance = False
        recognize_and_mark()
    if col2.button("🛑 Stop"): st.session_state.stop_attendance = True

elif menu == "📊 View Attendance":
    conn = sqlite3.connect(DB_NAME)
    # COALESCE ensures that if name or class is NULL (not found), it shows 'Unknown'
    query = """
        SELECT 
            COALESCE(s.name, 'Unknown') as Name, 
            a.student_id as ID, 
            a.date as Date, 
            a.time as Time, 
            COALESCE(s.class_name, 'Unknown') as Class 
        FROM attendance a 
        LEFT JOIN students s ON a.student_id = s.id 
        ORDER BY a.date DESC, a.time DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    st.subheader("Attendance Records")
    # Display table
    st.dataframe(df, use_container_width=True)
    
    # Download Button
    if not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Attendance as CSV",
            data=csv,
            file_name=f"attendance_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
        )

elif menu == "🗑️ Delete Student":
    del_id = st.number_input("Enter Student ID to remove", 1)
    if st.button("Delete Student"):
        delete_student_completely(del_id)
        st.success(f"ID {del_id} and associated data removed.")