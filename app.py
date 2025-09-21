import streamlit as st
import fitz
import spacy
import re
import cv2
from pyzbar.pyzbar import decode
from PIL import Image
from pdf2image import convert_from_path
import json
import os
import numpy as np

# -------------------------------
# Load NLP model
# -------------------------------
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_trf")

nlp = load_model()

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Regex patterns for PII
regex_patterns = {
    "PHONE": r"\+?\d[\d -]{8,12}\d",
    "AADHAAR": r"\d{4}-\d{4}-\d{4}",
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
}

# -------------------------------
# PII Detection (Text)
# -------------------------------
def detect_pii_text(text):
    pii = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON","GPE","ORG","DATE","CARDINAL","MONEY"]:
            pii.append(ent.text)
    for label, pattern in regex_patterns.items():
        pii.extend(re.findall(pattern, text))
    return list(set(pii))

# -------------------------------
# Visual Redaction (Face, Signatures, QR/Barcode)
# -------------------------------
def redact_visual(img_path, out_path, style="black"):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def apply_style(x, y, w, h):
        roi = img[y:y+h, x:x+w]
        if style == "black":
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,0), -1)
        elif style == "blur":
            if roi.size > 0:
                roi_blur = cv2.GaussianBlur(roi, (31,31), 30)
                img[y:y+h, x:x+w] = roi_blur
        elif style == "pseudonym":
            cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), -1)
            cv2.putText(img, "XXXX", (x+5, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    # Faces
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces:
        apply_style(x,y,w,h)

    # Signatures / handwritten ink
    _, thresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)
    contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if 50<w<500 and 10<h<200:
            apply_style(x,y,w,h)

    # QR/Barcode
    barcodes = decode(Image.open(img_path))
    for bc in barcodes:
        (x,y,w,h) = bc.rect
        apply_style(x,y,w,h)

    cv2.imwrite(out_path, img)
    return out_path

# -------------------------------
# PDF Redaction with Audit Log
# -------------------------------
def redact_pdf_with_log(input_pdf, output_pdf, style="black", log_file="audit_log.json"):
    log_data = {"file": input_pdf, "redactions": []}
    doc = fitz.open(input_pdf)

    for i, page in enumerate(doc):
        # --- Text PII ---
        text = page.get_text()
        pii_list = detect_pii_text(text)
        for pii in pii_list:
            for area in page.search_for(pii):
                page.add_redact_annot(area, fill=(0,0,0))
            log_data["redactions"].append({"page": i+1, "type": "text", "value": pii})
        page.apply_redactions()

        # --- Image PII ---
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_img = doc.extract_image(xref)
            ext = base_img["ext"]
            img_path = f"page{i}_img{img_index}.{ext}"
            with open(img_path,"wb") as f:
                f.write(base_img["image"])

            redacted_path = f"redacted_{img_path}"
            redact_visual(img_path, redacted_path, style=style)

            with open(redacted_path,"rb") as f:
                doc.update_stream(xref, f.read())

            log_data["redactions"].append({"page": i+1, "type": "image", "action": style})

    doc.save(output_pdf)

    with open(log_file,"w") as f:
        json.dump(log_data,f,indent=2)

    return output_pdf, log_file

# -------------------------------
# Image Redaction (JPG/PNG direct)
# -------------------------------
def redact_image_with_log(input_img, output_img, style="black", log_file="audit_log.json"):
    redact_visual(input_img, output_img, style=style)
    log_data = {"file": input_img, "redactions": [{"type": "image", "action": style}]}
    with open(log_file,"w") as f:
        json.dump(log_data,f,indent=2)
    return output_img, log_file

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="DeID-Guard – Privacy by Design", layout="centered")

st.title("🛡️ DeID-Guard – Privacy by Design")
st.write("Upload a PDF or Image and automatically de-identify sensitive text, faces, QR codes, and signatures.")

uploaded_file = st.file_uploader("Upload a PDF/Image", type=["pdf","jpg","png"])
style = st.radio("Choose Redaction Style", ["black", "blur", "pseudonym"])

if uploaded_file:
    st.success(f"File `{uploaded_file.name}` uploaded successfully!")

    if st.button("Run De-identification"):
        input_path = uploaded_file.name
        with open(input_path,"wb") as f:
            f.write(uploaded_file.getbuffer())

        if input_path.endswith(".pdf"):
            output_file, log_file = redact_pdf_with_log(input_path, "redacted_output.pdf", style=style)
            pages = convert_from_path(output_file, dpi=150)
            st.image(pages[0], caption="Redacted PDF Preview (Page 1)", use_container_width=True)
            with open(output_file,"rb") as f:
                st.download_button("⬇️ Download Redacted PDF", f, file_name="redacted.pdf")
        else:
            output_file, log_file = redact_image_with_log(input_path, "redacted_output.png", style=style)
            st.image(output_file, caption="Redacted Image", use_container_width=True)
            with open(output_file,"rb") as f:
                st.download_button("⬇️ Download Redacted Image", f, file_name="redacted.png")

        with open(log_file,"rb") as f:
            st.download_button("⬇️ Download Audit Log (JSON)", f, file_name="audit_log.json")
