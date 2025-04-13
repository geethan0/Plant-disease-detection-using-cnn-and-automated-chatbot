import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import google.generativeai as gen_ai
import base64
from weasyprint import HTML
from io import BytesIO

#  Load Plant Disease Model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model/plant_disease_prediction_model.h5")
model = tf.keras.models.load_model(model_path)

with open(os.path.join(working_dir, "class_indices.json")) as f:
    class_indices = json.load(f)
index_to_class = {int(k): v for k, v in class_indices.items()}

#  Image Preprocessing
def load_and_preprocess_image(image):
    img = Image.open(image).resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    return img_array

#  Prediction with Confidence Score
def predict_image_class(model, image):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)[0]
    
    sorted_indices = np.argsort(predictions)[::-1]  # Get top predictions
    top_3 = [(index_to_class[idx], predictions[idx] * 100) for idx in sorted_indices[:3]]
    
    return top_3

#  Secure API Key Handling
GOOGLE_API_KEY = "AIzaSyDXIk5iw2cuqnxz-sozRc_SzVSoXEIq4dQ"

if GOOGLE_API_KEY:
    gen_ai.configure(api_key=GOOGLE_API_KEY)
    chatbot_model = gen_ai.GenerativeModel('gemini-1.5-flash')

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = chatbot_model.start_chat(history=[])

#  Get Remedies from Chatbot
def get_remedies_from_chatbot(disease_name):
    prompt = f"The detected plant disease is **{disease_name}**. Suggest detailed remedies and best agricultural practices in clear bullet points."
    response = st.session_state.chat_session.send_message(prompt)
    return response.text

#  Download Remedies as PDF (Includes Image)
def generate_pdf(image, disease, remedies, filename="remedies.pdf"):
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

    formatted_remedies = remedies.replace("\n", "<br>")  # Preserve formatting

    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; padding: 20px;">
        <h1 style="color: green; text-align: center;">🌱 Plant Disease Report</h1>
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_base64}" alt="Uploaded Image" width="300" style="display: block; margin: 10px auto;">
        </div>
        <h2 style="color: red; text-align: center;">Detected Disease: {disease}</h2>
        <h3 style="margin-top: 20px;">💡 Remedies & Recommendations:</h3>
        <p style="line-height: 1.6;">{formatted_remedies}</p>
    </body>
    </html>
    """
    
    pdf_path = os.path.join(working_dir, filename)
    HTML(string=html_content).write_pdf(pdf_path)
    
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    
    return base64_pdf

#  Download Chat History as PDF (Proper Formatting)
def generate_chat_pdf(filename="chat_history.pdf"):
    chat_history = st.session_state.chat_session.history
    
    formatted_chat = ""
    for message in chat_history:
        if message.role == "user":
            formatted_chat += f'<p style="color: red; font-weight: bold;">User:</p><p style="border-left: 4px solid red; padding-left: 10px;">{message.parts[0].text}</p>'
        else:
            formatted_chat += f'<p style="color: green; font-weight: bold;">Assistant:</p><p style="border-left: 4px solid green; padding-left: 10px;">{message.parts[0].text}</p>'
    
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 20px;
            }}
            h1 {{
                color: blue;
                text-align: center;
            }}
            .chat-container {{
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 10px;
            }}
            p {{
                margin: 5px 0;
            }}
        </style>
    </head>
    <body>
        <h1>🔍 Chat History</h1>
        <div class="chat-container">
            {formatted_chat}
        </div>
    </body>
    </html>
    """
    
    pdf_path = os.path.join(working_dir, filename)
    HTML(string=html_content).write_pdf(pdf_path)
    
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    
    return base64_pdf

#  UI Enhancements
st.set_page_config(page_title="🌱 Plant Disease & Remedies AI", layout="wide")

# Layout Setup
col1, spacer, col2 = st.columns([1, 0.05, 1])

#  Image Upload
with col1:
    st.title("🌿 Plant Disease Classifier")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image.resize((150, 150)), caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Classify & Get Remedies"):
            with st.spinner("🔄 Analyzing..."):
                predictions = predict_image_class(model, uploaded_image)
                disease, confidence = predictions[0]  # Best prediction
                
                st.success(f'🌱 **Prediction:** {disease} ({confidence:.2f}% Confidence)')
                st.info("📊 **Alternative Predictions:**")
                for alt_disease, alt_conf in predictions[1:]:
                    st.write(f"- {alt_disease} ({alt_conf:.2f}%)")
                
                with st.spinner("🤖 Fetching remedies..."):
                    remedies = get_remedies_from_chatbot(disease)
                    st.info("💡 **Remedies & Recommendations:**")
                    st.markdown(remedies.replace("\n", "<br>"), unsafe_allow_html=True)
                    
                    pdf_base64 = generate_pdf(image, disease, remedies)
                    pdf_link = f'<a href="data:application/pdf;base64,{pdf_base64}" download="Plant_Disease_Remedies.pdf">💽 Download Remedies as PDF</a>'
                    st.markdown(pdf_link, unsafe_allow_html=True)

#  Chatbot Column
with col2:
    st.title("💬 ChatBot - Ask More!")
    
    if st.button("🛠 Clear Chat"):
        st.session_state.chat_session = chatbot_model.start_chat(history=[])
        st.rerun()
    
    for message in st.session_state.chat_session.history:
        with st.chat_message("user" if message.role == "user" else "assistant"):
            st.markdown(message.parts[0].text)
    
    user_query = st.chat_input("Ask about the disease or remedies...")
    if user_query:
        st.chat_message("user").markdown(user_query)
        gemini_response = st.session_state.chat_session.send_message(user_query)
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)
    
    pdf_base64 = generate_chat_pdf()
    chat_pdf_link =f'<a href="data:application/pdf;base64,{pdf_base64}" download="Chat_History.pdf">💽 Download Chat as PDF</a>'
    st.markdown(chat_pdf_link, unsafe_allow_html=True)