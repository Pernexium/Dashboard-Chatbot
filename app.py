import json
import base64
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title = "Pernexium", page_icon = "/Users/hibrantapia/Library/CloudStorage/OneDrive-InstitutoTecnologicoydeEstudiosSuperioresdeMonterrey/Academic/Semester 7/Various/Pernexium/Análitica/analitica/Source/scriptsAWS/Dashboard-Chatbot/Varios/Logos/PXM isotipo 3.png", layout = "wide")


st.markdown("""
<style>
/* Estilos para centrar el contenido */
.css-1l02zno {
    display: flex;
    justify-content: center;
}
/* Estilos para centrar el texto markdown */
.stMarkdown {
    text-align: center;
}
h1 {
    font-size: 3.1em;
}
h2 {
    font-size: 1.6em;
}
h3 {
    font-size: 2.8em;
}
h4 {
    font-size: 2.4em;
}
h7 {
    font-size: 1.3em;
}
</style>
""", unsafe_allow_html = True)

import streamlit as st
import base64

import streamlit as st
import base64

def inicio_page():
    image_path = "/Users/hibrantapia/Library/CloudStorage/OneDrive-InstitutoTecnologicoydeEstudiosSuperioresdeMonterrey/Academic/Semester 7/Various/Pernexium/Análitica/analitica/Source/scriptsAWS/Dashboard-Chatbot/Varios/Logos/PXM Imagotipo 3.png"
    
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    st.markdown(f"""<img src="data:image/png;base64,{encoded_image}" width="60%">""", unsafe_allow_html=True)
    st.markdown("<h1>Dashboard del <span style='color: #27A3D7;'>Chatbot</span></h1>", unsafe_allow_html=True)

    correct_password = "ola"

    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        password = st.text_input("Introduce la contraseña para acceder al Dashboard:", type="password")
        
        if password == correct_password:
            st.session_state['authenticated'] = True
            st.success("¡Contraseña correcta! Puedes acceder a la página del Dashboard.")
        elif password:
            st.warning("Contraseña incorrecta. Inténtalo de nuevo.")
    else:
        st.markdown("<p>¡Bienvenido!</p>", unsafe_allow_html=True)

def dashboard_page():
    st.markdown("<h1>Dashboard del <span style='color: #27A3D7;'>Chatbot</span></h1>", unsafe_allow_html=True)

######################################################################################################################
    
def main():
    st.sidebar.image(
    "/Users/hibrantapia/Library/CloudStorage/OneDrive-InstitutoTecnologicoydeEstudiosSuperioresdeMonterrey/Academic/Semester 7/Various/Pernexium/Análitica/analitica/Source/scriptsAWS/Dashboard-Chatbot/Varios/Logos/logo.png", width=185)

    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'Inicio'

    page = st.sidebar.selectbox("Menu", ["Inicio", "Dashboard"])

    if page == "Inicio":
        inicio_page()
    elif page == "Dashboard":
        if 'authenticated' in st.session_state and st.session_state['authenticated']:
            dashboard_page()

        else:
            st.markdown("<h1>Dashboard del <span style='color: #27A3D7;'>Chatbot</span></h1>", unsafe_allow_html=True)
            st.write("Aquí puedes ver el Dashboard del Chatbot, pero antes...")
            st.error("Por favor, ingresa la contraseña en la página de inicio para acceder al Dashboard.")


    st.sidebar.markdown("---")

    st.sidebar.markdown("### Contacto")
    st.sidebar.markdown(
     """
     @Pernexium<br>
     <div style="display:flex; align-items:center; gap:10px;">
         <a href="https://www.pernexium.com/" style="display:flex; align-items:center; margin-right:15px; margin-left:85px;">
            <div style="background-color:white; width:30px; height:30px; display:flex; justify-content:center; align-items:center; border-radius:50%;">
                <img src="https://www.svgrepo.com/show/438256/world-round.svg" alt="Web Page" style="width:45px; height:45px;">
            </div>
         </a>
         <a href="https://www.instagram.com/pernexium/" style="display:flex; align-items:center; margin-right:15px;">
            <div style="background-color:white; width:30px; height:30px; display:flex; justify-content:center; align-items:center; border-radius:50%;">
                 <img src="https://www.svgrepo.com/show/494277/instagram-round.svg" alt="Instagram" style="width:40px; height:40px;">
            </div>
         </a>
        <a href="https://www.linkedin.com/company/pernexium/" style="display:flex; align-items:center;">
            <div style="background-color:white; width:30px; height:30px; display:flex; justify-content:center; align-items:center; border-radius:50%;">
                <img src="https://www.svgrepo.com/show/494278/linkedin-round.svg" alt="LinkedIn" style="width:40px; height:40px;">
            </div>
        </a>

     </div>
     """,
     unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="color: grey;">© 2024 Pernexium.</p>', 
        unsafe_allow_html=True)

if __name__ == "__main__":
    main()