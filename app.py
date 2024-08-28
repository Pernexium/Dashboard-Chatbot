import pytz
import json
import boto3
import base64
import random
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from io import BytesIO
import streamlit as st
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")


######################################################################################################################


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title = "Pernexium", page_icon = "./Varios/Logos/PXM isotipo 3.png", layout = "wide")


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


######################################################################################################################


def read_aws_credentials(file_path):
    credentials = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            credentials[key] = value
    return credentials


######################################################################################################################


def create_aws_session(file_path):
    credentials = read_aws_credentials(file_path)
    session = boto3.Session(
        aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=credentials['AWS_SECRET_ACCESS_KEY'],
        region_name=credentials['AWS_REGION']
    )
    return session


######################################################################################################################


def obteniendo_df_bancoppel():
    """Función para conectar a S3 y obtener el DataFrame de BanCoppel."""
    session = create_aws_session('./credentials_aws.txt')
    s3_client = session.client('s3')

    current_date = datetime.now().strftime('%Y_%m')
    bucket_name = 's3-pernexium-report'
    prefix = f'raw/bancoppel/replica_dynamo_chatbot/{current_date}/'
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    files = response.get('Contents', [])
    
    if files: 
        files_sorted = sorted(files, key=lambda x: x['LastModified'], reverse=True)
        latest_file = files_sorted[0]['Key']
        response = s3_client.get_object(Bucket=bucket_name, Key=latest_file)
        content = response['Body'].read()
        json_data = json.loads(content)
        
        if isinstance(json_data, list):
            df_bancoppel = pd.DataFrame(json_data)
        else:
            df_bancoppel = pd.json_normalize(json_data)
        
        conversations = []
        for index, item in tqdm(df_bancoppel.iterrows(), total=df_bancoppel.shape[0]):
            temp_item = []
            for k in item['messages']:
                if 'content' not in k.keys():
                    continue
                if "Template" in k['content']:
                    k['content'] = " " + k['content']  
                temp_item.append(k)
            
            tag = item['tag'] if 'tag' in item.keys() else "sin_info_para_clasificar"
            rand_conversationid = random.randint(1e6, 1e8)
            
            for it in range(len(temp_item)):
                temp_item[it]['conversationId'] = rand_conversationid
                temp_item[it]['credit'] = item['credit']
                temp_item[it]['discount'] = item['discount']
                temp_item[it]['phone1'] = item['contact_phone']
                temp_item[it]['last_template_at'] = item['last_template_at']
                temp_item[it]['tag'] = tag
                temp_item[it]['send_status'] = item['status'] if 'status' in item.keys() else 'sin_estatus'
            
            conversations.extend(temp_item)

        conversations = pd.DataFrame(conversations)
        conversations.drop("send_status", axis=1, inplace=True)
        conversations.rename(columns={"status": "sent_status"}, inplace=True)
        conversations['sent_status'].fillna("sin_estatus", inplace=True)
        conversations['last_template_at'] = pd.to_datetime(conversations['last_template_at'])
        conversations['last_template_at'] = conversations['last_template_at'].dt.tz_convert('America/Mexico_City')
        conversations['last_template_at'] = conversations['last_template_at'].dt.tz_localize(None)
        conversations['createdAt'] = pd.to_datetime(conversations['createdAt'])
        conversations['createdAt'] = conversations['createdAt'].dt.tz_convert('America/Mexico_City')
        conversations['createdAt'] = conversations['createdAt'].dt.tz_localize(None)
        conversations.drop_duplicates(['createdAt', 'phone1', 'credit', 'content', 'role'], inplace=True)
        conversations.reset_index(drop=True, inplace=True)
        conversations['tag'] = conversations['tag'].replace(
            'no defineno defineno defineno defineno defineno defineno defineno defineno defineno define', 
            'no define'
        )

        data = conversations.copy()
        serie_count_celphones = pd.DataFrame(data.groupby("conversationId").phone1.nunique().sort_values()).reset_index()
        data = data.loc[data.conversationId.isin(serie_count_celphones.query("phone1 == 1").conversationId)]
        data.dropna(subset=["phone1", 'credit'], inplace=True)
        data['id'] = data.apply(lambda row: f"{row['credit']} - {row['phone1']}", axis = 1)
        data.credit = data.credit.astype(str)
        data.conversationId = data.conversationId.astype(str)
        data.reset_index(drop=True, inplace=True)
        data = data.sort_values(by = ['createdAt', 'content'])
        data = data.reset_index()
        for i, row in data.iterrows():
            if "Template" in row['content']:
                if i + 1 < len(data):
                    data.at[i, "sent_status"] = data.at[i+1, "sent_status"]
        dummies = pd.get_dummies(data['sent_status'], prefix='sent_status')
        data = pd.concat([data.drop('sent_status', axis=1), dummies], axis=1)

        df_bancoppel_conversations = data

        return df_bancoppel, df_bancoppel_conversations
    else:
        print("No se encontraron archivos para la fecha actual.")
        return None, None


######################################################################################################################


def inicio_page():
    image_path = "./Varios/Logos/PXM Imagotipo 3.png"
    
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
            st.success("¡Contraseña correcta! Puedes acceder a la página del Dashboard, pero primero solicita los datos en la página de 'Solicitar Datos'.")
        elif password:
            st.warning("Contraseña incorrecta. Inténtalo de nuevo.")
    else:
        st.markdown("<p>¡Bienvenido!</p>", unsafe_allow_html=True)


######################################################################################################################


def dashboard_page():
    st.markdown("<h1>Dashboard del <span style='color: #27A3D7;'>Chatbot</span></h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    if 'df_bancoppel' not in st.session_state or st.session_state.df_bancoppel is None:
        st.error("Por favor, solicita los datos primero en la página de 'Solicitar Datos' antes de acceder al Dashboard.")
    else:
        if 'df_bancoppel_conversations' not in st.session_state or st.session_state.df_bancoppel_conversations is None:
            st.error("Por favor, solicita los datos primero en la página de 'Solicitar Datos' antes de acceder al Dashboard.")
        else:
            df_bancoppel = st.session_state.df_bancoppel
            df_bancoppel_conversations = st.session_state.df_bancoppel_conversations
            
            respuestas = 0
            indexes_templates = np.where(df_bancoppel_conversations.content.str.contains("Template"))[0]
            for index in range(len(indexes_templates) - 1):
                if (indexes_templates[index + 1] - indexes_templates[index]) > 2:
                    respuestas += 1
            
            st.write(df_bancoppel_conversations.query("content.str.contains('Template')").role.value_counts())
            
            total_envios = df_bancoppel_conversations.query("content.str.contains('Template')").shape[0]
            total_respuestas = df_bancoppel_conversations.query("direction == 'incoming'").shape[0]

            if total_envios > 0: 
                porcentaje_respuestas = (total_respuestas / total_envios) * 100
            else:
                porcentaje_respuestas = 0

            st.write(f"Porcentaje de respuestas recibidas: {porcentaje_respuestas:.2f}%")
            
            # Código para el gráfico de distribución de tags
            tag_counts = df_bancoppel_conversations.query("content.str.contains('Template')").tag.value_counts()

            fig_pie = px.pie(
                values=tag_counts.values,
                names=tag_counts.index,
                title='Distribución de Tags',
                width=600,
                height=400,
                hole=0.4  
            )

            st.plotly_chart(fig_pie)
            
            # Código para el gráfico de embudo
            filtered_data = df_bancoppel_conversations.query("content.str.contains('Template')")

            response_count = total_respuestas  # o respuestas
            read_count = filtered_data['sent_status_read'].sum() 
            delivered_count = filtered_data['sent_status_delivered'].sum() + read_count
            sent_count = filtered_data['sent_status_sent'].sum() + read_count + delivered_count

            values = [sent_count, delivered_count, read_count, response_count]
            stages = ['Envíados', 'Recibidos', 'Leídos', 'Respondidos']

            fig_funnel = go.Figure(go.Funnel(
                y=stages,
                x=values,
                textinfo="value+percent initial"
            ))

            fig_funnel.update_layout(title='Funnel de Envíos')
            st.plotly_chart(fig_funnel)

            # Código para el gráfico de respuestas por día
            dias_respuestas = []

            for _, row in df_bancoppel.iterrows():
                try:
                    if isinstance(row['messages'], str):
                        messages = json.loads(row['messages'])
                    elif isinstance(row['messages'], list):
                        messages = row['messages']
                    else:
                        continue 
                    
                    for message in messages:
                        if message.get('role') != 'assistant' and message.get('direction') == 'incoming':
                            created_at = message.get('createdAt')
                            if created_at:
                                # Convertimos a datetime y restamos 6 horas
                                adjusted_time = pd.to_datetime(created_at) - pd.Timedelta(hours=6)
                                day_of_week = adjusted_time.day_name()
                                dias_respuestas.append(day_of_week)

                except (json.JSONDecodeError, TypeError) as e:
                    st.error(f"Error al procesar la fila: {e}")

            df_dias_respuestas = pd.DataFrame(dias_respuestas, columns=['Día de la semana'])

            conteo_respuestas = df_dias_respuestas['Día de la semana'].value_counts().reindex(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fill_value=0)

            fig_bar = px.bar(
                conteo_respuestas, 
                x=conteo_respuestas.index, 
                y=conteo_respuestas.values,
                labels={'x': 'Día de la semana', 'y': 'Cantidad de respuestas'},
                title='Cantidad de Respuestas por Día de la Semana'
            )

            st.plotly_chart(fig_bar)

            # Añadir el nuevo código aquí para el mapa de calor de respuestas por hora y día
            horas_dias_respuestas = []
            timezone_cdmx = pytz.timezone('America/Mexico_City')

            for _, row in df_bancoppel.iterrows():
                try:
                    if isinstance(row['messages'], str):
                        messages = json.loads(row['messages'])
                    elif isinstance(row['messages'], list):
                        messages = row['messages']
                    else:
                        continue 
                    
                    for message in messages:
                        if message.get('role') != 'assistant' and message.get('direction') == 'incoming':
                            created_at = message.get('createdAt')
                            if created_at:
                                dt = pd.to_datetime(created_at).tz_convert('UTC')  # Convertimos a UTC
                                dt_cdmx = dt.tz_convert(timezone_cdmx)  # Convertimos de UTC a la zona horaria de CDMX
                                day_of_week = dt_cdmx.strftime('%A')
                                hour_of_day = dt_cdmx.hour
                                horas_dias_respuestas.append((day_of_week, hour_of_day))

                except (json.JSONDecodeError, TypeError) as e:
                    st.error(f"Error al procesar la fila: {e}")

            df_horas_dias_respuestas = pd.DataFrame(horas_dias_respuestas, columns=['Día de la semana', 'Hora del día'])

            df_horas_dias_respuestas['Día de la semana'] = df_horas_dias_respuestas['Día de la semana'].replace({
                'Monday': 'lunes',
                'Tuesday': 'martes',
                'Wednesday': 'miércoles',
                'Thursday': 'jueves',
                'Friday': 'viernes',
                'Saturday': 'sábado',
                'Sunday': 'domingo'
            })

            heatmap_data = df_horas_dias_respuestas.pivot_table(index='Día de la semana', columns='Hora del día', aggfunc='size', fill_value=0)

            heatmap_data = heatmap_data.reindex(["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"])

            fig_heatmap = px.imshow(heatmap_data, labels=dict(x="Hora del día", y="Día de la semana", color="Cantidad de respuestas"),
                                    x=heatmap_data.columns, y=heatmap_data.index,
                                    title="Mapa de Calor de Respuestas por Hora del Día y Día de la Semana")

            st.plotly_chart(fig_heatmap)


######################################################################################################################


def solicitar_datos_page():
    st.markdown("<h1>Solicitar <span style='color: #27A3D7;'>Datos</span> desde <span style='color: #27A3D7;'>S3</span></h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("<h5>Selecciona una opción para hacer los requests de las bases de datos para campañas disponibles:</h5>", unsafe_allow_html=True)

    opcion = st.selectbox(
        'Selecciona la base de datos a cargar:',
        ('Ninguna', 'BanCoppel', 'Monte de Piedad')
    )
    
    if opcion == 'BanCoppel':
        if 'df_bancoppel' not in st.session_state or 'df_bancoppel_conversations' not in st.session_state:
            with st.spinner("Cargando base de datos de BanCoppel desde S3, por favor espera..."):
                df_bancoppel, df_bancoppel_conversations = obteniendo_df_bancoppel()
                
                if df_bancoppel is not None and df_bancoppel_conversations is not None:
                    st.session_state.df_bancoppel = df_bancoppel
                    st.session_state.df_bancoppel_conversations = df_bancoppel_conversations
                    st.success("¡Datos de BanCoppel cargados con éxito, puedes ver el dashboard ahora!")
                else:
                    st.session_state.df_bancoppel = None  
                    st.session_state.df_bancoppel_conversations = None
                    st.warning("No se pudo cargar el DataFrame de BanCoppel desde S3.")
        else:
            df_bancoppel = st.session_state.df_bancoppel
            df_bancoppel_conversations = st.session_state.df_bancoppel_conversations
            
            if df_bancoppel is not None and df_bancoppel_conversations is not None:
                st.success("¡Datos de BanCoppel cargados con éxito, puedes ver el dashboard ahora!")
            else:
                st.warning("No se pudo cargar el DataFrame de BanCoppel desde S3.")
    
    if 'df_bancoppel' in st.session_state and st.session_state.df_bancoppel is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h1>Base de Datos de <span style='color: #27A3D7;'>BanCoppel</span></h1>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h1>Base <span style='color: #27A3D7;'>Original</span></h1>", unsafe_allow_html=True)
        st.write(st.session_state.df_bancoppel)
    
    if 'df_bancoppel_conversations' in st.session_state and st.session_state.df_bancoppel_conversations is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h1>Base de <span style='color: #27A3D7;'>Conversaciones</span></h1>", unsafe_allow_html=True)
        st.write(st.session_state.df_bancoppel_conversations)


######################################################################################################################        

    
def main():
    st.sidebar.image("./Varios/Logos/logo.png", width=185)
    
    file_path = './credentials_aws.txt' 
    
    if 'aws_session' not in st.session_state:
        try:
            st.session_state['aws_session'] = create_aws_session(file_path) 
            #st.success('Sesión de AWS creada exitosamente.')
        except FileNotFoundError:
            st.error('No se encontró el archivo de credenciales. Verifica la ruta.')
        except Exception as e:
            st.error(f'Error al crear la sesión de AWS: {e}')
    
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'Inicio'
    
    page = st.sidebar.selectbox("Menu", ["Inicio", "Solicitar Datos", "Dashboard del Chatbot"])
    
    if page == "Inicio":
        inicio_page()
    elif page == "Dashboard del Chatbot":
        if 'authenticated' in st.session_state and st.session_state['authenticated']:
            dashboard_page()
        else:
            st.markdown("<h1>Dashboard del <span style='color: #27A3D7;'>Chatbot</span></h1>", unsafe_allow_html=True)
            st.write("Aquí puedes ver el Dashboard del Chatbot, pero antes...")
            st.error("Por favor, ingresa la contraseña en la página de inicio para acceder al Dashboard.")
    elif page == "Solicitar Datos":
        if 'authenticated' in st.session_state and st.session_state['authenticated']:
            solicitar_datos_page()
        else:
            st.markdown("<h1>Solicitar <span style='color: #27A3D7;'>Datos</span></h1>", unsafe_allow_html=True)
            st.write("Aquí puedes solicitar los datos para ver el Dashboard del Chatbot, pero antes...")
            st.error("Por favor, ingresa la contraseña en la página de inicio para solicitar los datos.")




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