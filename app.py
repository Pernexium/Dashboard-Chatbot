import toml
import pytz
import json
import boto3
import base64
import random
import hashlib
import requests
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


def create_aws_session():
    aws_access_key_id = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]
    aws_region = st.secrets["aws"]["AWS_REGION"]
    
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    return session

######################################################################################################################


def obteniendo_df_general(bucket_name, prefix):
    """
    Función para conectar a S3 y obtener el DataFrame de cualquier origen especificado.

    Parámetros:
    - bucket_name: Nombre del bucket en S3.
    - prefix: Prefijo para buscar los archivos en el bucket.
    
    Retorna:
    - df_general: DataFrame con los datos originales.
    - df_conversations: DataFrame con las conversaciones procesadas.
    """
    session = create_aws_session()
    s3_client = session.client('s3')

    current_date = datetime.now().strftime('%Y_%m')
    full_prefix = f'{prefix}/{current_date}/'
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=full_prefix)
    files = response.get('Contents', [])
    
    if files:
        files_sorted = sorted(files, key=lambda x: x['LastModified'], reverse=True)
        latest_file = files_sorted[0]['Key']
        response = s3_client.get_object(Bucket=bucket_name, Key=latest_file)
        content = response['Body'].read()
        json_data = json.loads(content)
        
        if isinstance(json_data, list):
            df_general = pd.DataFrame(json_data)
        else:
            df_general = pd.json_normalize(json_data)
        
        conversations = []
        for index, item in tqdm(df_general.iterrows(), total=df_general.shape[0]):
            temp_item = []
            for k in item['messages']:
                if 'content' not in k.keys():
                    continue
                if "Template" in k['content']:
                    k['content'] = " " + k['content']  
                temp_item.append(k)
            
            tag = item.get('tag', "sin_info_para_clasificar")
            rand_conversationid = random.randint(1e6, 1e8)
            
            for it in range(len(temp_item)):
                temp_item[it]['conversationId'] = rand_conversationid
                temp_item[it]['credit'] = item['credit']
                temp_item[it]['discount'] = item['discount']
                temp_item[it]['phone1'] = item['contact_phone']
                temp_item[it]['last_template_at'] = item['last_template_at']
                temp_item[it]['tag'] = tag
                temp_item[it]['send_status'] = item.get('status', 'sin_estatus')
            
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
        conversations['tag'] = conversations['tag'].replace('no defineno defineno defineno defineno defineno defineno defineno defineno defineno define', 'no define')
        conversations['tag'] = conversations['tag'].replace('no defineno defineno defineno defineno defineno defineno defineno definecontacto humano', 'no define')
        conversations['tag'] = conversations['tag'].replace('no defineno defineno defineno defineno defineno defineno defineno defineno define', 'no define')
        conversations['tag'] = conversations['tag'].replace('no defineno defineno definenegativa de pagono defineno defineno defineno definecontacto humano', 'no define')
        conversations['tag'] = conversations['tag'].replace('no defineno defineno defineno defineno defineno defineno definedetener promocionesno define', 'no define')
        conversations['tag'] = conversations['tag'].replace('contacto humanono defineno defineno defineno defineno defineno definecarta conveniocontacto humano', 'no define')
        conversations['tag'] = conversations['tag'].replace('contacto humanodetener promocionescontacto humanono definecontacto humanono definecarta conveniodetener', 'no define')


        data = conversations.copy()
        serie_count_celphones = pd.DataFrame(data.groupby("conversationId").phone1.nunique().sort_values()).reset_index()
        data = data.loc[data.conversationId.isin(serie_count_celphones.query("phone1 == 1").conversationId)]
        data.dropna(subset=["phone1", 'credit'], inplace=True)
        data['id'] = data.apply(lambda row: f"{row['credit']} - {row['phone1']}", axis=1)
        data.credit = data.credit.astype(str)
        data.conversationId = data.conversationId.astype(str)
        data.reset_index(drop=True, inplace=True)
        data = data.sort_values(by=['createdAt', 'content'])
        data = data.reset_index()
        for i, row in data.iterrows():
            if "Template" in row['content']:
                if i + 1 < len(data):
                    data.at[i, "sent_status"] = data.at[i+1, "sent_status"]
        dummies = pd.get_dummies(data['sent_status'], prefix='sent_status')
        data = pd.concat([data.drop('sent_status', axis=1), dummies], axis=1)

        df_conversations = data

        return df_general, df_conversations
    else:
        print("No se encontraron archivos para la fecha actual.")
        return None, None
    

######################################################################################################################


def load_secrets():
    """Carga los secretos desde el archivo secrets.toml."""
    secrets = toml.load(".streamlit/secrets.toml")
    return secrets["passwords"], secrets["roles"]


######################################################################################################################


def hash_password(password):
    """Devuelve el hash SHA-256 de la contraseña proporcionada."""
    return hashlib.sha256(password.encode()).hexdigest()


######################################################################################################################


def graficas(df, df_conversations, nombre):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h1>Dashboard del <span style='color: #27A3D7;'>Chatbot</span> de <span style='color: #27A3D7;'>{nombre}</span></h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    respuestas = 0
    indexes_templates = np.where(df_conversations.content.str.contains("Template"))[0]
    for index in range(len(indexes_templates) - 1):
        if (indexes_templates[index + 1] - indexes_templates[index]) > 2:
            respuestas += 1

    ########## ENVIOS TOTALES ##########
    envios_totales = df_conversations.query("content.str.contains('Template')").role.value_counts().sum()
    envios_totales_formateado = "{:,}".format(envios_totales)

    ########## PORCENTAJE DE RESPUESTAS ##########
    total_envios = df_conversations.query("content.str.contains('Template')").shape[0]
    total_respuestas = df_conversations.query("direction == 'incoming'").shape[0]

    if total_envios > 0: 
        porcentaje_respuestas = round((total_respuestas / total_envios) * 100, 2)
    else:
        porcentaje_respuestas = 0

    porcentaje_respuestas_str = f"{porcentaje_respuestas}%"

    ########## GASTOS DE META ##########
    def obtener_tipo_cambio():
        try:
            response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
            response.raise_for_status()  
            data = response.json()
            tipo_cambio_usd_mxn = data["rates"]["MXN"]
            return tipo_cambio_usd_mxn
        except requests.exceptions.RequestException as e:
            print(f"Error al obtener el tipo de cambio: {e}")
            return 19.33

    def calcular_costo_por_mes(data):
        fecha_actual = datetime.now()
        mes = fecha_actual.month
        año = fecha_actual.year
        data['createdAt'] = pd.to_datetime(data['createdAt'])
        mensajes_template = data[data['content'].str.contains('Template', na=False) & (data['sent_status_failed'] != True)]
        mensajes_template_mes = mensajes_template[(mensajes_template['createdAt'].dt.month == mes) &
                                                (mensajes_template['createdAt'].dt.year == año)]
        total_envios_mes = mensajes_template_mes.shape[0]
        tipo_cambio_usd_mxn = obtener_tipo_cambio()
        costo_usd_por_mensaje = 0.0436
        costo_mxn_por_mensaje = costo_usd_por_mensaje * tipo_cambio_usd_mxn
        total_a_pagar_mxn = total_envios_mes * costo_mxn_por_mensaje
        total_envios_mes_formateado = "{:,}".format(total_envios_mes)
        
        return total_envios_mes_formateado, total_a_pagar_mxn

    total_envios_mes_formateado, total_a_pagar_mxn = calcular_costo_por_mes(df_conversations)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            "<div style='text-align: center; color: #27A3D7; font-size: 25px; font-weight: bold;'>Envíos Totales</div>",
            unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size: 33px;'>{envios_totales_formateado}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            "<div style='text-align: center; color: #27A3D7; font-size: 24px; font-weight: bold;'>% Respuestas</div>",
            unsafe_allow_html=True
        )
        st.markdown(f"<div style='text-align: center; font-size: 33px;'>{porcentaje_respuestas_str}</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(
            "<div style='text-align: center; color: #27A3D7; font-size: 24px; font-weight: bold;'>Envíos del Mes</div>",
            unsafe_allow_html=True
        )
        st.markdown(f"<div style='text-align: center; font-size: 33px;'>{total_envios_mes_formateado}</div>", unsafe_allow_html=True)

    with col4:
        st.markdown(
            "<div style='text-align: center; color: #27A3D7; font-size: 24px; font-weight: bold;'>Gastos Meta</div>",
            unsafe_allow_html=True
        )
        st.markdown(f"<div style='text-align: center; font-size: 33px;'>${total_a_pagar_mxn:,.0f} MXN</div>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    ########## DICTAMINACIONES GENERAL ##########
    tag_counts = df_conversations.query("content.str.contains('Template')").tag.value_counts()

    fig = px.pie(
        values=tag_counts.values,
        names=tag_counts.index,
        title='Dictaminaciones General',
        width=850,
        height=500,
        hole=0.45
    )

    fig.update_layout(
        title={
            'text': '<b>DICTAMINACIONES GENERAL</b>',
            'font': {'size': 24, 'color': 'white'} 
        },
        legend={
            'font': {'size': 14}
        }
    )

    st.plotly_chart(fig)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    ########## FUNEL DE ENVIOS ##########
    filtered_data = df_conversations.query("content.str.contains('Template')")

    response_count = total_respuestas
    read_count = filtered_data['sent_status_read'].sum() 
    delivered_count = filtered_data['sent_status_delivered'].sum() + read_count
    sent_count = filtered_data['sent_status_sent'].sum() + read_count + delivered_count

    values = [sent_count, delivered_count, read_count, response_count]
    stages = ['<b>ENVÍADOS</b>', '<b>RECIBIDOS</b>', '<b>LEÍDOS</b>', '<b>RESPONDIDOS</b>']

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']  

    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textinfo="value+percent initial",
        marker=dict(color=colors) 
    ))

    fig.update_layout(
        title={
            'text': '<b>FUNEL DE ENVÍOS</b>',
            'font': {'size': 24, 'color': 'white'}
        },
        width=1000,
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',  
        plot_bgcolor='rgba(0,0,0,0)',  
        yaxis=dict(
            title_font=dict(size=15),   
            tickfont=dict(size=15)       
        )
    )

    st.plotly_chart(fig)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    ########## TABLA DE FILTROS ##########
    st.markdown("<h3>TABLA DE FILTROS</h3>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    ########## CANTIDAD DE RESPUESTAS ##########
    dias_respuestas = []
    for _, row in df.iterrows():
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
                        adjusted_time = pd.to_datetime(created_at) - pd.Timedelta(hours=6)
                        day_of_week = adjusted_time.day_name()
                        dias_respuestas.append(day_of_week)

        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error al procesar la fila: {e}")

    dias_semana_espanol = {
        "Monday": "Lunes",
        "Tuesday": "Martes",
        "Wednesday": "Miércoles",
        "Thursday": "Jueves",
        "Friday": "Viernes",
        "Saturday": "Sábado",
        "Sunday": "Domingo"
    }

    dias_respuestas_espanol = [dias_semana_espanol[day] for day in dias_respuestas]

    df_dias_respuestas = pd.DataFrame(dias_respuestas_espanol, columns=['Día de la Semana'])

    conteo_respuestas = df_dias_respuestas['Día de la Semana'].value_counts().reindex(
        ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"], fill_value=0)

    fig = px.bar(
        conteo_respuestas, 
        x=conteo_respuestas.index, 
        y=conteo_respuestas.values,
        labels={'x': 'Día de la Semana', 'y': 'Cantidad de Respuestas'},
        title='<b>CANTIDAD DE RESPUESTAS</b>',
        width=950, 
        height=550,
        text=conteo_respuestas.values
    )

    fig.update_traces(
        marker_color='#27A3D7',  
        marker_line_color='#27A3D7',  
        marker_line_width=1.5, 
        textposition='outside' 
    )

    fig.update_layout(
        title={
            'text': '<b>CANTIDAD DE RESPUESTAS</b>',
            'font': {
                'size': 24 
            }
        },
        plot_bgcolor='rgba(0, 0, 0, 0)',  
        paper_bgcolor='rgba(0, 0, 0, 0)', 
        xaxis=dict(
            linecolor='gray',  
            gridcolor='rgba(0, 0, 0, 0)'   
        ),
        yaxis=dict(
            linecolor='gray',  
            gridcolor='gray'   
        )
    )

    st.plotly_chart(fig)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    ########## MAPA DE CALOR DE RESPUESTAS ##########
    horas_dias_respuestas = []
    timezone_cdmx = pytz.timezone('America/Mexico_City')

    for _, row in df.iterrows():
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
                        dt = pd.to_datetime(created_at).tz_convert('UTC') 
                        dt_cdmx = dt.tz_convert(timezone_cdmx)  
                        day_of_week = dt_cdmx.strftime('%A')
                        hour_of_day = dt_cdmx.hour
                        horas_dias_respuestas.append((day_of_week, hour_of_day))

        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error al procesar la fila: {e}")

    df_horas_dias_respuestas = pd.DataFrame(horas_dias_respuestas, columns=['Día de la Semana', 'Hora del Día'])

    df_horas_dias_respuestas['Día de la Semana'] = df_horas_dias_respuestas['Día de la Semana'].replace({
        'Monday': 'Lunes',
        'Tuesday': 'Martes',
        'Wednesday': 'Miércoles',
        'Thursday': 'Jueves',
        'Friday': 'Viernes',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    })

    heatmap_data = df_horas_dias_respuestas.pivot_table(index='Día de la Semana', columns='Hora del Día', aggfunc='size', fill_value=0)

    heatmap_data = heatmap_data.reindex(["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Hora del Día", y="Día de la Semana", color="Cantidad de Respuestas"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        title="<b>MAPA DE CALOR DE RESPUESTAS</b>",
        width=1000,
        height=550
    )

    fig.update_layout(
        title={
            'text': "<b>MAPA DE CALOR DE RESPUESTAS</b>",
            'font': {
                'size': 24 
            }
        }
    )

    st.plotly_chart(fig)
    st.markdown("<hr>", unsafe_allow_html=True)

    
######################################################################################################################    


def dashboard_page():
    if 'df_bancoppel' in st.session_state and st.session_state.df_bancoppel is not None:
        if 'df_bancoppel_conversations' not in st.session_state or st.session_state.df_bancoppel_conversations is None:
            st.error("Por favor, solicita los datos de conversaciones de Bancoppel primero en la página de 'Solicitar Datos' antes de acceder al Dashboard.")
        else:
            graficas(st.session_state.df_bancoppel, st.session_state.df_bancoppel_conversations, "BanCoppel")
    
    elif 'df_monte' in st.session_state and st.session_state.df_monte is not None:
        if 'df_monte_conversations' not in st.session_state or st.session_state.df_monte_conversations is None:
            st.error("Por favor, solicita los datos de conversaciones de Monte de Piedad primero en la página de 'Solicitar Datos' antes de acceder al Dashboard.")
        else:
            graficas(st.session_state.df_monte, st.session_state.df_monte_conversations, "Monte de Piedad")
    
    else:
        st.error("Por favor, solicita los datos primero en la página de 'Solicitar Datos' antes de acceder al Dashboard.")


######################################################################################################################        


def cargar_datos():
    user_role = st.session_state['user_role']
    
    if user_role == "Developer":
        cargar_base_datos('BanCoppel')
        cargar_base_datos('Monte de Piedad')
    elif user_role == "BanCoppel":
        cargar_base_datos('BanCoppel')
    elif user_role == "Monte de Piedad":
        cargar_base_datos('Monte de Piedad')

        
 ######################################################################################################################        


def cargar_base_datos(rol):
    bucket_name = 's3-pernexium-report'
    
    if rol == 'BanCoppel':
        prefix = 'raw/bancoppel/replica_dynamo_chatbot'
        session_key = 'df_bancoppel'
        session_conv_key = 'df_bancoppel_conversations'
    elif rol == 'Monte de Piedad':
        prefix = 'raw/monte/replica_dynamo_chatbot'
        session_key = 'df_monte'
        session_conv_key = 'df_monte_conversations'

    if session_key not in st.session_state or session_conv_key not in st.session_state:
        with st.spinner(f"Cargando base de datos de {rol} desde S3, por favor espera..."):
            df_general, df_conversations = obteniendo_df_general(bucket_name=bucket_name, prefix=prefix)
            
            if df_general is not None and df_conversations is not None:
                st.session_state[session_key] = df_general
                st.session_state[session_conv_key] = df_conversations
                st.success(f"¡Datos de {rol} cargados con éxito!")
            else:
                st.session_state[session_key] = None  
                st.session_state[session_conv_key] = None
                st.warning(f"No se pudo cargar el DataFrame de {rol} desde S3.")
                
                
###################################################################################################################### 
   

def main():
    st.sidebar.image("./Varios/Logos/logo.png", width=185)
    image_path = "./Varios/Logos/PXM Imagotipo 3.png"
    
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    st.markdown(f"""<img src="data:image/png;base64,{encoded_image}" width="60%">""", unsafe_allow_html=True)
    st.markdown("<h1>Dashboard del <span style='color: #27A3D7;'>Chatbot</span></h1>", unsafe_allow_html=True)
    
    passwords, roles = load_secrets()
    
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        st.session_state['user_role'] = None

    if not st.session_state['authenticated']:
        password_input = st.text_input("Introduce la contraseña para acceder al Dashboard:", type="password")
        
        if password_input:
            hashed_input = hash_password(password_input)

            for key, stored_hash in passwords.items():
                if hashed_input == stored_hash:
                    st.session_state['authenticated'] = True
                    st.session_state['user_role'] = roles[key]
                    st.success(f"¡Contraseña correcta! ¡Bienvenido al Dashboard del Chatbot de {roles[key]}!")
                    
                    cargar_datos()
                    
                    st.experimental_rerun()
                    return
                    
            st.warning("Contraseña incorrecta. Inténtalo de nuevo.")
    else:
        dashboard_page()


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
