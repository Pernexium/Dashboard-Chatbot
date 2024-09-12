import io
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
                temp_item[it]['product'] = item['product'] if 'product' in item.keys() else 'sin_producto'
            
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
    st.markdown("<h1 style='font-size: 26px; color: black; text-align: left;'>FILTROS</h1>",unsafe_allow_html=True)   
    
    df_conversations['createdAt'] = pd.to_datetime(df_conversations['createdAt'])
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')

    fecha_minima = pd.to_datetime(df_conversations['createdAt'].min())
    fecha_maxima = pd.to_datetime(df_conversations['createdAt'].max())

    primer_dia_mes_actual = pd.Timestamp.today().replace(day=1)

    rango_fechas = st.date_input(
        "FECHA", 
        value=(primer_dia_mes_actual, fecha_maxima), 
        min_value=fecha_minima, 
        max_value=fecha_maxima
    )
    
    productos_unicos = list(set(df['product'].unique()).union(set(df_conversations['product'].unique())))

    productos_seleccionados = st.multiselect(
        "PRODUCTOS", 
        options=productos_unicos,
        default=productos_unicos 
    )
    
    if not productos_seleccionados:
        st.error("Por favor, selecciona al menos un producto.")
        return

    if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
        fecha_inicio = rango_fechas[0]
        fecha_fin = rango_fechas[1]
        
        if fecha_inicio > fecha_fin:
            st.error("La fecha de inicio no puede ser mayor que la fecha de fin")
            return
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<h1>Dashboard del <span style='color: #145CB3;'>Chatbot</span> de <span style='color: #145CB3;'>{nombre}</span></h1>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        df_conversations_filtered = df_conversations[
            (df_conversations['createdAt'] >= pd.to_datetime(fecha_inicio)) &
            (df_conversations['createdAt'] <= pd.to_datetime(fecha_fin)) &
            (df_conversations['product'].isin(productos_seleccionados))
        ]

        df_filtered = df[
            (df['created_at'] >= pd.to_datetime(fecha_inicio).tz_localize('UTC')) &
            (df['created_at'] <= pd.to_datetime(fecha_fin).tz_localize('UTC')) &
            (df['product'].isin(productos_seleccionados))
        ]
        
        respuestas = 0
        indexes_templates = np.where(df_conversations_filtered.content.str.contains("Template"))[0]
        for index in range(len(indexes_templates) - 1):
            if (indexes_templates[index + 1] - indexes_templates[index]) > 2:
                respuestas += 1
        
        ###################################################### ENVIOS TOTALES ######################################################   
          
        #envios_totales = df_conversations_filtered.query("content.str.contains('Template')").role.value_counts().sum()
        filtered_data_2 = df_conversations_filtered.query("content.str.contains('Template')")
        read_count_2 = filtered_data_2['sent_status_read'].sum() 
        delivered_count_2 = filtered_data_2['sent_status_delivered'].sum() + read_count_2
        envios_totales_2 = filtered_data_2['sent_status_sent'].sum() + read_count_2 + delivered_count_2
        envios_totales_formateado = "{:,}".format(envios_totales_2)

        ###################################################### PORCENTAJE DE RESPUESTAS ###################################################### 
        total_envios = envios_totales_2
        total_respuestas = df_conversations_filtered.query("direction == 'incoming'").shape[0]

        if total_envios > 0:
            porcentaje_respuestas = round((total_respuestas / total_envios) * 100, 2)
        else:
            porcentaje_respuestas = 0

        porcentaje_respuestas_str = f"{porcentaje_respuestas}%"

        ###################################################### GASTOS DE META ###################################################### 
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

        total_envios = df_conversations_filtered.query("content.str.contains('Template') and sent_status_failed == False").shape[0]

        tipo_cambio_usd_mxn = obtener_tipo_cambio()
        costo_usd_por_mensaje = 0.0436
        costo_mxn_por_mensaje = costo_usd_por_mensaje * tipo_cambio_usd_mxn
        total_a_pagar_mxn = total_envios * costo_mxn_por_mensaje

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                "<div style='text-align: center; color: #145CB3; font-size: 25px; font-weight: bold;'>Envíos Totales</div>",
                unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 33px;'>{envios_totales_formateado}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(
                "<div style='text-align: center; color: #145CB3; font-size: 24px; font-weight: bold;'>% Respuestas</div>",
                unsafe_allow_html=True
            )
            st.markdown(f"<div style='text-align: center; font-size: 33px;'>{porcentaje_respuestas_str}</div>", unsafe_allow_html=True)

        with col3:
            st.markdown(
                "<div style='text-align: center; color: #145CB3; font-size: 24px; font-weight: bold;'>Gastos Meta</div>",
                unsafe_allow_html=True
            )
            st.markdown(f"<div style='text-align: center; font-size: 33px;'>${total_a_pagar_mxn:,.0f} MXN</div>", unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)

        ###################################################### DICTAMINACIONES GENERAL ###################################################### 

        color_palette = ['#145CB3', '#1C6AD1', '#2682EF', '#3A98F7', '#5BB4FC', '#7DD0FF', '#A0E6FF']

        tag_counts = df_conversations_filtered.query("content.str.contains('Template')").tag.value_counts()

        fig = px.pie(
            values=tag_counts.values,
            names=tag_counts.index,
            title='Dictaminaciones General',
            color_discrete_sequence=color_palette,
            width=1300,
            height=650,
            hole=0.40
        )
        
        fig.update_traces(
            textinfo='percent', 
            textfont_size=15, 
            textfont_color='white', 
            hovertemplate='%{label}: %{value:,.0f}'
        )

        fig.update_layout(
            title={
                'text': '<b>DICTAMINACIONES GENERAL</b><br><span style="font-size: 14px;">'f'{fecha_inicio.strftime("%d/%m/%Y")} - {fecha_fin.strftime("%d/%m/%Y")}</span>',
                'font': {'size': 29, 'color': 'black'}
            },
            legend={
                'font': {'size': 16},
                'orientation': 'h', 
                'yanchor': 'top', 
                'y': -0.25, 
                'xanchor': 'center', 
                'x': 0.5, 
                'title_text': '',
                'itemwidth': 100 
            },
            legend_tracegroupgap=5 
        )
        
        st.plotly_chart(fig)
        st.markdown("<hr>", unsafe_allow_html=True)
        
        ###################################################### FUNEL DE ENVIOS ###################################################### 
        filtered_data = df_conversations_filtered.query("content.str.contains('Template')")

        response_count = total_respuestas
        read_count = filtered_data['sent_status_read'].sum() 
        delivered_count = filtered_data['sent_status_delivered'].sum() + read_count
        sent_count = filtered_data['sent_status_sent'].sum() + read_count + delivered_count

        values = [sent_count, delivered_count, read_count, response_count]
        stages = ['<b>ENVÍADOS</b>', '<b>RECIBIDOS</b>', '<b>LEÍDOS</b>', '<b>RESPONDIDOS</b>']

        colors = ['#145CB3', '#1C6AD1', '#2682EF', '#5BB4FC']

        fig = go.Figure(go.Funnel(
            y=stages,
            x=values,
            marker=dict(color=colors),
            texttemplate='%{percentInitial:.1%}<br><span style="font-size:14px">%{value:,.0f}</span>',
            textfont=dict(
                size=18, 
                color='white'
            )
        ))


        fig.update_layout(
            title={
                'text': '<b>FUNEL DE ENVÍOS</b><br><span style="font-size: 14px;">'f'{fecha_inicio.strftime("%d/%m/%Y")} - {fecha_fin.strftime("%d/%m/%Y")}</span>',
                'font': {'size': 29, 'color': 'black'}
            },
            width=1300,
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',  
            plot_bgcolor='rgba(0,0,0,0)',  
            yaxis=dict(
                title_font=dict(size=17),   
                tickfont=dict(size=17)       
            )
        )

        st.plotly_chart(fig)
        st.markdown("<hr>", unsafe_allow_html=True)

        ###################################################### CANTIDAD DE RESPUESTAS ###################################################### 
        df_filtered['created_at'] = pd.to_datetime(df_filtered['created_at']).dt.tz_convert('UTC')
        dias_respuestas = []
        for _, row in df_filtered.iterrows():
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
            width=1300, 
            height=550,
            text=conteo_respuestas.values
        )

        fig.update_traces(
            marker_color='#145CB3',  
            marker_line_color='#145CB3',  
            marker_line_width=1.5, 
            textposition='outside',  
            textfont=dict(
                size=14  
            )
        )

        fig.update_layout(
            title={
                'text': '<b>CANTIDAD DE RESPUESTAS</b><br><span style="font-size: 14px;">'f'{fecha_inicio.strftime("%d/%m/%Y")} - {fecha_fin.strftime("%d/%m/%Y")}</span>',
                'font': {
                    'size': 29
                }
            },
            plot_bgcolor='rgba(0, 0, 0, 0)',  
            paper_bgcolor='rgba(0, 0, 0, 0)', 
            xaxis=dict(
                linecolor='gray',  
                gridcolor='rgba(0, 0, 0, 0)',
                tickfont=dict(
                    size=15 
                ),
                titlefont=dict(
                    size=18 
                )
            ),
            yaxis=dict(
                linecolor='gray',  
                gridcolor='gray',
                tickfont=dict(
                    size=15 
                ),
                titlefont=dict(
                    size=18 
                )
            )
        )

        st.plotly_chart(fig)
        st.markdown("<hr>", unsafe_allow_html=True)
        
        ###################################################### MAPA DE CALOR DE RESPUESTAS ###################################################### 
        horas_dias_respuestas = []
        timezone_cdmx = pytz.timezone('America/Mexico_City')

        for _, row in df_filtered.iterrows():
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
            width=1300,
            height=600,
            color_continuous_scale=[(0, '#145CB3'), (1, 'white')]
        )

        fig.update_layout(
            title={
                'text': '<b>MAPA DE CALOR DE RESPUESTAS</b><br><span style="font-size: 14px;">'
                        f'{fecha_inicio.strftime("%d/%m/%Y")} - {fecha_fin.strftime("%d/%m/%Y")}</span>',
                'font': {
                    'size': 29
                }
            },
            xaxis=dict(
                title_font=dict(size=18),
                tickfont=dict(size=15)
            ),
            yaxis=dict(
                title_font=dict(size=18),
                tickfont=dict(size=15)
            ),
            coloraxis_colorbar=dict(
                title="Cantidad de Respuestas",
                titleside="right",  
                titlefont=dict(size=15),
                tickfont=dict(size=12),
                thickness=15,
                len=0.75
            )
        )

        st.plotly_chart(fig)
        st.markdown("<hr>", unsafe_allow_html=True)


        ###################################################### MAPA DE MEXICO ###################################################### 
        st.markdown("<h1 style='font-size: 29px; color: black; text-align: center;'>MAPA DE MÉXICO</h1>", unsafe_allow_html = True)
        with open('./mexicoHigh.json', encoding = 'utf-8') as f:
            mexico_geojson = json.load(f)

        data = {
            'Estado': ['Aguascalientes', 'Baja California', 'Baja California Sur', 'Campeche', 'Chiapas', 'Chihuahua', 
                    'Ciudad de México', 'Coahuila', 'Colima', 'Durango', 'Guanajuato', 'Guerrero', 'Hidalgo', 
                    'Jalisco', 'México', 'Michoacán', 'Morelos', 'Nayarit', 'Nuevo León', 'Oaxaca', 
                    'Puebla', 'Querétaro', 'Quintana Roo', 'San Luis Potosí', 'Sinaloa', 'Sonora', 'Tabasco', 
                    'Tamaulipas', 'Tlaxcala', 'Veracruz', 'Yucatán', 'Zacatecas'],
            'poblacion': [1434634, 3769020, 798447, 928363, 5543828, 3801487, 9209944, 3146771, 731391, 1832650, 6213799, 
                        3533251, 3082841, 8348481, 17427790, 4825401, 1971520, 1284977, 5611512, 4132148, 6583278, 
                        2279630, 1857985, 2822235, 3026943, 2944845, 2395272, 3650605, 1342977, 8112505, 2320894, 1622138]
        }

        df = pd.DataFrame(data)
        
        fig = px.choropleth(df, 
                            geojson = mexico_geojson, 
                            locations = 'Estado', 
                            featureidkey = "properties.name", 
                            color = 'poblacion', 
                            color_continuous_scale = [(0, '#145CB3'), (1, 'white')],
                            labels = {'poblacion':'Población'},
                            title = 'Población por Estado en México')

        fig.update_geos(fitbounds = "locations", visible = False)
        fig.update_layout(margin = {"r":0,"t":0,"l":0,"b":0}, width = 1200, height = 700)
        st.plotly_chart(fig, use_container_width = True)
        st.markdown("<hr>", unsafe_allow_html = True)

        ###################################################### TABLA DE FILTROS ###################################################### 
        st.markdown("<h1 style='font-size: 26px; color: black; text-align: left;'>FILTROS PARA LA TABLA DE FILTROS</h1>",unsafe_allow_html=True) 

        fecha_actual = datetime.now()
        anio_actual = fecha_actual.strftime("%Y")
        mes_actual = fecha_actual.strftime("%m")

        session = create_aws_session()
        s3_client = session.client('s3')
        bucket_name = 's3-pernexium-report'

        if nombre == "BanCoppel":
            prefix = f'master/bancoppel/reportes/chatbot/{anio_actual}_{mes_actual}/'
        elif nombre == "Monte de Piedad":
            prefix = f'master/monte/reportes/chatbot/{anio_actual}_{mes_actual}/'


        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        if 'Contents' in response:
            xlsx_files = [
                obj for obj in response['Contents'] if obj['Key'].endswith('.xlsx')
            ]
            xlsx_files = sorted(xlsx_files, key=lambda x: x['LastModified'], reverse=True)
            latest_file = xlsx_files[0]['Key']
            file_object = s3_client.get_object(Bucket=bucket_name, Key=latest_file)
            file_content = file_object['Body'].read()
            df_filtros = pd.read_excel(BytesIO(file_content))
            columnas_a_eliminar = ['segmento','ultima_respuesta_cliente_trim', 'detonaciones_enviadas_trim','detonaciones_vistas_trim', 'detonaciones_entregadas_trim','detonaciones_fallidas_trim', 
                                   'detonaciones_aceptadas_por_meta_trim','detonaciones_sin_estatus_trim', 'total_respuestas_trim','eficiencia_respuestas_trim', 'primera_detonacion_trim',
                                   'ultima_detonacion_trim', 'ha_respondido_trim', 'status_trim','chatbot_ofrece_carta_convenio_trim', 'trim 7 a 8', 'trim 8 a 9',
                                   'trim 9 a 10', 'trim 10 a 11', 'trim 11 a 12', 'trim 12 a 13', 'trim 13 a 14', 'trim 14 a 15', 'trim 15 a 16', 'trim 16 a 17',
                                   'trim 17 a 18', 'trim 18 a 19', 'trim 19 a 20', 'trim 20 a 21','trim 21 a 22', 'ultima_hora_respuesta_cliente_trim',    'chatbot_ofrece_carta_convenio_periodo', 'periodo 7 a 8', 'periodo 8 a 9',
                                   'periodo 9 a 10', 'periodo 10 a 11', 'periodo 11 a 12', 'periodo 12 a 13',
                                   'periodo 13 a 14', 'periodo 14 a 15', 'periodo 15 a 16', 'periodo 16 a 17',
                                   'periodo 17 a 18', 'periodo 18 a 19', 'periodo 19 a 20', 'periodo 20 a 21',
                                   'periodo 21 a 22', 'ultima_hora_respuesta_cliente_periodo']
            df_filtros.drop(columnas_a_eliminar, axis=1, inplace=True)
        

        columnas_filtrables = [
            'credito', 
            'detonaciones_enviadas_periodo', 
            'detonaciones_vistas_periodo', 
            'total_respuestas_periodo', 
            'ultima_detonacion_periodo', 
            'status_periodo'
        ]
        col1, col2 = st.columns(2)

        df_filtros_filtrado = df_filtros.copy()

        if 'ultima_detonacion_periodo' in df_filtros_filtrado.columns:
            df_filtros_filtrado['ultima_detonacion_periodo'] = pd.to_datetime(df_filtros_filtrado['ultima_detonacion_periodo']).dt.date
        
        if 'primera_detonacion_periodo' in df_filtros_filtrado.columns:
            df_filtros_filtrado['primera_detonacion_periodo'] = pd.to_datetime(df_filtros_filtrado['primera_detonacion_periodo']).dt.date
        
        if 'ultima_respuesta_cliente_periodo' in df_filtros_filtrado.columns:
            df_filtros_filtrado['ultima_respuesta_cliente_periodo'] = pd.to_datetime(df_filtros_filtrado['ultima_respuesta_cliente_periodo']).dt.date

        mitad = len(columnas_filtrables) // 2
        columnas_primera_mitad = columnas_filtrables[:mitad]
        columnas_segunda_mitad = columnas_filtrables[mitad:]

        with col1:
            for columna in columnas_primera_mitad:
                opciones = df_filtros[columna].unique()
                seleccion = st.multiselect(f'Selecciona valores para la columna "{columna}":', opciones)
                
                if seleccion:
                    df_filtros_filtrado = df_filtros_filtrado[df_filtros_filtrado[columna].isin(seleccion)]

        with col2:
            for columna in columnas_segunda_mitad:
                opciones = df_filtros[columna].unique()
                seleccion = st.multiselect(f'Selecciona valores para la columna "{columna}":', opciones)
                
                if seleccion:
                    df_filtros_filtrado = df_filtros_filtrado[df_filtros_filtrado[columna].isin(seleccion)]

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h1 style='font-size: 35px; color: #145CB3; text-align: center;'>TABLA DE FILTROS</h1>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.dataframe(df_filtros_filtrado.head(100))


        def descargar_excel(df):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Datos Filtrados')
            processed_data = output.getvalue()
            return processed_data

        if not df_filtros_filtrado.empty:
            #st.write("Esta tabla tiene sus propios filtros, no son afectados por los filtros generales.")

            excel_data = descargar_excel(df_filtros_filtrado)

            st.download_button(
                label=f"Descargar Tabla de Filtros de {nombre} en Excel",
                data=excel_data,
                file_name=f"Tabla_de_Filtros_{nombre}_Dashboard_Chatbot.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.write(f"No se encontraron archivos en el bucket con el prefijo {prefix} especificado.")

    else:
        st.error("Por favor, selecciona una fecha de inicio y una fecha de fin válidas.")
        



    
######################################################################################################################    


def dashboard_page():
    if 'df_bancoppel' in st.session_state and st.session_state.df_bancoppel is not None:
        graficas(st.session_state.df_bancoppel, st.session_state.df_bancoppel_conversations, "BanCoppel")
    elif 'df_monte' in st.session_state and st.session_state.df_monte is not None:
        graficas(st.session_state.df_monte, st.session_state.df_monte_conversations, "Monte de Piedad")


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
    image_path = "./Varios/Logos/PXM Imagotipo 2.png"
    
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    st.markdown(f"""<img src="data:image/png;base64,{encoded_image}" width="45%">""", unsafe_allow_html=True)
    st.markdown("<h1>Dashboard del <span style='color: #145CB3;'>Chatbot</span></h1>", unsafe_allow_html=True)
    
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
                    
                    cargar_base_datos(roles[key])
                    
                    dashboard_page()
                    return
                    
            if not st.session_state['authenticated']:
                st.warning("Contraseña incorrecta. Inténtalo de nuevo.")
    
    if st.session_state['authenticated']:
        dashboard_page()




    # st.sidebar.markdown("---")

    # st.sidebar.markdown("### Contacto")
    # st.sidebar.markdown(
    #  """
    #  @Pernexium<br>
    #  <div style="display:flex; align-items:center; gap:10px;">
    #      <a href="https://www.pernexium.com/" style="display:flex; align-items:center; margin-right:15px; margin-left:85px;">
    #         <div style="background-color:white; width:30px; height:30px; display:flex; justify-content:center; align-items:center; border-radius:50%;">
    #             <img src="https://www.svgrepo.com/show/438256/world-round.svg" alt="Web Page" style="width:45px; height:45px;">
    #         </div>
    #      </a>
    #      <a href="https://www.instagram.com/pernexium/" style="display:flex; align-items:center; margin-right:15px;">
    #         <div style="background-color:white; width:30px; height:30px; display:flex; justify-content:center; align-items:center; border-radius:50%;">
    #              <img src="https://www.svgrepo.com/show/494277/instagram-round.svg" alt="Instagram" style="width:40px; height:40px;">
    #         </div>
    #      </a>
    #     <a href="https://www.linkedin.com/company/pernexium/" style="display:flex; align-items:center;">
    #         <div style="background-color:white; width:30px; height:30px; display:flex; justify-content:center; align-items:center; border-radius:50%;">
    #             <img src="https://www.svgrepo.com/show/494278/linkedin-round.svg" alt="LinkedIn" style="width:40px; height:40px;">
    #         </div>
    #     </a>

    #  </div>
    #  """,
    #  unsafe_allow_html=True)

    # st.sidebar.markdown("---")
    # st.sidebar.markdown(
    #     '<p style="color: grey;">© 2024 Pernexium.</p>', 
    #     unsafe_allow_html=True)

if __name__ == "__main__":
    main()
