import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

# Configuração da página
st.set_page_config(
    page_title= "MyxoNet",
    page_icon= "🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Conteúdo da guia "Home"
tab1, tab2, tab3 = st.tabs(["Home", "MixoNet", "USB"])

with tab1:
    st.subheader("| A Classe Myxozoa")
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("./images/sera.png", width=638)
        st.caption("""Courtesy W.L. Current
                   Myxobolus/Myxosoma sp.
                   """, unsafe_allow_html=True)
        st.text("")

    with col2:
        st.markdown(""*20)
        intro_text = """
        Os myxozoários são parasitas com ciclos de vida complexos, pertencentes ao filo Cnidaria, como águas-vivas e medusas.
        Com mais de 65 gêneros e 2.200 espécies, a maioria parasita peixes, causando doenças graves e alta mortalidade.
        Myxobolus é o gênero mais conhecido, especialmente a espécie Myxobolus cerebralis, responsável pela \"Doença do rodopio\"
        em salmonídeos e danos à aquicultura e populações de peixes selvagens. Outros gêneros notáveis são Henneguya, Kudoa
        e Ellipsomyxa. Alguns myxozoários já foram relatados em humanos, causando surtos após o consumo de peixe cru infectado
        no Japão. O ciclo de vida envolve hospedeiros intermediários (peixes) e definitivos (anelídeos). Apesar da importância
        zoonótica, esses parasitas não são inspecionados no pescado brasileiro, ao contrário dos Estados Unidos. A abordagem 
        da Saúde Única promove a saúde sustentável de pessoas, animais e ecossistemas, reconhecendo sua interdependência e
        envolvendo vários setores para enfrentar ameaças à saúde, ecossistemas, segurança alimentar e mudanças climáticas,
        contribuindo para o desenvolvimento sustentável.
        """
        st.write(f'<p style="color:#9c9d9f">{intro_text}</p>', unsafe_allow_html=True)
        audio_file = open("images/p_9841290_826.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mpeg")

        st.subheader("| Seu ciclo de vida")
        st.write('<p style="color:#9c9d9f">Seu ciclo de vida é indireto, envolvendo hospedeiros intermediários (peixes) e definitivos (anelídeos)</p>',
                unsafe_allow_html=True,)
        st.subheader("| Saúde Única")
        st.write('<p style="color:#9c9d9f">A abordagem da Saúde Única promove uma visão integrada e multissetorial da saúde, envolvendo humanos, animais e ecossistemas. Reconhece a interdependência desses elementos e mobiliza diversos setores e disciplinas para promover o bem-estar e lidar com ameaças à saúde e aos ecossistemas, incluindo água limpa, segurança alimentar, mudanças climáticas e desenvolvimento sustentável.</p>',
                unsafe_allow_html=True,)

# Conteúdo da guia "MixoNet"
with tab2:
    st.subheader("MyxoNet")
    st.write('Por favor, carregue a imagem para obter a identificação')

    with st.spinner('Por favor, aguarde enquanto analisamos a sua imagem'):
        yolo = YOLO_Pred(onnx_model='./best.onnx', data_yaml='./data.yaml')

    object = upload_image()

    if object:
        image_obj = Image.open(object['file'])
        st.info('Pré-visualização da imagem')
        st.image(image_obj)

        st.subheader('Confira abaixo os detalhes do arquivo')
        st.json(object['details'])

        button = st.button('Descubra qual o Myxozoário pode estar presente em sua imagem')
        if button:
            predict_image(yolo, image_obj)

# Conteúdo da guia "USB"
with tab3:
    st.subheader("USB")

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        pred_img = yolo.predictions(img)
        return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

    webrtc_streamer(key="example", 
                    video_frame_callback=video_frame_callback,
                    media_stream_constraints={"video":True,"audio":False})
