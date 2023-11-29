import streamlit as st

# Configuração da página
st.set_page_config(
    page_title= "MyxoNet",
    page_icon= "🔬",  # Defina o ícone da página como um emoji de tubarão
    layout="wide",  # Defina o layout como "wide" para aproveitar melhor o espaço na tela
    initial_sidebar_state="collapsed"  # Defina a barra lateral como colapsada
)

# Criação das guias
tab1, tab2, tab3, tab4 = st.tabs(["Home", "MixoNet", "USB", "Informações"])

# Conteúdo da página "Home"
with tab1:
    st.subheader("| A Classe Myxozoa")
    # Use uma única coluna para posicionar a imagem e o texto na mesma linha
    col1, col2 = st.columns([1,0.85])  # Defina a largura da primeira coluna

    with col1:
        # Adicione a imagem ao espaço em branco
        st.image("./images/sera.png", width=638)
        # Adicione a legenda da imagem
        st.caption("""Courtesy W.L. Current
                   Myxobolus/Myxosoma sp.
                   """, unsafe_allow_html=True)  
        # Adicione um espaçamento para criar espaço entre a imagem e o texto
        st.text("")  # Ajuste o espaço conforme necessário

    with col2:
        # Ajuste a largura da coluna 2 (texto)
        st.markdown(""*20)  # Isso cria um espaço em branco para ajustar a largura
        intro_text = """
        Os myxozoários são parasitas com ciclos de vida complexos, e etcc.
        """
        #st.markdown(intro_text)
        
        st.write(f'<p style="color:#9c9d9f">{intro_text}</p>', unsafe_allow_html=True)
        audio_file = open("images/p_9841290_826.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mpeg")

        st.subheader("| Seu ciclo de vida")
        st.write(
            '<p style="color:#9c9d9f">Seu ciclo de vida é indireto, envolvendo hospedeiros intermediários (peixes) e definitivos (anelídeos)</p>',
            unsafe_allow_html=True,
            )
        st.subheader("| Saúde Única")
        st.write(
            '<p style="color:#9c9d9f">Adicionar.</p>',
            unsafe_allow_html=True,
            )

# Adicione as informações adicionais
st.write("Desenvolvido por [Carneiro, G.S](http://lattes.cnpq.br/3771047626259544)")


#######################################################

# Conteúdo da página "MyxoNet"
with tab2:
   import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np

st.set_page_config(page_title="YOLO Object Detection",
                   layout='wide',
                   page_icon='./images/object.png')

st.header('Get Object Detection for any Image')
st.write('Please Upload Image to get detections')

with st.spinner('Please wait while your model is loading'):
    yolo = YOLO_Pred(onnx_model='./models/best.onnx',
                    data_yaml='./models/data.yaml')
    #st.balloons()

def upload_image():
    # Upload Image
    image_file = st.file_uploader(label='Upload Image')
    if image_file is not None:
        size_mb = image_file.size/(1024**2)
        file_details = {"filename":image_file.name,
                        "filetype":image_file.type,
                        "filesize": "{:,.2f} MB".format(size_mb)}
        #st.json(file_details)
        # validate file
        if file_details['filetype'] in ('image/png','image/jpeg'):
            st.success('VALID IMAGE file type (png or jpeg')
            return {"file":image_file,
                    "details":file_details}
        
        else:
            st.error('INVALID Image file type')
            st.error('Upload only png,jpg, jpeg')
            return None
        
def main():
    object = upload_image()
    
    if object:
        prediction = False
        image_obj = Image.open(object['file'])       
        
        col1 , col2 = st.columns(2)
        
        with col1:
            st.info('Preview of Image')
            st.image(image_obj)
            
        with col2:
            st.subheader('Check below for file details')
            st.json(object['details'])
            button = st.button('Get Detection from YOLO')
            if button:
                with st.spinner("""
                Geting Objets from image. please wait
                                """):
                    # below command will convert
                    # obj to array
                    image_array = np.array(image_obj)
                    pred_img = yolo.predictions(image_array)
                    pred_img_obj = Image.fromarray(pred_img)
                    prediction = True
                
        if prediction:
            st.subheader("Predicted Image")
            st.caption("Object detection from YOLO V5 model")
            st.image(pred_img_obj)
    
    
    
if __name__ == "__main__":
    main()


# Conteúdo da página "USB"
with tab3:
    st.header("USB")

    from streamlit_webrtc import webrtc_streamer
    import av
    from yolo_predictions import YOLO_Pred

    # load yolo model
    yolo = YOLO_Pred('./best.onnx',
                    './data.yaml')


    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        # any operation 
        #flipped = img[::-1,:,:]
        pred_img = yolo.predictions(img)

        return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


    webrtc_streamer(key="example", 
                    video_frame_callback=video_frame_callback,
                    media_stream_constraints={"video":True,"audio":False})


with tab4:
    st.subheader("| A Classe Myxozoa")
    pass
