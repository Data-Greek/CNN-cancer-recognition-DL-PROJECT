# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 20:10:22 2020

@author: TBEL972
"""
## CHARGEMENT DES LIBRAIRIES--------------------------------------------------
import pandas as pd
import numpy as np
import tensorflow 
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

## CHARGEMENT DU MODÈLE-------------------------------------------------------

model = load_model('cnn.h5')

## FONCTION DE PRÉDICTION-----------------------------------------------------

def traitement(image_data, model):
        
    size = (165,133)
     
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image=np.asarray(image)
    image = (image.astype(np.float32) / 255.0)
    img_reshape = np.expand_dims(image, axis = 0)
    img_reshape = image[np.newaxis,...,np.newaxis]

    prediction = model.predict(img_reshape)
    
    return prediction


## LA WEB APP-----------------------------------------------------------------

#création et personnalisation de la sidebar-----------------------------------

picture_1 = Image.open('Logo.png')

st.sidebar.image(picture_1, use_column_width=True)

picture_2 = Image.open("Lung_disease.jpg")
    
st.sidebar.image(picture_2, use_column_width = True)
    
st.sidebar.info("""Cette application web a été réalisée à des fins de démonstration par l'**Agence Marketic**.""")
                     
st.sidebar.success("""Vous souhaitez concevoir votre propre application web pour faciliter le travail de vos collaborateurs ? Rejoignez-nous sur **http://www.agence-marketic.fr**""")

#Page centrale----------------------------------------------------------------
    
html_template = """
<div style = "background-color : #32CD32 ; padding:15px">
<h2 style="color : white; text-align : center; "> Interface de détection du cancer des poumons </h2>
    
"""
st.markdown(html_template, unsafe_allow_html=True)
    
st.write("")

st.write(""" Cette interface de prédiction a pour objectif de diagnostiquer le cancer du poumon à partir de l'analyse de scanners au rayon X. Elle permet d'identifier les patients atteints par la maladie, avec une **précision de 99%** 🩺. Le modèle de prédiction repose sur un réseau de neurones à reconnaissance d'image, aussi connu sous l'appellation de réseaux de neurones convolutifs (CNN). """)

st.write("")

st.write(""" [🟢 - Réalisez un test avec une image de poumons présentant un état sain](https://drive.google.com/uc?export=download&id=1Muzi-Fzf0z4B81Tcpd_5gvDwkQkl40GM)""")
st.write(""" [🔴 - Réalisez un test avec une image de poumons présentant un état cancéreux](https://drive.google.com/uc?export=download&id=1Muzi-Fzf0z4B81Tcpd_5gvDwkQkl40GM)""")
         
xray = st.file_uploader(""" ⬇️ Veuillez insérer votre image de poumons en cliquant sur ''browse files'' ⬇️ """, type=['jpeg', 'jpg', 'png'])

#Traitement de l'image------------------------------------------------------
if xray is not None:

    lung_image = Image.open(xray)
    st.image(lung_image, use_column_width=True)
    predict = traitement(lung_image, model)
            
    #Prédiction et légence explicative
    if predict > 0.5:
            
        st.success("""Les résultats indiquent un **état sain des poumons** chez le patient""")
    else: 
        st.error("""Les résultats indiquent un **état cancéreux des poumons** chez le patient""")
        
    np.set_printoptions(suppress=True)
    st.write("""Le **seuil de probabilité** est estimé à {}""".format(predict))
    st.warning("""Plus le seuil de probabilité **se rapproche de 0**, plus le risque de cancer est élevé. À l'inverse, plus il **s'approche de 1**, plus le risque de cancer du poumon est faible.""")

