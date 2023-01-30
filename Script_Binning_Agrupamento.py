# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:31:51 2023

@author: REGIS CARDOSO
"""

######################################################################################################
## ALGORITMO PARA FEATURE ENGINEERING UTILIZANDO BINNING ###
######################################################################################################

## IMPORTAR AS BIBLIOTECAS UTILIZADAS ###

import pandas as pd
import numpy as np
import statistics
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime
from scipy.fftpack import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler



## FUNÇÕES ###

# FUNÇÃO PARA FEATURE ENGINEERING VIA BINNING


def binning(df):
    
    max_valor = df['Valor'].max()
    
    min_valor = df['Valor'].min()
    
    divisor = 100
    
    total_valor = abs(max_valor) + abs(min_valor)
    
    acumulativo_feature = total_valor/divisor
    
    
    features = []
    
    for i in range(divisor+1):
        features.append(int(min_valor+(acumulativo_feature*(i))))
        
    
    dados = []
    dados = pd.DataFrame(dados, columns=features)
    
    dados = pd.DataFrame(data=None, index=['0'], columns=features)
    
        
    for i in range(len(features)):
        contagem = 0
        if i == 0:
            df_mask = df['Valor'] < features[i]
            dado_filtrado = df[df_mask]
            contagem = dado_filtrado['Valor'].count()
            dados[features[i]] = contagem
            
            contagem = 0
    
        if i > 0:
    
            df_mask = df['Valor'] > features[i-1]
            dado_filtrado = df[df_mask]
            
            df_mask = dado_filtrado['Valor'] < features[i]
            dado_filtrado = dado_filtrado[df_mask]
            
            contagem = dado_filtrado['Valor'].count()
            dados[features[i]] = contagem
            
            contagem = 0
            
            print(i)
            
    return (dados)
    


## IMPORTAR OS ARQUIVOS DE DADOS ###

df = pd.read_csv('Dado_Vibracao.csv', sep=';')

df.columns = ['Tempo', 'Valor']

columns = ['Tempo', 'Valor']


## CONVERTE OS DADOS EM FLOAT E ADICIONA O PONTO COMO SEPARADOR DECIMAL ###

df[columns] = df[columns].apply(lambda x: x.str.replace(',', '.').astype('float'))


df_binning = binning(df)