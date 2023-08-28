import pandas as pd
import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import statsmodels.api as sm
from red_neuronal import RedNeuronal
from grafico_serie_tiempo import Grafico

#Descarga la serie de tiempo que se quiere pronosticar
btc_usd = yf.download('BTC-USD', start = '2020-01-01', end = '2020-12-31', interval= '1d')

grafico_btc_usd = Grafico("BTC - USD", "Price", "Day")

grafico_btc_usd.serie_tiempo(btc_usd['Close'])

print(type(btc_usd['Close']))

#Se define la matriz con las variables de entrada y la matriz con los resultados que se esperan (train)
valor_esperado = btc_usd['Open']
valores_entrada = btc_usd['Close']

#Con los datos, se corre la función para entrenar la red neuronal con variables de entrada y los valores esperados
rn_btc = RedNeuronal(valores_entrada, valor_esperado, 5)

rn_btc.ajuste()

#Pronostico para el tiempo t+1
rn_btc.pronostico()



