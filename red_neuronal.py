import pandas as pd
import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import statsmodels.api as sm
from decoradores import validar_listas, validar_mismo_tamano


class RedNeuronal():
    
    @validar_mismo_tamano
    def __init__(self, serie_tiempo, valor_pronostico, retardos):
        """
        Inicializa una red neuronal para el pronóstico de una serie de tiempo.

        Parametros:
            serie_tiempo (pd.Series): Serie de tiempo de entrada.
            valor_pronostico (pd.Series): Valores de pronóstico objetivo.
            retardos (int): Número de retardos para el modelo AR.
        
        Atributos:
            ydk (pd.Series): serie donde se encuentra los valores contra los cuales queremos comparar nuestro modelo.
            x (pd.Series): matriz con los valores de los retardos previos, los cuales serán als variables de entrada para el modelo.
            NUMERO_ENTRADAS (int): será una constante que determinará la cantidad de retardos
            w (pd.Series): matríz de pesos, los valores en esta matriz determinan cómo las entradas se ponderan y se transmiten información a las neuronas ocultas. Durante el proceso de entrenamiento, estos pesos se ajustan para que la red pueda aprender a capturar patrones y relaciones en los datos de entrada.
            c (pd.Series): matriz de conexiones que conecta las neuronas de la capa oculta con la capa de salida, determina cómo las señales de las neuronas ocultas influyen en las salidas finales de la red. Durante el entrenamiento, los valores de esta matriz se ajustan gradualmente para minimizar el error entre las salidas pronosticadas y los valores reales.
        """

        #ydk: precio de apertura -> el que queremos pronosticar
        #x(k-t): precios anteriores de cierre
        self.ydk = valor_pronostico

        #Modelo AR(p)
        self.x = pd.DataFrame()
        #Matriz con los retardos de la serie de tiempo
        for i in range(1, retardos + 1):
            self.x[f"x{i}"] = serie_tiempo.shift(i) / serie_tiempo.max()

        self.x = self.x.dropna()        

        self.NUMERO_ENTRADAS = retardos
        NEURONAS_OCULTAS = 10
        CANTIDAD_DATOS = len(self.x)
        
        # Matrices de pesos y conexiones
        self.w = np.random.normal(loc=0, scale=1, size=self.NUMERO_ENTRADAS*NEURONAS_OCULTAS)
        self.w = pd.DataFrame(self.w.reshape(NEURONAS_OCULTAS, self.NUMERO_ENTRADAS))

        self.c = np.random.normal(loc=0, scale=1, size= 1*NEURONAS_OCULTAS)
        self.c = pd.DataFrame(self.c.reshape(1, NEURONAS_OCULTAS))

        # Matrices para almacenar resultados intermedios
        self.hj = pd.DataFrame(0, index= range(NEURONAS_OCULTAS), columns= range(1) )
        self.ys = pd.DataFrame(np.zeros((CANTIDAD_DATOS, 1))) 
        self.yr = pd.DataFrame(0, index= range(CANTIDAD_DATOS), columns= range(1) )
        self.ek = pd.DataFrame(0, index= range(CANTIDAD_DATOS), columns= range(1) )


    def ajuste(self):
        """
        Realiza el ajuste de la red neuronal utilizando el método de backpropagation.
        
        El método de backpropagation ajusta los pesos y conexiones de la red neuronal
        utilizando el error entre los valores pronosticados y los valores reales.
        
        Backpropagation: 
            1. Calcula el error de pronóstico (ek) como la diferencia entre el valor pronosticado (yr) y el valor real (ydk).
            2. Actualiza la conexión (c) entre la capa oculta y la capa de salida utilizando el error de pronóstico y el valor intermedio (hj) de la capa oculta.
            3. Actualiza los pesos (w) entre la capa de entrada y la capa oculta utilizando el error de pronóstico, la conexión (c) y los valores de entrada (x).

        """
        NEURONAS_OCULTAS = 10
        CANTIDAD_DATOS = len(self.x)

        ALFA = 0.00001

        NUMERO_ITERACIONES = 2000

        for i in range(NUMERO_ITERACIONES):
            for k in range(CANTIDAD_DATOS):
                self.hj = pd.DataFrame(np.dot(self.w, self.x.iloc[k]))
                self.ys.iloc[k] = np.dot(self.c, self.hj)

                #función de activación
                #yr.iloc[k] = 1 / (1 + np.exp(ys.iloc[k]))
                self.yr.iloc[k] = self.ys.iloc[k]

                #derivada
                #dev = np.exp(-ys.iloc[k]) / ((1 + np.exp(-ys.iloc[k]))**2)
                dev = 1

                #error
                self.ek.iloc[k] = self.ydk.iloc[k] - self.yr.iloc[k]

                #corrección a partir del error
                self.c = self.c + ALFA * np.dot(self.ek.iloc[k], self.hj.T) * dev
                self.w = self.w + ALFA * self.ek.iloc[k].values * np.dot(self.c.T, pd.DataFrame(self.x.iloc[k,:]).T ) * dev
        
        self.yr= self.yr.set_index(self.x.index)


        return self.yr
    
    def pronostico(self):
        hj_nuevo = pd.DataFrame(np.dot(self.w, self.x.iloc[-1]))
        ys_nuevo = np.dot(self.c, hj_nuevo)

        #función de activación
        #yr.iloc[k] = 1 / (1 + np.exp(ys.iloc[k]))
        
        yr_nuevo = ys_nuevo

        return yr_nuevo





