import pandas as pd
import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import statsmodels.api as sm
from decoradores import validar_listas


"""
Este módulo proporciona una clase 'Grafico' que permite crear gráficos que contengan varias series de tiempo.
"""


class Grafico:
    """
    Grafica varias series de tiempo en un mismo gráfico, con el fin de poder realizar comparaciones entre ellas
    """
    def __init__(self, titulo_grafico, titulo_x, titulo_y):
       """
        Inicializa los nombre y títulos que contendrá el gráfico general.

        Parametros:
            titulo_grafico (str): Título del gráfico.
            titulo_x (str): Título del eje X.
            titulo_y (str): Título del eje Y.
        """
       self.titulo_grafico = titulo_grafico
       self.titulo_x = titulo_x
       self.titulo_y = titulo_y

    #@validar_listas
    def serie_tiempo(self, primera_serie, segunda_serie = None, tercera_serie = None, cuarta_serie = None):
        """
        Crea y visualiza un gráfico con varias serie de tiempo.

        Parametros:
            primera_serie (list): Datos de la primera serie de tiempo (requerida).
            segunda_serie (list, optional): Datos de la segunda serie de tiempo. Por defecto None.
            tercera_serie (list, optional): Datos de la tercera serie de tiempo. Por defecto None.
            cuarta_serie (list, optional): Datos de la cuarta serie de tiempo. Por defecto None.
            trade_signal (DataFrame, optional): Datos de señales de trading. Por defecto None.

        Returns:
            Visualización de gráfico con varias series de tiempo
        """



        """gráfico de la serie de tiempo"""

        fig, ax = plt.subplots(dpi = 500) #tamaño del gráfico

        #formato de fecha
        date_format = DateFormatter('%h-%d-%y')
        ax.xaxis.set_major_formatter(date_format)
        ax.tick_params(axis='x', labelsize=8)
        fig.autofmt_xdate()

        #gráficar
        ax.plot(primera_serie, lw=0.75)
        ax.set_ylabel(self.titulo_y)
        ax.set_xlabel(self.titulo_x)
        ax.set_title(self.titulo_grafico)
        ax.grid()
        if segunda_serie is not None:
            ax.plot(segunda_serie, lw=0.75, alpha=0.75, color='orange')
            if tercera_serie is not None:
               ax.plot(tercera_serie, lw=0.75, alpha=0.75, color='green')
               if cuarta_serie is not None:
                  ax.plot(cuarta_serie, lw=0.75, alpha=0.75, color='purple')

        plt.show()