import pandas as pd
import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import statsmodels.api as sm

def validar_listas(func):
    """
    Decorador que verifica que todos los argumentos pasados a la función sean listas.
    """
    def wrapper(*args, **kwargs):
        for arg in args:
            if not isinstance(arg, (pd.Series)):
                raise ValueError("Todos los argumentos deben ser listas.")
        return func(*args, **kwargs)
    return wrapper




def validar_mismo_tamano(cls):
    original_init = cls.__init__

    def new_init(self, lista1, lista2, *args, **kwargs):
        if len(lista1) != len(lista2):
            raise ValueError("Las dos listas deben tener el mismo tamaño.")
        original_init(self, lista1, lista2, *args, **kwargs)

    cls.__init__ = new_init
    return cls

