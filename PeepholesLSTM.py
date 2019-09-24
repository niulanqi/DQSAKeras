from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.activations import *
from config import config



def PeepholesLSTM(units=config.LstmUnits, stateful=True, return_sequences=True):
    lstmcell = PeepholeLSTMCell(units=units)
    return RNN(cell=lstmcell, return_sequences=return_sequences, stateful=stateful)