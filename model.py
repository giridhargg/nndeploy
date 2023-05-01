import pandas as pd
import seaborn as sns
import numpy as np
import pickle
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.losses import BinaryCrossentropy
from sklearn.metrics import confusion_matrix


def get_model():
    model = Sequential()
    model.add(Dense(8,input_dim=7,activation="relu")) #input layer
    model.add(Dense(5,activation='relu')) #hidden layer 1

    model.add(Dense(3,activation='relu')) #hidden layer 2
    model.add(Dense(1,activation='sigmoid')) #output layer

    return model