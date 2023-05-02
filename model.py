from keras.layers import Dense
from keras.models import Sequential


def get_model():
    model = Sequential()
    model.add(Dense(8,input_dim=7,activation="relu"))
    model.add(Dense(5,activation='relu'))

    model.add(Dense(3,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    return model