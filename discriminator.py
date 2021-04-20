from tensorflow.keras.layers import Flatten,Conv2D,Reshape,Dense,Conv2DTranspose,BatchNormalization,LeakyReLU,Add,Activation,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam


def discriminator_model(img_size):
    
    i = Input(shape=(img_size,))
    x = Dense(1024,activation=LeakyReLU())(i)
    
    x = Dense(512,activation=LeakyReLU())(x)
    
    x = Dense(256,activation=LeakyReLU())(x)
    
    x = Dense(1,activation='sigmoid')(x)
    
    model = Model(i,x)
    return model

