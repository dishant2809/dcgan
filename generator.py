from tensorflow.keras.layers import Flatten,Conv2D,Reshape,Dense,Conv2DTranspose,BatchNormalization,LeakyReLU,Add,Activation,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam



latent_dim = 100
d = 64*64

def generator_model():
    
    i = Input(shape = (latent_dim,))
    x = Dense(256,activation=LeakyReLU())(i)
    x = BatchNormalization()(x)
    
    x = Dense(512,activation=LeakyReLU())(x)
    x = BatchNormalization()(x)
    
    x = Dense(1024,activation=LeakyReLU())(x)
    x = BatchNormalization()(x)
    
    x = Dense(d,activation='tanh')(x)
   
    
    model = Model(i,x)

    return model


