import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import cv2
from skimage.transform import resize
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Conv2D,Reshape,Dense,Conv2DTranspose,BatchNormalization,LeakyReLU,Add,Activation,Input

from img_saver import *
from generator import *
from discriminator import *

folder_path = 'C:/Users/my_wo/ML&AI/Dishant/Gan/football'
list_img = os.listdir(folder_path)

img_h = 64
img_w = 64
img_c = 3
d = img_h*img_w



img = []
for i in list_img:
    a = cv2.imread(folder_path+'/'+i)
    a = cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
#    a = a / 255.0 * 2 - 1
    a = cv2.resize(a,(img_h,img_w))
#    a = a.reshape(-1,d)
    img.append(a)
    

img = np.array(img)
img = img / 255.0 * 2 - 1
img = img.reshape(-1,d)


discriminator = discriminator_model(d)
discriminator.compile( loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

generator = generator_model()


discriminator.summary()

generator.summary()



z = Input(shape=(latent_dim,))
z.shape


gen = generator(z)
gen


discriminator.trainable = False

fake_pred = discriminator(gen)

combined_model = Model(z, fake_pred)

combined_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

batch_size = 32
epochs = 1000
sample_period = 50

zeros = np.zeros(batch_size)
ones = np.ones(batch_size)

d_losses = []
g_losses = []

if not os.path.exists('gan_img'):
    os.makedirs('gan_img')
    
    
def sample_images(epoch):
  rows, cols = 5, 5
  noise = np.random.randn(rows * cols, latent_dim)
  imgs = generator.predict(noise)

  # Rescale images 0 - 1
  imgs = 0.5 * imgs + 0.5

  fig, axs = plt.subplots(rows, cols)
  idx = 0
  for i in range(rows):
    for j in range(cols):
      axs[i,j].imshow(imgs[idx].reshape(img_h, img_w), cmap='gray')
      axs[i,j].axis('off')
      idx += 1
  fig.savefig("gan_img/%d.png" % epoch)
  plt.close()



for epoch in range(epochs):
      
    idx = np.random.randint(0, img.shape[0], batch_size)
    real_imgs = img[idx]
    
    
    noise = np.random.randn(batch_size, latent_dim)
    fake_imgs = generator.predict(noise)
    
    
    plt.imshow(fake_imgs)
    
    d_loss_real, d_acc_real = discriminator.train_on_batch(real_imgs, ones)
    d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_imgs, zeros)

    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    d_acc  = 0.5 * (d_acc_real + d_acc_fake)
    
    noise = np.random.randn(batch_size, latent_dim)
    g_loss = combined_model.train_on_batch(noise, ones)
    
    
    noise = np.random.randn(batch_size, latent_dim)
    g_loss = combined_model.train_on_batch(noise, ones)
    
    
    d_losses.append(d_loss)
    g_losses.append(g_loss)
    
    if epoch % 100 == 0:
      print(f"epoch: {epoch+1}/{epochs}, d_loss: {d_loss:.2f}, \
        d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}")
    
    if epoch % sample_period == 0:
      sample_images(epoch)


