from matplotlib import pyplot as plt
import numpy as np
import SimpleITK as sitk
import os
from sklearn.model_selection import train_test_split
from keras.losses import binary_crossentropy
import numpy as np
import os
from keras import backend as K
from keras import  metrics
from keras import utils
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, BatchNormalization, Dropout, UpSampling2D, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merging import concatenate
from fonction_model import * 
import segmentation_models as sm
from skimage.restoration import denoise_tv_chambolle

new_dim=512
iou=sm.metrics.IOUScore()   # Returns:	A callable iou_score instance. Can be used in model.compile(...) function.
dice = sm.metrics.FScore() 
dices=[]
ious=[]
ct_bm06 = []
masque_bm06 = []
ct_path_pretraite = "/Users/smain/Downloads/archive/ct_scans_pretraite/"
contour_path_pretraite = "/Users/smain/Downloads/archive/lungs_mask_pretraite/"
patients = os.listdir(ct_path_pretraite) # Get patients list by reading all directories names in the CT folder
patients.sort()


def compare_actual_and_predicted(image_no,ct_bm06,masque_bm06): # fonction qui affiche l'image CT de BM06, son masque expert et son masque générée par le U-Net

    mask_result = model.predict(ct_bm06[image_no].reshape(1,new_dim, new_dim, 1)) #on mets en entrée du réseau une image du np.array ct_BM06, on obtient un masque en sortie (mask_result)
    model.compile(loss='binary_crossentropy',metrics=[dice,iou])
    score=model.evaluate(ct_bm06[image_no].reshape(1,new_dim, new_dim, 1),masque_bm06[image_no].reshape(1,new_dim, new_dim, 1))
    dices.append(score[1])
    ious.append(score[2])
    
    fig = plt.figure(figsize=(15,15))
    plt.subplot(1,3,1)
    plt.imshow(ct_bm06[image_no].reshape(new_dim, new_dim), cmap='bone') #affiche l'image CT
    plt.title('Original Image (CT)')

    plt.subplot(1,3,2)
    plt.imshow(masque_bm06[image_no].reshape(new_dim,new_dim)) #affiche le masque expert liée à l'image CT
    plt.title('Actual mask')

    plt.subplot(1,3,3)
    plt.imshow(mask_result.reshape(new_dim,new_dim)) #affiche le masque U-net 
    plt.title('Predicted mask')

    plt.show()

# création tableau de coupes 2D à partir des images 3D 
for patient_id in patients:
  if patient_id.startswith("im") : # Pour les images de bm06
      ct_img = sitk.ReadImage(ct_path_pretraite+patient_id) 
      contour_poumon = sitk.ReadImage(contour_path_pretraite+patient_id) #masque contour poumon 
      ct_img_array = sitk.GetArrayFromImage(ct_img)
      contour_poumon_array = sitk.GetArrayFromImage(contour_poumon)   
      ct_bm06.append(ct_img_array)
      masque_bm06.append(contour_poumon_array)

# transformation tableau en np.array pour le réseau U-net      

ct_bm06 = np.array(ct_bm06)
masque_bm06 = np.array(masque_bm06)


masque_bm06[masque_bm06==2]=1

# Ajout d'une dimension à toutes les np.array, nécaissaire pour entrer dans la première couche du réseau U-net (1,512,512,1)

ct_bm06 = ct_bm06.reshape(len(ct_bm06), 512, 512, 1)
masque_bm06 = masque_bm06.reshape(len(ct_bm06), 512, 512, 1)






#Configuration du réseau U-Net
inputs = Input((512, 512, 1)) # Input = Entrée du réseau : image (512,512,1)

c00 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (inputs) #Conv2D = Couche de convolution 
c00 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c00)
c00 = BatchNormalization()(c00) # Batch-Normalization : applique une transformation en maintenant la sortie moyenne proche de 0 et l’écart type de sortie proche de 1. 
p00 = MaxPooling2D((2, 2)) (c00) # Max-pooling : Le pooling consiste à sous-échantillonner une représentation d’entrée en réduisant sa dimension et ainsi de réduire le nombre de paramètres et de calculs dans le réseau
p00 = Dropout(0.25)(p00) # Dropout : La couche Dropout définit de manière aléatoire des unités d'entrée sur 0  ce qui aide à prévenir le surentraînement.
                     #   Les entrées non définies sur 0 sont mises à l'échelle de 1/(1 - taux) de sorte que la somme sur toutes les entrées sont inchangées.

c0 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p00)
c0 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c0)
c0= BatchNormalization()(c0)
p0 = MaxPooling2D((2, 2)) (c0)
p0 = Dropout(0.25)(p0)

c1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p0)
c1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c1)
c1 = BatchNormalization()(c1)
p1 = MaxPooling2D((2, 2)) (c1)
p1 = Dropout(0.25)(p1)

c2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p1)
c2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c2)
c2 = BatchNormalization()(c2)
p2 = MaxPooling2D((2, 2)) (c2)
p2 = Dropout(0.25)(p2)

c3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p2)
c3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c3)
c3 = BatchNormalization()(c3)
p3 = MaxPooling2D((2, 2)) (c3)
p3 = Dropout(0.25)(p3)

c4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p3)
c4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c4)
c4 = BatchNormalization()(c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
p4 = Dropout(0.25)(p4)

c5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p4)# Couche entièrement connectée
c5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c5)# Couche entièrement connectée

u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5) # Conv2DTranspose : Couche de convolution transposé
u6 = concatenate([u6, c4]) # Concatenate = Skip de connection pour stabiliser l'entraînement et améliorer la convergence du modèle
u6 = BatchNormalization()(u6)
c6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u6)
c6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c6)


u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
u7 = BatchNormalization()(u7)
c7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u7)
c7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c7)


u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
u8 = BatchNormalization()(u8)
c8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u8)
c8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c8)


u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
u9 = BatchNormalization()(u9)
c9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u9)
c9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c9)

u10 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c9)
u10 = concatenate([u10, c0], axis=3)
u10 = BatchNormalization()(u10)
c10 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u10)
c10 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c10)

u11 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c10)
u11 = concatenate([u11, c00], axis=3)
u11 = BatchNormalization()(u11)
c11 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u11)
c11 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c11)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c11) # Output = Sortie du réseau : image (512,512,1)

model = Model(inputs=[inputs], outputs=[outputs]) # On définit le modèle (U-net) par l'entrée input, et la sortie output 
model.summary() #affichage du résumé du réseau construit

model.load_weights('/Users/smain/Documents/Tensorflow/unet_covid_weights_val_accuracy.hdf5') # application des poids enregistrés après l'entrainement au réseau U-net

for i in range(10):
    compare_actual_and_predicted(i,ct_bm06,masque_bm06)
print('DICES:',dices) #récupération de toutes les valeurs de dices des masques U-net
print("IOUS:",ious) #récupération de toutes les valeurs de IOU des masques U-net