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

new_dim=512
iou=sm.metrics.IOUScore()   # Returns:	A callable iou_score instance. Can be used in model.compile(...) function.
dice = sm.metrics.FScore() 
dices=[]
ious=[]
ct_base_donnes = []
masque_base_donnes = []
ct_bm06 = []
masque_bm06 = []
ct_path_pretraite = "/Users/smain/Downloads/archive/ct_scans_pretraite/"
contour_path_pretraite = "/Users/smain/Downloads/archive/lungs_mask_pretraite/"
patients = os.listdir(ct_path_pretraite) # Get patients list by reading all directories names in the CT folder
patients.sort()


# création tableau de coupes 2D à partir des images 3D 
for patient_id in patients:
  print(patient_id)
  if patient_id.startswith("coronacas") or patient_id.startswith("radio"): # pour les images de la  base de données
      ct_img = sitk.ReadImage(ct_path_pretraite+patient_id) 
      contour_poumon = sitk.ReadImage(contour_path_pretraite+patient_id) #masque contour poumon 
      ct_img_array = sitk.GetArrayFromImage(ct_img)
      contour_poumon_array = sitk.GetArrayFromImage(contour_poumon)
      for i in range(64) :        
        ct_base_donnes.append(ct_img_array[i,:,:])
        masque_base_donnes.append(contour_poumon_array[i,:,:])
  elif patient_id.startswith("im") : # Pour les images de bm06
      ct_img = sitk.ReadImage(ct_path_pretraite+patient_id) 
      contour_poumon = sitk.ReadImage(contour_path_pretraite+patient_id) #masque contour poumon 
      ct_img_array = sitk.GetArrayFromImage(ct_img)
      contour_poumon_array = sitk.GetArrayFromImage(contour_poumon)   
      ct_bm06.append(ct_img_array)
      masque_bm06.append(contour_poumon_array)

# transformation tableau en np.array pour le réseau U-net      
ct_base_donnes = np.array(ct_base_donnes)
masque_base_donnes = np.array(masque_base_donnes)
ct_bm06 = np.array(ct_bm06)
masque_bm06 = np.array(masque_bm06)

##Si les masques on deux valeurs (pour les deux poumons), on fait en sorte qu'ils en ont qu'une -> nécaissaire pour l'utilisation de la loss function Binary cross Entropy
masque_base_donnes[masque_base_donnes==2]=1
masque_bm06[masque_bm06==2]=1

# Ajout d'une dimension à toutes les np.array, nécaissaire pour entrer dans la première couche du réseau U-net (1,512,512,1)
ct_base_donnes = ct_base_donnes.reshape(len(ct_base_donnes), 512, 512, 1)
masque_base_donnes = masque_base_donnes.reshape(len(ct_base_donnes), 512, 512, 1)
ct_bm06 = ct_bm06.reshape(len(ct_bm06), 512, 512, 1)
masque_bm06 = masque_bm06.reshape(len(ct_bm06), 512, 512, 1)



#séparation des données en test, en utilisant les images de la base de données et validation  x=ct y=label
x_train,x_valid,y_train,y_valid=train_test_split(ct_base_donnes,masque_base_donnes,test_size=0.3,random_state=42)

x_valid=np.concatenate((ct_bm06,x_valid))
y_valid=np.concatenate((masque_bm06,y_valid))


valid_datagen = ImageDataGenerator() #Générator qui ne transforme en rien l'image ( pas de Data_augmentation par exemple)

train_data = valid_datagen.flow(ct_base_donnes,masque_base_donnes,# les coupes des images CT et les coupes des masques sont liées quand ils passeront dans le réseau dans la phase d'entrainement
                                             batch_size=8) #On fait en sorte que les données d'entraînement est un batch de 8 
        #-> Le Batch correspond à le nombre d'images qui passent en parallèles dans le réseau à chaque itération d'une Epoch 


valid_data = valid_datagen.flow(ct_bm06,masque_bm06, # les coupes des images CT et les coupes des masques sont liées quand ils passeront dans le réseau pour la phase de validation
                                             batch_size=1)    




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


epochs = 1 #nombre de fois où tous les données d'entraînement vont passer à travers le réseau

filepath_loss = "unet_covid_weights_val_accuracy_data_augmen.hdf5" 
filepath_dice = "unet_covid_weights_val_dice_data_augmen.hdf5"

checkpoint_acc = ModelCheckpoint(filepath_loss, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
#on enregistre les poids des neurones quand il a un nouveau pax de val_accuracy
checkpoint_dice = ModelCheckpoint(filepath_dice, monitor='val_dice_coeff', verbose=1, save_best_only=True, mode='max')
#on enregistre les poids des neurones quand il a un nouveau de val_dice_coeff


model.compile(optimizer=Adam(learning_rate= 0.0005), loss='binary_crossentropy', metrics=('accuracy',dice_coeff))
#On utilise la binary-cross entropy comme fonction de coût pour re-évaluer les poids des neurones à travers le réseau, à chaque itérations (dans une epoch).
# On applique un learning_rate (taux de de changement des poids des neurones) de 0.0005.
# Le taux de changement des poids sera optimiser par l'opimiseur Adam
#On utilise 2 metriques : L'accuracy (pixel_ = pixel ) et le DICE (qui mesure la superposition du masque expert et le masque générée)

results = model.fit(train_data, epochs=epochs,
                    validation_data=valid_data,
                    callbacks=[checkpoint_acc,checkpoint_dice]) #On donne au model, le nombre d'epoch construit des données d'entrainement de validation et les calbacks pour sauvegarder les poids aux checkpoints crées 


model.save('model_covid_lungs')

