import SimpleITK as sitk
import numpy as np
import pandas as pd 
import os
from skimage import exposure
from skimage.restoration import denoise_tv_chambolle

def normalize_img(image, val_inf=-500, val_sup=1600):
    image[image<val_inf] = val_inf
    image[image>val_sup] = val_sup
    return image
    
final_img_size = (512,512,2)

def idx_slices_to_delete(mr_array):
    '''Get index of slices outside of the MRI external contour (z axis)
    Both CT and MRI have the same spacing, origin and number of slices
    mr_array : MR image as numpy array type
    return : rm_index, a list containing all the empty slices index
    '''
    rm_index = []
    for i in range(0,mr_array.shape[0]): 
        if np.all(np.mean(mr_array[i])<=0.2): 
            rm_index.append(i)
    
    return rm_index

def resample_image(input_image, new_size, is_mask=False): 
    ''' Resample an image to have the desired number of slices
    input_image: image to resample, has to be a sitk image type
    new_size: size of the output image (x,y,z)
    return: image with the new dimension (sitk img)
    '''
    x_spacing = input_image.GetSize()[0] * input_image.GetSpacing()[0] / new_size[0]
    y_spacing = input_image.GetSize()[1] * input_image.GetSpacing()[1] / new_size[1]

    new_spacing = (x_spacing, y_spacing)

    resample = sitk.ResampleImageFilter()
        
    if is_mask: 
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else: 
        resample.SetInterpolator(sitk.sitkLinear)
        
    resample.SetOutputOrigin(input_image.GetOrigin())
    resample.SetOutputDirection(input_image.GetDirection())
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)

    output_image = resample.Execute(input_image)
    
    return output_image


 # Get patients list by reading all directories names in the CT folder

final_img_size = (512,512,64)
results_path_CT = "/Users/smain/Downloads/archive/ct_scans_pretraite/"
results_path_contour = "/Users/smain/Downloads/archive/lungs_mask_pretraite/"

patient_id = "im0.nii"
ct_img = sitk.ReadImage(results_path_CT+patient_id)  
# From sitk img to numpy array
ct_img_array = sitk.GetArrayFromImage(ct_img)
ct_img_array = np.rot90(ct_img_array,1,axes=(0,1))
writer = sitk.ImageFileWriter()
contour_poumon = sitk.ReadImage(results_path_contour+patient_id) #masque contour poumon 

# From sitk img to numpy array
ct_img_array = sitk.GetArrayFromImage(ct_img)
ct_img_array= np.rot90(ct_img_array,1,(1,0))
#mr_cext_array = sitk.GetArrayFromImage(mr_cext)
contour_poumon_array = sitk.GetArrayFromImage(contour_poumon)

# Get empty slices id
id_empty_slices = idx_slices_to_delete(ct_img_array)
print(id_empty_slices)
for i in range(2) :  
    #print(img_name)
        if i==0 :
            img = ct_img
            spacing = ct_img.GetSpacing()
            origin = ct_img.GetOrigin() 
        else :
            img = contour_poumon
            spacing = contour_poumon.GetSpacing()
            origin = contour_poumon.GetOrigin() 

        img_array = np.rot90(sitk.GetArrayFromImage(img),1,(1,0))
        new_array = np.delete(img_array, id_empty_slices, 0)
    
        new_array = np.rot90(new_array,-1,(1,0))
        final_img = sitk.GetImageFromArray(new_array)
        final_img.SetSpacing(spacing)
        final_img.SetOrigin(origin)

        
        if i==0 :
            final_img = resample_image(final_img,final_img_size,is_mask=False)
            final_path = results_path_CT+'im11.nii'        
        else:
            final_img = resample_image(final_img,final_img_size,is_mask=True) 
            final_path = results_path_contour+'im11.nii'
        writer.SetFileName(final_path)
        writer.Execute(final_img)
            

