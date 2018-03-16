# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import imageio as im
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.color import rgb2gray


'''

im=im.imread('pn.png')

ig=rgb2gray(im)


#Here we are setting a threshhold value ,which we can use to separate the values above aspecified value
from skimage.filters import threshold_otsu
tv=threshold_otsu(ig)
mask=np.where(ig>tv,1,0)
#if the amount of nuclei pixels are more the we have to reverse the case.
if (np.sum(mask==0)<np.sum(mask==1)):mask=np.where(mask,0,1)
     
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(mask,cmap='gray')
plt.imshow(np.where(mask,mask,np.nan),cmap='rainbow',alpha=0.5)
plt.axis('off')
plt.title('masked map')

plt.subplot(1,2,2)
pixels=ig.flatten()
plt.hist(pixels,bins=100)
plt.vlines(tv,0,10000,colors='b',linestyles='--')       

plt.show



#ndimage.label gives us the the labeled image in label and number of labels in nlabels 
label,nlabels=ndimage.label(mask)
print('there are {} labels'.format(nlabels))

#Here we have to decrease the number of cells that are haveing low size (<10)
for j,c in enumerate(ndimage.find_objects(label)) : #ndimage.fund_objects gives us the different masks that are present in the images   
   #plt.imshow(c,cmap='gray')
    cell=ig[c]#c contains the splies like "(slice(0L, 6L, None), slice(0L, 11L, None))"
              #which gives us the different masks pices so ig[c] gives the image part 
    if np.product(cell.shape)<10 :
        print('the size of {} cell is very small'.format(j))
        mask=np.where(label==j+1,0,mask) #output 0 where cell is very small else it gives the value of mask


label,nlabels=ndimage.label(mask)#updated label
print('Updated mask has {} labels'.format(nlabels))

#Now we want to separate out the different labels and store them in an array labeled. Here each labeled[i] will contain the whole image displaying 
#only the i th nucleus in a image.ndimage.find_objects shows us the different nucleus rather than full image 
labeled=[]
for x in range(1,nlabels+1):
    l=np.where(label==x,1,0)
    labeled.append(l)


    

        
        





ce=ndimage.find_objects(label)[5]
cell_mask=mask[ce]
cell_open=ndimage.binary_opening(cell_mask,iterations=8)
plt.imshow(cell_mask,cmap='gray')
plt.imshow(cell_open,cmap='gray')  

'''
#RLE encoding
def encoding(x):
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])

def analyze_image(im_path):
    img=im.imread(str(im_path))
    im_id=im_path.parts[-3]
    ig=rgb2gray(img)
    
    from skimage.filters import threshold_otsu
    tv=threshold_otsu(ig)
    mask=np.where(ig>tv,1,0)
    #if the amount of nuclei pixels are more the we have to reverse the case.
    if (np.sum(mask==0)<np.sum(mask==1)):mask=np.where(mask,0,1)
    
    #ndimage.label gives us the the labeled image in label and number of labels in nlabels 
    
    label,nlabels=ndimage.label(mask)#updated label
    print('Updated mask has {} labels'.format(nlabels))
    
    #Now we want to separate out the different labels and store them in an array labeled. Here each labeled[i] will contain the whole image displaying 
    #only the i th nucleus in a image.ndimage.find_objects shows us the different nucleus rather than full image 
    im_df=pd.DataFrame()
    for x in range(1,nlabels+1):
        l=np.where(label==x,1,0)
        if l.flatten().sum()>10:
            rle=encoding(l)
            print('RLE Encoding for the current label is: {}'.format(rle))
            s=pd.Series({'Imadge ID':im_id,'Encoding values': rle})
            im_df=im_df.append(s,ignore_index=True)
    return im_df

def separate_each_image_path(im_list):
    im_rows=pd.DataFrame()
    for i in im_list:
        temp=analyze_image(i)
        im_rows=im_rows.append(temp,ignore_index=True)
    return im_rows

def main():
    testing = pathlib.Path('D:\\AI\\kaggle super bowl 18\\train').glob('*/images/*.png')
    df = separate_each_image_path(list(testing))
    df.to_csv('final.csv',index=None)


        
     
 
    








