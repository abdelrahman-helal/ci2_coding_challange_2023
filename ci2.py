#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install opencv-python


# In[21]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2


# In[67]:


class Chest_Image:
    def __init__(self, path): # an instance that takes the path of the image everytime the class is called 
        self.path = path 
        data = np.load(self.path) # loading the data as a dictionary
        i = random.randint(1, len(data['train_images'])) 
        self.selection = data["train_images"][i]
        print(i)
        # choosing a random image from the train images 
        
    def show_image(self): 
        plt.imshow(self.selection, cmap = 'gray')    # displaying the image with a grayscale 
        plt.savefig('D:\\ Abdelrahman_Helal_1.png')  
        
    def invert_image(self):
        self.inverted_image = 255 - self.selection 
        # Each pixel is in the range (0, 255), so we substract each pixel value from 255 to get the inversion
        plt.imshow(self.inverted_image, cmap = 'gray')
        plt.savefig('D:\\ Abdelrahman_Helal_2.png')
        
    def enhanced(self):
        '''
           I am changing the size of the image from 28x28 pixels to 200x200, and approximating the values of the new pixels with
           Lanczos4 interplotation because it results in the least information loss and approximates each new pixel based on 
           the nearest 8x8 area. 
        '''
        third_image = cv2.resize(self.selection, [200, 200], interpolation=cv2.INTER_LANCZOS4)    
        plt.imshow(third_image, cmap = 'gray')
        plt.savefig('D:\\ Abdelrahman_Helal_3.png')
        fig, (ax1, ax2) = plt.subplots(ncols = 2)
        sns.heatmap(third_image, cmap = 'gray', ax = ax1)
        sns.heatmap(self.selection, cmap = 'gray', ax = ax2)
        # plotting the heatmap of the original vs the modified image to show the difference in the amount of information in both

        
        



        
    


# In[68]:


chest = Chest_Image(r"C:\Users\Abdelrahman Helal\Downloads\chestmnist.npz")
# becaue of the random value, each time the class is run, there will be a different image, but this image has an index of 4599


# In[69]:


chest.show_image()


# In[70]:


chest.invert_image()


# In[71]:


chest.enhanced()


# In[ ]:




