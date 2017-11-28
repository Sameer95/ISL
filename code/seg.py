import numpy as np
from scipy import misc
import math
from os import listdir
from PIL import Image
import PIL
import skimage.io

# data=np.genfromtxt('Skin_NonSkin.txt',delimiter='\t')
# lst=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
lst=['a','b','c','d','g','h','j','k','q','s','t','u','x','y']
Mat_YUV=np.asarray([[0.299,0.587,0.114],[-0.14713,-0.28886,0.436],[0.615,-0.51499,-0.10001]])
Mat_YIQ=np.asarray([[0.299,0.587,0.114],[0.596,-0.275,-0.321],[0.212,-0.528,0.311]])
# Mat_YUV=np.asarray([[0.399,0.687,0.214],[0.696,-0.175,-0.221],[0.312,-0.428,0.411]])

def calc_I(rgb):
	Y,I,Q=np.dot(Mat_YIQ,rgb)
	return I

def calc_theta(rgb):
	Y,U,V=np.dot(Mat_YUV,rgb)
	return math.degrees(math.atan2(V,U))

def check_skin(rgb):
	I=calc_I(rgb)
	Theta=calc_theta(rgb)
	return I>=12 and I<=180 and Theta>=0 and Theta<=130

ind=2
for l in lst:
	ct=0
 	files=listdir('images/'+l+'/'+str(ind))
 	files.sort()
 	for fil in files:
 		img=misc.imread('images/'+l+'/'+str(ind)+'/'+fil)
 		for i in range(img.shape[0]):
 			for j in range(img.shape[1]):
 				if not check_skin(img[i][j]):
 					img[i][j]=np.asarray([0,0,0])
 		misc.imsave('badaset/'+l+'/'+fil,img)
 		ct=ct+1
 		if(ct==60):
 			break
	# ind=ind+1






