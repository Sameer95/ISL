from os import listdir
import PIL
from PIL import Image
from skimage.feature import hog
import skimage.io
import numpy as np
import pickle

X_list=[]
Y_list=[]
Z_list=[]
#for i in range(97,123):
#	print i
	#char=chr(i)
files=listdir('/path_to_dataset/')
files.sort()
for fil in files:
	im=skimage.io.imread('/path_to_dataset/'+fil,as_grey=True)
	x=hog(im)
	# print len(x)
	X_list.append(x)
	Y_list.append(fil[0])
	Z_list.append(fil[1])

X=np.asarray(X_list)
Y=np.asarray(Y_list)
Z=np.asarray(Z_list)
pickle.dump(X,open('X.p','wb'))
pickle.dump(Y,open('Y.p','wb'))
pickle.dump(Z,open('Z.p','wb'))
