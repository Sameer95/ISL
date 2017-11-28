import pickle
import numpy as np
from sklearn import svm

x=pickle.load(open("proj_hog.p","rb"))
y=pickle.load(open("Y.p","rb"))
z=pickle.load(open("Z.p","rb"))
x_train=[]
y_train=[]
y_test=[]
x_test=[]
ct=0
y=[ord(l)-97 for l in y]
# print y[160:200]
for row in x:
	if(z[ct]!='4'):
		x_train.append(row)
		y_train.append(y[ct])
	else:
		x_test.append(row)
		y_test.append(y[ct])
	ct=ct+1

clf = svm.SVC(kernel="linear",max_iter=10000000)
clf.fit(x_train,y_train)
res=clf.predict(x_test)
# print len(y_test)
# pickle.dump(res,open("res.p","wb"))
a=np.zeros(shape=(26,26))
c=0
print "a b c d e f g h i j k l m n o p q r s t u v w x y z"
for i in res:
	a[y_test[c]][i]+=1
	c=c+1
a=a.astype(int)
print "[["
for row in a:
	for i in row:
		print i,
		print ",",
	print "],"
	print "\n"
	print "[",
pickle.dump(a,open("new_cm_hog1.p","wb"))