import pickle
import numpy as np 
from sklearn.svm import LinearSVC as linsvm
X=pickle.load(open('X.p','rb'))
Y=pickle.load(open('Y.p','rb'))
Z=pickle.load(open('Z.p','rb'))
Y=[ord(x)-97 for x in Y]
Y=np.asarray(Y)
#Directly training on all the 26 classes
ind_train=[i for i in range(0,len(Z)) if Z[i]!='4']
ind_test=[i for i in range(0,len(Z)) if Z[i]=='4']
X_train=X[ind_train]
X_test=X[ind_test]
Y_train=Y[ind_train]
Y_test=Y[ind_test]
model1=linsvm()
model1.fit(X_train,Y_train)
print np.sum(model1.predict(X_test)==Y_test)/float(len(Y_test))
#One try more
misclass=[4,5,6,7,8,12,13,18,21]
# misclass=[0,16,4,5,8,24,6,9,11,20,12,13,14,19,20,21]
needed=[i for i in range(0,26) if i not in misclass]
Y_level1=np.asarray([1 if Y[i] in misclass else 2 for i in range(0,len(Y))])
Y_train2=Y_level1[ind_train]
# print Y_train2
Y_test2=Y_level1[ind_test]
modelfirst=linsvm()
modelfirst.fit(X_train,Y_train2)
print np.sum(modelfirst.predict(X_test)==Y_test2)/float(len(Y_test2))


ind_train_misclass=[i for i in range(0,len(Y_train)) if Y_train[i] in misclass]
ind_train_needed=[i for i in range(0,len(Y_train)) if Y_train[i] in needed]
ind_test_misclass=[i for i in range(0,len(Y_test)) if Y_test[i] in misclass]
ind_test_needed=[i for i in range(0,len(Y_test)) if Y_test[i] in needed]

X_train_needed=X_train[ind_train_needed]
Y_train_needed=Y_train[ind_train_needed]
X_train_misclass=X_train[ind_train_misclass]
Y_train_misclass=Y_train[ind_train_misclass]
X_test_needed=X_test[ind_test_needed]
Y_test_needed=Y_test[ind_test_needed]
X_test_misclass=X_test[ind_test_misclass]
Y_test_misclass=Y_test[ind_test_misclass]

modelneeded=linsvm()
modelmisclass=linsvm()

print "main"
modelneeded.fit(X_train_needed,Y_train_needed)
# print np.sum(modelneeded.predict(X_test_needed)==Y_test_needed)/float(len(Y_test_needed))

print "misclass"
modelmisclass.fit(X_train_misclass,Y_train_misclass)
# print np.sum(modelmisclass.predict(X_test_misclass)==Y_test_misclass)/float(len(Y_test_misclass))


def predictor(x):
	label1=modelfirst.predict(x)[0]
	if label1==1:
		return modelmisclass.predict(x)[0]
	else:
		return modelneeded.predict(x)[0]


preds=[predictor(x) for x in X_test]
print np.sum(preds==Y_test)/float(len(Y_test))

a=np.zeros(shape=(26,26))
c=0
print "a b c d e f g h i j k l m n o p q r s t u v w x y z"
for i in preds:
	a[Y_test[c]][i]+=1
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
pickle.dump(a,open("hier_hog.p","wb"))










# #First heirarchy
# onehand=[2,11,14,20,21,22]
# twohand=[i for i in range(0,26) if i not in onehand]
# Y_level1=np.asarray([1 if Y[i] in onehand else 2 for i in range(0,len(Y))])
# Y_train2=Y_level1[ind_train]
# Y_test2=Y_level1[ind_test]
# model2=linsvm()
# model2.fit(X_train,Y_train2)
# print np.sum(model2.predict(X_test)==Y_test2)/float(len(Y_test2))
# #Second heirarchy
# #One hand only

# ind_onehand=[i for i in range(0,len(Y_train)) if Y_train[i] in onehand]
# ind_twohand=[i for i in range(0,len(Y_train)) if Y_train[i] in twohand]
# X_train_onehand=X_train[ind_onehand]
# X_train_twohand=X_train[ind_twohand]
# Y_train_onehand=Y_train[ind_onehand]
# Y_train_twohand=Y_train[ind_twohand]
# #Testing one hand model
# ind_1_test=[i for i in range(0,len(Y_test)) if Y_test[i] in onehand]
# X_test_onehand=X_test[ind_1_test]
# Y_test_onehand=Y_test[ind_1_test]
# model4=linsvm()
# model4.fit(X_train_onehand,Y_train_onehand)
# print np.sum(model4.predict(X_test_onehand)==Y_test_onehand)/float(len(Y_test_onehand))



# #Testing for two hand model
# ind_2_test=[i for i in range(0,len(Y_test)) if Y_test[i] in twohand]
# X_test_twohand=X_test[ind_2_test]
# Y_test_twohand=Y_test[ind_2_test]
# model3=linsvm()
# model3.fit(X_train_twohand,Y_train_twohand)
# print np.sum(model3.predict(X_test_twohand)==Y_test_twohand)/float(len(Y_test_twohand))


# def predictor(x):
# 	label1=model2.predict(x)[0]
# 	if label1==1:
# 		return model4.predict(x)[0]
# 	else:
# 		return model3.predict(x)[0]