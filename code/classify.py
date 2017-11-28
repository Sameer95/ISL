import libsvm
import os
import argparse
from cPickle import load
from learn import extractSift, computeHistograms, writeHistogramsToFile
import pickle
import numpy as np

HISTOGRAMS_FILE = 'testdata.svm'
CODEBOOK_FILE = 'codebook.file'
MODEL_FILE = 'trainingdata.svm.model'


def parse_arguments():
    parser = argparse.ArgumentParser(description='classify images with a visual bag of words model')
    parser.add_argument('-c', help='path to the codebook file', required=False, default=CODEBOOK_FILE)
    parser.add_argument('-m', help='path to the model  file', required=False, default=MODEL_FILE)
    parser.add_argument('input_images', help='images to classify', nargs='+')
    args = parser.parse_args()
    return args


print "---------------------"
print "## extract Sift features"
all_files = []
all_files_labels = {}
all_features = {}

args = parse_arguments()
model_file = args.m
codebook_file = args.c
farr=[]
val=[]
fol = args.input_images
fl=os.listdir(fol[0])
fl.sort()
for f in fl:
    fnames=[fol[0]+'/'+f]
    farr.append(f[0])
    all_features = extractSift(fnames)
    for i in fnames:
        all_files_labels[i] = 0  # label is unknown

    print "---------------------"
    print "## loading codebook from " + codebook_file
    with open(codebook_file, 'rb') as f:
        codebook = load(f)

    print "---------------------"
    print "## computing visual word histograms"
    all_word_histgrams = {}
    for imagefname in all_features:
        word_histgram = computeHistograms(codebook, all_features[imagefname])
        all_word_histgrams[imagefname] = word_histgram

    print "---------------------"
    print "## write the histograms to file to pass it to the svm"
    nclusters = codebook.shape[0]
    writeHistogramsToFile(nclusters,
                          all_files_labels,
                          fnames,
                          all_word_histgrams,
                          HISTOGRAMS_FILE)

    print "---------------------"
    print "## test data with svm"
    hold=libsvm.test(HISTOGRAMS_FILE, model_file)
    vl=hold[0]
    val.append(vl)
    print hold

print farr
print val
cm=np.zeros(shape=(26,26))
# confusion matrix
ct=0
# dct={'a' : 0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10,'l':11,'m':12,'n':13,'o':14,'p':15,'q':16,'r':17,'s':18,'t':19,'u':20,'v':21,'w':22,'x':23,'y':24,'z':25}
dct=pickle.load(open("dict1.p","rb"))
for item in farr:
  # print item[0]
  c=ord(item[0])-97
  col=ord(dct.keys()[dct.values().index(val[ct])])-97
  cm[c][col]+=1 
  ct=ct+1

print cm
f=open('table.txt','wb')
f.write(' & ')
for i in range(65,90):
  f.write(chr(i)+' &')
f.write(r'Z \\ \hline'+'\n')
for i in range(0,26):
  f.write(chr(i+65)+' & ')
  for j in range(0,25):
    f.write(str(cm[i][j])+' & ')
  f.write(str(cm[i][25])+r'\\ \hline'+'\n')
f.close()