from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
import glob
import os
from config import *
import numpy as np
import pickle
import sys

def train_svm():
    pos_feat_path = '../data/features/pos'
    neg_feat_path = '../data/features/neg'

    # Classifiers supported
    clf_type = sys.argv[1]
    print clf_type
    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)

    print np.array(fds).shape,len(labels)
    '''
    if clf_type=="SVM_Regressor":
        print "Training a Linear SVM Regression model"
	clf = svm.SVR(cache_size=7000).fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        pickle.dump(clf, open("%sSVM_Regressor" % model_path,'wb'))
        print "Classifier saved to {}".format(model_path)

    if clf_type=="GBRT":
        print "Training a GBRT model"
	est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=1, random_state=0, loss='ls').fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        pickle.dump(est, open("%sGBRT" % model_path,'wb'))
        print "Classifier saved to {}".format(model_path)
    '''
    if clf_type=="SVM_Classifier":
        clf = SVC(kernel="linear", C=0.025)
        print "Training a Linear SVM Classifier"
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        pickle.dump(clf, open("%sSVM_Classifier" % model_path,'wb'))
        print "Classifier saved to {}".format(model_path)

    if clf_type=="RBF_SVM":
        clf = SVC(gamma=2, C=1)
        print "Training a RBF SVM Classifier"
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        pickle.dump(clf, open("%sRBF_SVM" % model_path,'wb'))
        print "Classifier saved to {}".format(model_path)

    if clf_type=="K_Neighbor":
        print "Training a K-Neighbour Classifier"
	clf = neighbors.KNeighborsClassifier()
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        pickle.dump(clf, open("%sK_Neighbor" % model_path,'wb'))
        print "Classifier saved to {}".format(model_path)

    if clf_type=="Gaussian":
        print "Training a Gaussian Boost Classifier"
	clf = GaussianProcessClassifier(1.0 * RBF(1.0))
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        pickle.dump(clf, open("%sGaussian" % model_path,'wb'))
        print "Classifier saved to {}".format(model_path)

    if clf_type=="Ada":
        print "Training a ADA Boost Classifier"
	clf = AdaBoostClassifier()
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        pickle.dump(clf, open("%sAda" % model_path,'wb'))
        print "Classifier saved to {}".format(model_path)

    if clf_type=="RandomForest":
        print "Training a Random Forest Classifier"
	clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=2)
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        pickle.dump(clf, open("%sRandomForest" % model_path,'wb'))
        print "Classifier saved to {}".format(model_path)


train_svm()