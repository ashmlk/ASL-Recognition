# -*- coding: utf-8 -*-
"""asl_recognition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/172hP1_dVsQfY2oIwBlDinbKQWAgGJCrB
"""

from google.colab import drive
drive.mount('/content/gdrive')

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import PIL 

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as ooptim
import torch.nn.functional as F
from torch.autograd import Variable

import os
import warnings
warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount('/content/drive')

#Lets load our data
train = pd.read_csv('/content/drive/My Drive/data/sign_mnist_train.csv')
test = pd.read_csv('/content/drive/My Drive/data/sign_mnist_test.csv')

print("Training data shape: ", train.shape)
print("Testing data shape: ", test.shape)

#View a sample of data
train.head()

x_train = np.array(train.iloc[:,1:])
x_train_img = np.array([np.reshape(i, (28,28)) for i in x_train])
x_test = np.array(test.iloc[:,1:])
x_test_img = np.array([np.reshape(i, (28,28)) for i in x_test])

num_classes = 26
train_label = np.array(train.iloc[:,0]).reshape(-1)
test_label = np.array(test.iloc[:,0]).reshape(-1)

X_train_img = x_train.reshape((27455, 28, 28, 1))
X_test_img = x_test.reshape((7172, 28, 28, 1))

#Transfering data to one-hot labels
y_train = np.eye(num_classes)[train_label]
y_test = np.eye(num_classes)[test_label]

#Viewing samples of images
fig, axes = plt.subplots(2,4, figsize=(8,5),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
  ax.imshow(x_train_img[i],cmap='gray')

"""#Principal Component Analysis (PCA)"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

pca = PCA(n_components=.99)
x_pca = pca.fit_transform(x_train_scaled)

variance = pca.explained_variance_ratio_ #calculate variance ratios
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(30,100.5)
plt.style.context('seaborn-whitegrid')
plt.plot(var)

#We can se that at ~60 components over 90% variance is captured and after is nearly constant
pca = PCA(n_components=60)
x_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(28, 28), cmap='gray')

"""#Ensembles and classification models

We first use PCA components for four different classifiers(decision tree, kNN, Logistic regression and Random forest classifier) and then we use three different ensemble methods, bagging, boosting and stacking to compare the results.

##Bagging
"""

import matplotlib.gridspec as gridspec
import itertools

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score, train_test_split

from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions

"""###Decision Trees"""

#Lets train sample decision tree with default values
dt0 = DecisionTreeClassifier()
dt0.fit(x_pca, train_label)

y_pred = dt0.predict(x_test_pca)

from sklearn.metrics import accuracy_score
acc = accuracy_score(test_label,y_pred)
acc

max_depths = np.linspace(1, 26, 26, endpoint=True)

train_results = []
test_results = []

for max_depth in max_depths:
   dt = DecisionTreeClassifier(max_depth=max_depth)
   dt.fit(x_pca, train_label)

   train_pred = dt.predict(x_pca)
   acc = accuracy_score(train_label,train_pred)
   train_results.append(acc)

   y_pred = dt.predict(x_test_pca)
   acc = accuracy_score(test_label,y_pred)
   test_results.append(acc)
  
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label="Train Score")
line2, = plt.plot(max_depths, test_results, 'r', label="Test Score")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy score')
plt.xlabel('Tree depth')
plt.show()

"""We can see that the dept of 20 is the best score for our decision tree so we set that to our max_depth

###k-Nearest-Neighbors
"""

#Train knn with only 1-nearest-neighbor
knn0 = KNeighborsClassifier(n_neighbors=1)
knn0.fit(x_pca, train_label)

y_pred = knn0.predict(x_test_pca)

acc = accuracy_score(test_label,y_pred)
acc

n_neighbors = np.linspace(1, 10, 10, endpoint=True,dtype=np.int64)

train_results = []
test_results = []

for k in n_neighbors:
   knn = KNeighborsClassifier(n_neighbors=k)
   knn.fit(x_pca, train_label)

   train_pred = knn.predict(x_pca)
   acc = accuracy_score(train_label,train_pred)
   train_results.append(acc)

   y_pred = knn.predict(x_test_pca)
   acc = accuracy_score(test_label,y_pred)
   test_results.append(acc)
  
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_neighbors, train_results, 'b', label="Train Score")
line2, = plt.plot(n_neighbors, test_results, 'r', label="Test Score")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy score')
plt.xlabel('Number of neighbors')
plt.show()

"""We can see that setting 1 as 'k' we acieve the best score for our decision tree so we set that to our n_neighbors.

###Bagging all models
"""

dt_f = DecisionTreeClassifier(max_depth=20)
knn_f = KNeighborsClassifier(n_neighbors=1)
lr_f = LogisticRegression()

bagging_dt = BaggingClassifier(base_estimator= dt_f , n_estimators=10, max_samples=0.7, max_features=0.9)
bagging_knn = BaggingClassifier(base_estimator= knn_f, n_estimators=10, max_samples=0.7, max_features=0.9)
bagging_lr = BaggingClassifier(base_estimator= lr_f, n_estimators=10, max_samples=0.7, max_features=0.9)

label = ['Decision Tree', 'K-NN', 'Logistic Regression', 'Bagging Tree', 'Bagging K-NN', 'Bagging Logistic Regression']
classifiers = [dt_f, knn_f, lr_f, bagging_dt, bagging_knn,bagging_lr]

for clf, label in zip(classifiers, label):        
    scores = cross_val_score(clf, x_pca, train_label, cv=3, scoring='accuracy')
    print("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))

"""###Random Forest Classification

Above we have seen all the cross validation scores for single classifiers and using bagging as an ensemble method
"""

#Now lets see the effect of max_samples meaning the effect of subsampling the data
bags = [bagging_dt, bagging_knn, bagging_lr]
x_train0, x_test0, y_train0, y_test0 = train_test_split(x_pca, train_label, test_size=0.3, random_state=7)
for b in bags:
 plt.figure()
 plot_learning_curves(x_train0, y_train0, X_test0, y_test0, b, print_model=False, style='ggplot')
 plt.show()

"""Tables are for  'Bagging Tree', 'Bagging K-NN' and 'Bagging Logistic Regression' respectively. 
As we can see in all tables choosing ~80% data as training data we achieve the best ensemble models.
"""

from sklearn.model_selection import GridSearchCV

param_grid = {
    'bootstrap': [True],
    'max_depth': [10,20,30,50],
    'max_features': [0.8, 0.9],
    'n_estimators': [10,20,50,100]
}
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(x_pca, train_label)
grid_search.best_params_

best_grid = grid_search.best_estimator_
accuracy = best_grid.score(x_test_pca, test_label)
print("Accuracy score for Random Forest Classification is: %.2f" %accuracy)

"""##Boosting"""

import matplotlib.gridspec as gridspec
import itertools

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, train_test_split

from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions

dt_f = DecisionTreeClassifier(max_depth=20)
boosting_dt = AdaBoostClassifier(base_estimator= dt_f , n_estimators=10)

label = ['Decision Tree','Boosting Tree']
classifiers = [dt_f, boosting_dt]

for clf, label in zip(classifiers, label):        
    scores = cross_val_score(clf, x_pca, train_label, scoring='accuracy')
    print("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))

"""Now we will plot the error based on the number of subsamples from the decision tree and plot the results"""

x_train0, x_test0, y_train0, y_test0 = train_test_split(x_pca, train_label, test_size=0.3, random_state=7)

plt.figure()
plot_learning_curves(x_train0, y_train0, x_test0, y_test0, boosting_dt, print_model=False, style='ggplot')
plt.show()

"""We will now see the effect of number of estimators on our final result, we use the standard deviation as the identifier of the error for each number estimators."""

num_est = map(int, np.linspace(1,100,20))
boosting_mean = []
boosting_std = []
for n_est in num_est:
    boosting = AdaBoostClassifier(base_estimator=clf, n_estimators=n_est)
    scores = cross_val_score(boosting, x_pca, train_label, cv=3, scoring='accuracy')
    boosting_mean.append(scores.mean())
    boosting_std.append(scores.std())

num_est = np.linspace(1,100,20).astype(int)
plt.figure()
plt.errorbar(num_est, boosting_mean, yerr=boosting_std, uplims=True, lolims=True, fmt='-o', capsize=5,
             marker='.',mfc='purple', mec='yellow',
             label='uplims=True, lolims=True')
plt.ylabel('Accuracy'); plt.xlabel('Ensemble Size'); plt.title('AdaBoost Ensemble');
plt.show()

"""##Stacking"""

import matplotlib.gridspec as gridspec
import itertools

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, train_test_split

from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions

"""Now lets stack the models and using logistic regression as our meta classifier to see the results"""

knn = KNeighborsClassifier(n_neighbors=1)
rf = RandomForestClassifier(random_state=1)
gbc = GaussianNB()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[knn, rf, gbc], 
                          meta_classifier=lr)

label = ['KNN', 'Random Forest', 'Naive Bayes', 'Stacking Classifier']
clf_list = [knn, rf, gbc, sclf]

stacking_mean = []
stacking_std = []
for clf, label in zip(clf_list, label):    
    scores = cross_val_score(clf, x_pca, train_label, cv=3, scoring='accuracy')
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
    stacking_mean.append(scores.mean())
    stacking_std.append(scores.std())

"""Now lets plot the accuracy of each model and the possible error in each"""

plt.figure()
plt.errorbar(range(4), stacking_mean, yerr=stacking_std, uplims=True, lolims=True, fmt='-o', capsize=3,
             marker='.',mfc='purple', mec='yellow',
             label='uplims=True, lolims=True')
plt.xticks(range(4), ['KNN', 'RF', 'NB', 'Stacking'])        
plt.ylabel('Accuracy'); plt.xlabel('Classifier'); plt.title('Stacking Ensemble Model');
plt.show()

x_train0, x_test0, y_train0, y_test0 = train_test_split(x_pca, train_label, test_size=0.3, random_state=7)

plt.figure()
plot_learning_curves(x_train0, y_train0, x_test0, y_test0, sclf, print_model=False, style='ggplot')
plt.show()

"""We can seee that as number of training samples goes highere due to high variance of data the accuracy of model drasrically drops, however we can see that in Stacking we do not see any over fitting.

###Final Predictions
"""

knn = KNeighborsClassifier(n_neighbors=1)
dt = DecisionTreeClassifier(max_depth=20)
gbc = GaussianNB()
lr = LogisticRegression()
bagging_dt = BaggingClassifier(base_estimator= dt , n_estimators=10, max_samples=0.8, max_features=0.9)
bagging_knn = BaggingClassifier(base_estimator= knn, n_estimators=10, max_samples=0.8, max_features=0.9)
bagging_lr = BaggingClassifier(base_estimator= lr, n_estimators=10, max_samples=0.8, max_features=0.9)
boosting_dt = AdaBoostClassifier(base_estimator= dt_f , n_estimators=10)
sclf = StackingClassifier(classifiers=[knn, rf, gbc], meta_classifier=lr)

from sklearn.metrics import accuracy_score

classifiers_unsupervised = [knn, dt, bagging_dt,bagging_knn, bagging_lr,  boosting_dt, sclf]
classifiers_names1 = ['K-NN', 'Decision Tree', 'Bagging Tree',
                     'Bagging K-NN', 'Bagging Logistic Regression', 'Boosting tree', 'Stacking Model']
for clf, n in zip(classifiers_unsupervised, classifiers_names1):
  clf.fit(x_pca, train_label)
  pred = clf.predict(x_test_pca)
  accuracy = accuracy_score(pred, test_label)
  print("Accuracy score for %s is: %.2f" %(n, accuracy))

classifiers_supervised  = [gbc, lr]
classifiers_names2 = ['Naive Bayes', 'Logistic Regression']

for clf2, n in zip(classifiers_supervised, classifiers_names2):
  clf2.fit(x_pca, train_label)
  pred = clf2.predict(x_test_pca)
  accuracy = accuracy_score(pred, test_label)
  print("Accuracy score for %s is: %.2f" %(n, accuracy))

"""#Support Vector Machines

##Support Vector Machines (SVM) + PCA
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100]},
                    {'kernel': ['linear'], 'C': [1, 10, 100]}]
svm = GridSearchCV(SVC(probability=True), tuned_parameters, refit=True, verbose=1)
svm.fit(x_pca, train_label)
print(svm.best_params_)
print(svm.best_estimator_)

from sklearn.svm import SVC

svm = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
    verbose=False)

#Training SVM
svm.fit(x_pca,train_label)
pred=svm.predict(x_test_pca)
accuracy = accuracy_score(pred, test_label)
print("Accuracy score for Support Vector Machine is: %.2f" %accuracy)

#Graph confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
labels = np.linspace(0, 25, 25, endpoint=True,dtype=np.int64)
cm = confusion_matrix(test_label, pred)
# Normalize
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)

"""##Exctracting Hog Features"""

from skimage.feature import hog
from skimage import data, color, exposure
hogtrain = []
hogimg = []
for i in range(x_train_img.shape[0]):
  fd, hog_image = hog(x_train_img[i], orientations=8, pixels_per_cell=(8,8),cells_per_block=(2,2), visualise=True)
  hogtrain.append(fd)
  hogimg.append(hog_image)

hog_feature = np.array(hogtrain)

hogtest = []
hogimg_test = []
for i in range(x_test_img.shape[0]):
  fd, hog_image = hog(x_test_img[i], orientations=8, pixels_per_cell=(8,8),cells_per_block=(2,2), visualise=True)
  hogtest.append(fd)
  hogimg_test.append(hog_image)

hog_feature_test = np.array(hogtest)

fig, axes = plt.subplots(2, 2, figsize=(11, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(hogimg[i], cmap='gray')

"""##Support Vector Machines (SVM) + HOG"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100]}]
svm2 = GridSearchCV(SVC(probability=True), tuned_parameters, refit=True, verbose=1)
svm2.fit(hog_feature, train_label)
print(svm2.best_params_)
print(svm2.best_estimator_)

from sklearn.metrics import accuracy_score

svm2.fit(hog_feature,train_label)
pred = svm2.predict(hog_feature_test)
accuracy = accuracy_score(pred, test_label)
print("Accuracy score for Support Vector Machine using HOG features is: %.2f" %accuracy)

#Graph confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

labels = np.linspace(0, 25, 25, endpoint=True,dtype=np.int64)
cm = confusion_matrix(test_label, pred)
# Normalize
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)

"""##Support Vector Machines (SVM) + HOG + PCA"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca_hogs = PCA(n_components=.99)
x_hogs_pca = pca_hogs.fit_transform(hog_feature)
x_hogs_pca_test = pca_hogs.transform(hog_feature_test)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svm2 = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
    verbose=False)
svm2.fit(x_hogs_pca,train_label)
pred = svm2.predict(x_hogs_pca_test)
accuracy = accuracy_score(pred, test_label)
print("Accuracy score for Support Vector Machine by applying PCA on HOG features is: %.2f" %accuracy)

#Graph confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

labels = np.linspace(0, 25, 25, endpoint=True,dtype=np.int64)
cm = confusion_matrix(test_label, pred)
# Normalize
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)

"""#k-Nearest-Neighbors + HOG"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

n_neighbors = np.linspace(1, 10, 10, endpoint=True,dtype=np.int64)

train_results = []
test_results = []

for k in n_neighbors:
   knn = KNeighborsClassifier(n_neighbors=k)
   knn.fit(hog_feature, train_label)

   train_pred = knn.predict(hog_feature)
   acc = accuracy_score(train_label,train_pred)
   train_results.append(acc)

   y_pred = knn.predict(hog_feature_test)
   acc = accuracy_score(test_label,y_pred)
   test_results.append(acc)
  
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_neighbors, train_results, 'b', label="Train Score")
line2, = plt.plot(n_neighbors, test_results, 'r', label="Test Score")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy score')
plt.xlabel('Number of neighbors')
plt.show()

"""We can observe that 3 neighbors performs the best therefore in our final model we will set 'K' to three."""

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(hog_feature, train_label)

pred = knn.predict(hog_feature_test)
acc = accuracy_score(test_label, pred)

print("Accuracy score for Support Vector Machine by using HOG features is: %.2f" %acc)

"""#Stacking Support Vectors Machine and kNN using HOG features"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=3)
svm2 = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
    verbose=False)
lr = LogisticRegression()

sclf = StackingClassifier(classifiers=[knn,svm2], 
                          meta_classifier=lr)

scores = cross_val_score(sclf, hog_feature, train_label, cv=3, scoring='accuracy')
print ("Accuracy: %.2f (+/- %.2f) [Stacking Classifier]" %(scores.mean(), scores.std()))

sclf.fit(hog_feature, train_label)
pred = sclf.predict(hog_feature_test)
accuracy = accuracy_score(pred, test_label)
print("Accuracy score for Stacking Classifier is: %.2f" %accuracy)

"""#Convolutional Neural Network (CNN)"""

#For CNN we need RGB images lets transform our images in 3D dimension
from skimage.color import gray2rgb

x_train_cnn = gray2rgb(x_train_img)
x_test_cnn = gray2rgb(x_test_img)

#Normalizing the pixels
x_train_cnn = x_train_cnn / 255
x_test_cnn = x_test_cnn / 255

#Lets view some samples
fig, axes = plt.subplots(2, 4, figsize=(7, 3),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train_cnn[i])

#For CNN we need One-hot labels which are stored in y_train nd y_test
y_train[0]

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import warnings
warnings.filterwarnings("ignore")

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(2000, activation='relu'))
model.add(Dense(26, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Lets check available gpu
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

hist = model.fit(x_train_cnn, y_train,
           batch_size=128, epochs=25, validation_split=0.3 )

"""We achieved an accuracy of 99.8% over the training data"""

model.summary()

model.evaluate(x_test_cnn, y_test)[1]

"""The model is around 96% on test data"""

#Visualize the models accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

#Visualize the models loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

"""#Saving the trained model"""

#Save the model
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# 1. Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# 2. Save Keras Model or weights on google drive

# create on Colab directory
model.save('model.h5')    
model_file = drive.CreateFile({'title' : 'model.h5'})
model_file.SetContentFile('model.h5')
model_file.Upload()

# download to google drive
drive.CreateFile({'id': model_file.get('id')})

#Save the model weights
model.save_weights('model_weights.h5')
weights_file = drive.CreateFile({'title' : 'model_weights.h5'})
weights_file.SetContentFile('model_weights.h5')
weights_file.Upload()
drive.CreateFile({'id': weights_file.get('id')})

# 3. reload weights from google drive into the model

# use (get shareable link) to get file id
last_weight_file = drive.CreateFile({'id': '1N6JlAflv2fIrXFVZ14DPK0CEPicpQyVd'}) 
last_weight_file.GetContentFile('last_weights.mat')
model.load_weights('last_weights.mat')

"""#Capturing live images and predicting the output

The code below is used for using the local webcam as this device is using a virtual machine to run the webcam set to the Jupyter notebook is not the local webcam. The code is retrieved from Google Colab team from:  https://colab.research.google.com/notebooks/snippets/advanced_outputs.ipynb#scrollTo=2viqYx97hPMi
"""

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename

from IPython.display import Image
try:
  filename = take_photo()
  print('Saved to {}'.format('image1.jpg'))
  
  # Show the image which was just taken.
  display(Image('photo.jpg'))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))

import cv2 as cv

img = cv.imread('/content/photo.jpg', cv.IMREAD_UNCHANGED)
# to crop required part
im2 = img
# convert to grayscale    
image_grayscale = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
# blurring the image 
image_grayscale_blurred =cv.GaussianBlur(image_grayscale, (15,15), 0)
# resize the image to 28x28
im3 = cv.resize(image_grayscale_blurred, (28,28), interpolation = cv.INTER_AREA)
# expand the dimensions from 28x28 to 1x28x28x1
im4 = np.resize(im3, (28, 28, 3))
im5 = np.expand_dims(im4, axis=0)

data = np.asarray( im4, dtype="int32" )
pred_probab = model.predict(data)[0]
# softmax gives probability for all the alphabets hence we have to choose the maximum probability alphabet 
pred_class = list(pred_probab).index(max(pred_probab))
max(pred_probab), getLetter(pred_class)

"""The Three cells above are just for testing therefore the outputs have been removed, for demo you can run the blocks in order in order to see the final result predicted by the neural network."""

'''This method will simply change the 
final result to letters since the nueral net will predict indexes of teh alphabet
'''
def getLetter(i):
  alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  return alphabet[i]