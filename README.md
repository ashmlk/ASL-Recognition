# ASL-Recognition
**American Sign Language Detection Program**
The project aims to build a machine learning model that can classify different hand gestures used for the English language. In this model, the classification of the model is built based on a training set of data independent of the testing set. Various machine learning algorithms were used which resulted in better results compared to others. For the image dataset, various algorithms were applied such as Support Vector Machines (SVM), Nearest Neighbors (k-NN), and Convolutional Neural Network (CNN). The purpose of this project was to find the best algorithms to achieve the highest accuracy in prediction. 

**The Dataset used is retrieved from [here](https://www.kaggle.com/datamunge/sign-language-mnist)**

***The project consists of the following algorithms and techniques***

## Principal component analysis(PCA)

As mentioned before, the dataset contains images that are 28x28, this will give us 784 dimensions which can cause major overfitting on the training dataset and therefore will dramatically decrease our prediction accuracy on the testing set. Therefore by using Principal Component Analysis (PCA) we are focusing on two major areas:
  - Future Extraction 
  - Future Elimination 
## Ensembles
Ensemble modeling is applying diverse models at the same time to predict an outcome, using different training sets or different models on the same training set. In our case we are using different models on our training sets and then using using ensembling models such bagging, boosting, and stacking to find the final outcome. 
The main idea here is to find a set of weak learners and then use an ensembling model to combine the weak learners to find the best outcome. We will describe the three main ensembling meta-algorithms

## Support vector machines
Support Vector Machines (SVM) is essentially a binary classifier that essentially used support vectors to measure the distance between two data points in a hyperplane (Since our class is not linear) and map the data into a higher dimension feature space which then the basic idea will be each separated data will belong to different classes and can be classified. The SVM will use two degrees of measurement. Hence we define terms functional margin and geometric margin. A functional margin tells you about the accuracy and correctness of classification of a point. Geometric margin on the other hand, is the scaled version of functional margin and tells us about the euclidean distance between the hyperplane and the data points.

## Histogram of oriented gradient features
“The histogram of oriented gradients (HoG) is a feature descriptor used in computer and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in localized portions of an image”. A feature descriptor is a representation of an image or an image patch that simplifies the image by extracting useful information and eliminating any extraneous features that may be redundant in identifying an image. A typical feature descriptor will transform a ‘width x height x 3 ( Three channels of RGB ) into an array/vector of length N. 

## K-Nearest-Neighbors
“The essential idea behind the HoG descriptors is that neighborhood object appearance and shape inside a picture can be depicted by circulation of edge bearings and power angles. That is each window area can be displayed by the adjacent scattering of the edge introductions and the relating angle size”. Setting K = 1 yields the closest neighbor order govern, as it can be seen in the below graph. Maybe the easiest and the most broadly utilized as a part of practice. The "separation" between two items is taken as the Euclidean distance between them. Using the HoG may be computationally costly, however, we can argue that HoG will give us the chance to determine the neighbors of every datapoint by a high probability, and therefore achieve the best results. They way that HoG works makes it easier for the kNN classifier to classify images and therefore getting better results. Using kNN and HoG features were able to achieve an accuracy of 92%. Furthermore, it is important to choose the best number of neighbors for training the kNN classifiers. Below it can be seen the tradeoff between the number of neighbors and the accuracy on both test and training data. 

## Convolutional neural networks
A convolutional neural networks (CNN) are a type of neural networks that apply a mathematical operation called ‘convolution’ which is a linear operation.  Neural networks have an input and an output which can both be binary and multiclass. Convolutional neural networks are mostly used for image classification and recognition. The convolutional neural network consists of many layers and ‘hidden’ layers that perform different operations on the input and will assign weights to inputs in every layer. The hidden layers of a CNN typically consist of a series of convolutional layers that convolve with a multiplication or other dot product. Neural networks will have an activation layers usually called rectified linear unit (ReLU) layers and finally a convolution layer that usually uses backpropagation to achieve better weights. The hidden layer weights are usually masked between the initial ReLU layer and the final layer.

 ### The input of the neural network
  
When we initially give the inputs to the convolutional neural network which is a tensor with shape (number of samples) x (image width) x (image height) x (image depth (channels)).  It is important to note that the channel of the input images is important since different frameworks require different channels for images which is usually RGB channel and therefore we needed to transform our images to RGB channel in order to feed it to neural network. Therefore in our case the tensor shape will be 64,574,190 in size. It is important as we are using a three dimensional RGB channel the convolution layers must also have a depth of three in order to operate on images. 

###  Pooling layers in our model
  
Using pooling layers in our model is an essential step in building a convolutional neural network. Pooling layers will reduce the dimensions of features and therefore the transition from one layer to another layers will be easier. Reducing the number of dimensions in a convolutional neural net is an essential step. As in passing the input from one layer to another requires weights to be adjusted based on every input and if the input is not deduced in dimension which will make the task extremely hard. Below is a representation of how the neural network changes the input by applying pooling layer.

In our model we use a 2 x 2 ‘max pooling’ layers to reduce the dimensions of inputs. Max pooling uses the maximum of the values in an N x N  window which in our case is 2 x 2. Most pooling layers are 2 x 2 since if we choose a bigger window size it can cause an extreme drop in number of features and unless our input image is relatively large and contains many noisy features, apply a pooling layer with a bigger window size may cause underfitting and lower the final accuracy of the model. 

 ### Layers in our model
  
The next and most important aspect of the neural network is the hidden layers in the model. The number of layers are responsible for convolving the input from every image and adjusting the weights. Our model consists of 6 layers in total, two of them are max-pool layers and the rest are convolutional layers. The layers layers build a features map of the neurons and the convolutional neural nets use a concept called parameter sharing. Each single neuron shares the weights from every layer.  Each filter is replicated across the entire visual field. These replicated units share the same parameterization (weight vector and bias) and form a feature map. This means that all the neurons in a given convolutional layer respond to the same feature within their specific response field. In our model we chose 2 layers with depth of 32 and then 2 layers with depth of 64. The size of all layers was 3x3 size the input image was 28x28 through testing various different sizes we can argue that picking a bigger window size for the convolutional layers can result in loss of data, therefore we decided to keep the window size small in order to preserve maximum data for pooling layer as well. It is also notable that in order to preserve more information about the input would require keeping the total number of activations (number of feature maps) non-decreasing therefore the initial layer which is the ReLU layer should have less layers which is the 64 in depth.

After the last layer we dropout half the parameters. Because fully connected layer occupies most of them it is prone to overfitting. One method to reduce it, is using a ‘dropout’  layer. In training stages the probability that a hidden node will dropout is 50%. However, for the initial input nodes the probability should be lower if not there is a problem with the model and the number of redundancies in the nodes can be relatively higher than the average expected model. By avoiding to train all the nodes on the input the dropout avoided the problems that could be caused by overfitting. Another advantage the the dropout layer provides is that it significantly improves the training speed since many nodes are dropped out and therefore during backpropagation the number of nodes that require adjustment decreases and therefore we will be able to train our model faster. Another important parameter to consider is ‘normalized data’. Normalization will add additional error, and therefore the level of acceptable model complexity will be reduced. Below you can see the parameters of the model after training the data: 


 ### Number of epochs and batch size
  
Through testing various numbers of epochs that were used in the model were 25, with batch size of 128 which we were able to achieve over 96% accuracy on our testing data ad 99.8% accuracy on the validation set. 


