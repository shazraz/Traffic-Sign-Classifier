# **Traffic Sign Classification** 

[//]: # (Image References)
[image1]: ./graphics/DataHistogram.png "Visualization"
[image2]: ./graphics/10-image-set.png "31 - Wild Animals Crossing"
[image3]: ./graphics/y-flipped_image.png "Image Mirrored along Y-axis"
[image4]: ./graphics/AugDataHistogram.png "Data Distribution after Basic Augmentation"
[image5]: ./graphics/augmentation.png "Augmentation via Translation/Rotation"
[image6]: ./graphics/FinalDataHistogram.png "Data Distribution after Final Augmentation"
[image7]: ./graphics/image-processing.png "Image Processing Results"
[image8]: ./graphics/accuracy-loss-final.png "Accuracy, Loss Trends"
[image9]: ./graphics/softmax.png "Top 5 Probabilities"
[image10]: ./graphics/conv1.png "Conv1 Feature Maps"
[image11]: ./graphics/conv2.png "Conv2 Feature Maps"

---
**1. Repository Files**

This repo consists of:

1. The IPython notebook containing the project code: [Traffic_Sign_Classifier.ipynb](https://github.com/shazraz/P2-TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)
1. The trained and saved model: LeNet5_Run3
1. A set of images downloaded from the internet used to test the model: [Images](https://github.com/shazraz/P2-TrafficSignClassifier/tree/master/Images)
 {credit: Sonja Krause-Harder for posting some of these images on the p-traffic-signs Slack channel}

**2. Data Set Summary & Exploration**

*2.1 Data Set Summary*

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799 images
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

*2.2 Exploratory Visualization of the dataset*

Here is an exploratory visualization of the data set. The histogram below shows the relative distribution of each of the three data splits (training, validation & test) for each of the 43 labels in the data. A number of the labels aren't adequately represented and training a model on this dataset without any augmentation will result in a model biased towards the over-represented labels. 

![alt text][image1]

In addtion, let's take a look at some of the images in the dataset. The following image shows a plot of 10 sequential images starting randomly somewhere in the training data set. The images are from Label 31 - Wild Animals Crossing and appear to be poorly illuminated in some instances and some pre-processing will be required to produce better results when training the model.

![alt text][image2]

**3. Design and Test a Model Architecture**

*3.1 Preprocessing*

3.1.1 Data Augmentation

The first step was to experiment with simple data augmentation using the basic_augment() function. This function is fed with the training data set (X_train, y_train) along with a list of labels that can be mirrored in the x-axis and/or y-axis as well as rotated. By limiting the labels augmented to those that are under-represented in the data set, we can quickly obtain additional images from the existing data set. The image below shows an example of a 38 - Keep Right image that can be mirrored along the Y-axis to create an image with label 39 - Keep left which is an under-represented label.

![alt text][image3]

The following table provides an analysis of the labels that can be augmented in this method to create additional images in either the same class or other classes.

| ClassId | SignName                                           | Under-Represented? | Flip-Y | Flip-X | Rot 180 | Rot 120 | New Class |
|---------|----------------------------------------------------|--------------------|--------|--------|---------|---------|-----------|
| 0       | Speed limit (20km/h)                               | Y                  |        |        |         |         |           |
| 1       | Speed limit (30km/h)                               |                    |        | X      |         |         | 1         |
| 2       | Speed limit (50km/h)                               |                    |        |        |         |         |           |
| 3       | Speed limit (60km/h)                               |                    |        |        |         |         |           |
| 4       | Speed limit (70km/h)                               |                    |        |        |         |         |           |
| 5       | Speed limit (80km/h)                               |                    |        | X      |         |         | 5         |
| 6       | End of speed limit (80km/h)                        | Y                  |        |        |         |         |           |
| 7       | Speed limit (100km/h)                              |                    |        |        |         |         |           |
| 8       | Speed limit (120km/h)                              |                    |        |        |         |         |           |
| 9       | No passing                                         |                    |        |        |         |         |           |
| 10      | No passing for vehicles over 3.5 metric tons       |                    |        |        |         |         |           |
| 11      | Right-of-way at the next intersection              |                    |        |        |         |         |           |
| 12      | Priority road                                      |                    |        |        |         |         |           |
| 13      | Yield                                              |                    |        |        |         |         |           |
| 14      | Stop                                               |                    |        |        |         |         |           |
| 15      | No vehicles                                        |                    |        |        |         |         |           |
| 16      | Vehicles over 3.5 metric tons prohibited           | Y                  |        |        |         |         |           |
| 17      | No entry                                           |                    | X      | X      | X       |         | 17        |
| 18      | General caution                                    |                    | X      |        |         |         | 18        |
| 19      | Dangerous curve to the left                        | Y                  | X      |        |         |         | 20        |
| 20      | Dangerous curve to the right                       | Y                  | X      |        |         |         | 19        |
| 21      | Double curve                                       | Y                  |        |        |         |         |           |
| 22      | Bumpy road                                         | Y                  | X      |        |         |         | 22        |
| 23      | Slippery road                                      | Y                  |        |        |         |         |           |
| 24      | Road narrows on the right                          | Y                  |        |        |         |         |           |
| 25      | Road work                                          |                    |        |        |         |         |           |
| 26      | Traffic signals                                    | Y                  | X      |        |         |         | 26        |
| 27      | Pedestrians                                        | Y                  |        |        |         |         |           |
| 28      | Children crossing                                  | Y                  |        |        |         |         |           |
| 29      | Bicycles crossing                                  | Y                  |        |        |         |         |           |
| 30      | Beware of ice/snow                                 | Y                  | X      |        |         |         | 30        |
| 31      | Wild animals crossing                              |                    |        |        |         |         |           |
| 32      | End of all speed and passing limits                | Y                  |        |        | X       |         | 32        |
| 33      | Turn right ahead                                   |                    | X      |        |         |         | 34        |
| 34      | Turn left ahead                                    | Y                  | X      |        |         |         | 33        |
| 35      | Ahead only                                         |                    | X      |        |         |         | 35        |
| 36      | Go straight or right                               | Y                  | X      |        |         |         | 37        |
| 37      | Go straight or left                                | Y                  | X      |        |         |         | 36        |
| 38      | Keep right                                         |                    | X      |        |         |         | 39        |
| 39      | Keep left                                          | Y                  | X      |        |         |         | 38        |
| 40      | Roundabout mandatory                               | Y                  |        |        |         | X       | 40        |
| 41      | End of no passing                                  | Y                  |        |        |         |         |           |
| 42      | End of no passing by vehicles over 3.5 metric tons | Y                  |        |        |         |         |           |

The following histogram shows a result of basic augmentation with a number of under-represented labels being passed in for augmentation. The size of the new training set is now 41458 images compared to the earlier 34799 images. It is seen that while some labels (i.e. 17, 26, 33, 34, 39) are somewhat more represented, additional augmentation is required.

![alt text][image4]

Consequently, minor random pertubations were applied to the training set images to further balance the dataset. This consisted of translating the images along the both the x and y axes by a random amount within a fixed range or rotating the images about their center by a random amount within a fixed range. This was accomplished by the use of the augment_set() function which was passed a number of parameters including the dataset of images and labels, the labels within the dataset to augment, the minimum threshold quantity of each label and the ranges for the translation and rotation when augmenting the dataset. This led to the introduction of the following hyper-parameters within the model and their corresponding values used for augmentation:

* n_threshold: 1500,  threshold quantity of images within each label
* rt_range: 15, range of rotation for each image
* xlate_range: 5, range of (x,y) translation for each image

The figure below shows a series of augmented images created from a single original image. Each image is a minor perturbation of the original image.

![alt text][image5]

Once the augmentation was complete, the dataset was much more balanced and ready for further processing prior to use in training. This is shown in the histogram below:

![alt text][image6]

This augmentation allows us to extend the training set from an original size of 34799 images to 68490 images which is approximately a two-fold increase.

It is worth mentioning that randomly augmenting the brightness of the image was also experimented with but this seemed to have a detrimental effect on the observed image quality. In particular, artifacts were observed in the image, therefore brightness augmentation was not done.

3.1.2 Image Processing

One the dataset is balanced, the images now need to be further processed before used for training. This image processing consists of a number of steps:

1. Conversion of images to grayscale using OpenCV to allow the model to train on traffic sign features
1. Image histogram equalization using the OpenCV CLAHE (Constrast-Limited Adaptive Histogram Equalization) to improve the illumination in the images. The poor brightness across some images can be seen in the data set visualization presented earlier in Section 2.2 Conversely, there are images included in the dataset that are over-exposed and also need to be normalized. This equalization is carried out on both the RGB images and grayscale images independently. The use of the CLAHE algorithm was based on the OpenCV [documentation](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) and introduces two additional hyper-parameters:
     * Tile Grid Size: 4x4, defines the # of tiles the image is divided into prior to equalization
     * Clip Limit: 2.0, defines the upper contrast limit of tiles to prevent noise amplification 
1. The (32,32,1) equalized grayscale images and (32,32,3) equalized RGB images are then merged to create a combined image of dimensions (32,32,4). This is done to provide the model with both the grayscale features as well as the color information embedded within the image which provides additional information for the classifier to train on.
1. Finally, the 4-channel images are normalized by subtracting the mean and  dividing by the standard deviation of the entire training data split. It is important to note that the validation and test data splits are also normalized using the mean and std dev of the training data split to maintain consistency in pre-processing.

The following image shows a 17 - No Entry sign at index 35242 of the final augmented training set that has been grayscaled and equalized.

![alt text][image7]

This final augmented & processed training data set is now ready to be fed into the model for training.

*3.2 Model Architecture*

During the course of investigating alternative models for the potential architecture, a number of additional well-reknowned architectures were examined from the list of ILSVRC winners over the past years (e.g. AlexNet, VGG, GoogLeNet, ResNet, etc.). However, this traffic sign classifier model was based on the LeNet-5 architecture examined in the Udacity course lectures. The decision was based on implementing a known, simple architecture that could be trained easily on limited computing resources to see it's effectiveness.

The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x4 gray-RGB image   							| 
| Convolution L1 5x5     	| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU	Activation				|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x10 				|
| Convolution L2 5x5	    | 1x1 stride, valid padding, outputs 10x10x20      									|
| RELU Activation		|         									|
| Max pooling				| 2x2 stride, valid padding, outputs 5x5x20        									|
|	Fully Connected Layer L3					| Input flattened 500 dims from previous max pooling layer, outputs 120												|
| RELU Activation						|												|
| Dropout						|	 											|
| Fully Connected Layer L4						|	outputs 84											|
| RELU Activation						|												|
| Dropout						|	 											|
| Fully Connected Layer L5						|	outputs 43											|

The layers are based on the LeNet lab exercise from the Udacity course material with the input and output dimensions adjusted for the merged gray-RGB images being passed into the model. The model also includes two dropout layers after the first two fully connected layers to improve the generalization performance.

*3.3 Model Training*

The LeNet-5 architecture was initially trained using an Adam optimizer with the following hyper-parameters on a dataset with basic augmentation and only grayscale images. The starting weights and biases were initialized using the tensorflow truncated normal function with an average of 0 and a std dev of 0.1.

   * Batch Size: 128
   * Learning Rate: 0.001
   * Epochs: 10
   * Keep Probability (for Dropout): 0.7

The training was conducted on an Intel i7-5600 series CPU which took approximately 10-20 minutes depending on the size of the dataset (augmented vs un-augmented) and number of epochs. A discussion of training approach is provided below.

*3.4 Solution Approach*

The initial training run already revealed a validation accuracy just north of the required 93%. However, it was felt that this could be significantly improved since many of the methodologies recommended in the lectures had not been implemented. As a result, several concepts were introduced into the training model one at a time to observe their effect on the overall validation accuracy. The steps roughly followed the proceeding order:

   * Increase # of epochs
   * Decrease learning rate
   * Adjust image equalization hyper-parameters
   * Convert dataset from grayscale images to 4-channel grayscale/RGB images (i.e. increasing # of model parameters)
   * Apply augmentation using pertubations of original images
   * Adjust Keep Probability used in Dropout layers
   * Introduce L2 norm regularization and experiment with values of beta
 
At each step, a single hyper-parameter was adjusted to determine a general trend in the performance of the model. The accuracy and loss trends were observed to make a decision on which parameter to adjust next. e.g. once overfitting was observed, the generalization performance of the model was improved by augmenting the data by different amounts by adjusting n_threshold, adjusting the dropout probability and varying the severity of the pertubations with rt_range and xlate_range one at a time and observing the effects on the accuracy/loss plots. 

The accuracy/loss plots were generated by modifying the evaluate() function to return both accuracy and loss and saving them an in array during training for both the training and validation datasets.

In total, each of the hyper-parameters below were adjusted until the provided listed final values were obtained:

   * Batch Size: 128
   * Epochs: 15
   * Learning Rate: 0.0005
   * Keep Probability: 0.5
   * Beta: 0.001
   * n_threshold: 1500
   * rt_range: 15
   * xlate_range: 5

The figure below shows the accuracy loss trends for this run:

![alt text][image8]

These resulted in the following final values in the last epoch:

   * training set accuracy of 99.6%
   * validation set accuracy of 97.3% 
   * test set accuracy of 96.4%
   
The following table provides the precision and recall over the test data for each label in this model:

| Label | Label name                                         | # of samples | Precision | Recall  |
|-------|----------------------------------------------------|--------------|-----------|---------|
| 0     | Speed limit (20km/h)                               | 60           | 95.08%    | 96.67%  |
| 1     | Speed limit (30km/h)                               | 720          | 92.76%    | 97.92%  |
| 2     | Speed limit (50km/h)                               | 750          | 95.24%    | 98.67%  |
| 3     | Speed limit (60km/h)                               | 450          | 95.70%    | 94.00%  |
| 4     | Speed limit (70km/h)                               | 660          | 98.46%    | 96.67%  |
| 5     | Speed limit (80km/h)                               | 630          | 94.92%    | 89.05%  |
| 6     | End of speed limit (80km/h)                        | 150          | 98.63%    | 96.00%  |
| 7     | Speed limit (100km/h)                              | 450          | 95.46%    | 98.22%  |
| 8     | Speed limit (120km/h)                              | 450          | 97.70%    | 94.22%  |
| 9     | No passing                                         | 480          | 99.38%    | 100.00% |
| 10    | No passing for vehicles over 3.5 metric tons       | 660          | 100.00%   | 97.58%  |
| 11    | Right-of-way at the next intersection              | 420          | 98.55%    | 97.14%  |
| 12    | Priority road                                      | 690          | 99.70%    | 96.96%  |
| 13    | Yield                                              | 720          | 98.76%    | 99.17%  |
| 14    | Stop                                               | 270          | 99.26%    | 100.00% |
| 15    | No vehicles                                        | 210          | 88.09%    | 98.57%  |
| 16    | Vehicles over 3.5 metric tons prohibited           | 150          | 100.00%   | 100.00% |
| 17    | No entry                                           | 360          | 100.00%   | 99.72%  |
| 18    | General caution                                    | 390          | 96.86%    | 86.92%  |
| 19    | Dangerous curve to the left                        | 60           | 73.17%    | 100.00% |
| 20    | Dangerous curve to the right                       | 90           | 91.67%    | 97.78%  |
| 21    | Double curve                                       | 90           | 80.25%    | 72.22%  |
| 22    | Bumpy road                                         | 120          | 91.67%    | 91.67%  |
| 23    | Slippery road                                      | 150          | 90.07%    | 90.67%  |
| 24    | Road narrows on the right                          | 90           | 96.39%    | 88.89%  |
| 25    | Road work                                          | 480          | 96.32%    | 98.12%  |
| 26    | Traffic signals                                    | 180          | 86.67%    | 86.67%  |
| 27    | Pedestrians                                        | 60           | 76.39%    | 91.67%  |
| 28    | Children crossing                                  | 150          | 94.23%    | 98.00%  |
| 29    | Bicycles crossing                                  | 90           | 93.75%    | 100.00% |
| 30    | Beware of ice/snow                                 | 150          | 89.13%    | 82.00%  |
| 31    | Wild animals crossing                              | 270          | 97.41%    | 97.41%  |
| 32    | End of all speed and passing limits                | 60           | 100.00%   | 98.33%  |
| 33    | Turn right ahead                                   | 210          | 98.58%    | 99.52%  |
| 34    | Turn left ahead                                    | 120          | 97.54%    | 99.17%  |
| 35    | Ahead only                                         | 390          | 98.97%    | 98.97%  |
| 36    | Go straight or right                               | 120          | 96.77%    | 100.00% |
| 37    | Go straight or left                                | 60           | 92.19%    | 98.33%  |
| 38    | Keep right                                         | 690          | 99.71%    | 99.13%  |
| 39    | Keep left                                          | 90           | 94.68%    | 98.89%  |
| 40    | Roundabout mandatory                               | 90           | 95.60%    | 96.67%  |
| 41    | End of no passing                                  | 60           | 100.00%   | 100.00% |
| 42    | End of no passing by vehicles over 3.5 metric tons | 90           | 98.90%    | 100.00% |

It is worth mentioning, almost confessing, at this point that this model is not extensively tweaked. There is likely a better solution for this existing architecture if further iterations of the aforementioned hyper-parameters are conducted. However, the iterative process was abandoned at this stage since the proposed model significantly exceeded the requirements of the project and due to the limitation in available computational resources. Regardless, it was interesting to note that a simple 5 layer architecture was able to provide reasonable performance on this traffic sign dataset with minimal effort.

Additional avenues of investigation for this architecture would include improving the generalization performance by iterating over different values of beta, experimenting with additional augmentation techniques to improve the precision and recall for poorly performing labels and training over a larger # of epochs with more severe regularization.

**4. Test the model on new images**

*4.1 Acquiring new images*

The following 13 images were obtained from a combination of the p-traffic-signs slack channel as well as a google search. The images are plotted below after cropping:

<img src="./test_images/1-13.jpg" width="100"> <img src="./test_images/2-12.jpg" width="100"> <img src="./test_images/3-28.jpg" width="100"> 
<img src="./test_images/4-36.jpg" width="100"> <img src="./test_images/5-17.jpg" width="100"> <img src="./test_images/6-25.jpg" width="100"> 
<img src="./test_images/7-33.jpg" width="100"> <img src="./test_images/8-15.jpg" width="100"> <img src="./test_images/9-11.jpg" width="100"> 
<img src="./test_images/10-18.jpg" width="100"> <img src="./test_images/11-32.jpg" width="100"> <img src="./test_images/12-3.jpg" width="100"> 
<img src="./test_images/13-9.jpg" width="100"> 

The most concerning images for classification are the ones with finer details i.e. children crossing and roadwork. This assumption is based on the knowledge that the images downloaded needed to be cropped and resized prior to being processed. The resizing was done via OpenCV's resize() function using an inter-area interpolation to minimize the amount of pixelation as a result of downsampling. However, there was concern that the features of these particular signs would be compromised during the resizing which may lead to classification errors.

*4.2 Performance on new images*

Here are the results of the prediction:

| Image                                 | Prediction                            |
|---------------------------------------|---------------------------------------|
| Yield                                 | Yield                                 |
| General Caution                       | General Caution                       |
| End of all speed and passing limits   | End of all speed and passing limits   |
| Speed limit (60 km/hr)                | Speed limit (60 km/hr)                |
| No passing                            | No passing                            |
| Priority Road                         | Priority Road                         |
| Children Crossing                     | Children Crossing                     |
| Go straight or right                  | Go straight or right                  |
| No Entry                              | No Entry                              |
| Road work                             | Road work                             |
| Turn right ahead                      | Turn right ahead                      |
| No vehicles                           | No vehicles                           |
| Right-of-way at the next intersection | Right-of-way at the next intersection |

The model was able to correctly guess all 13 of the traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.4%. The images chosen in this test were all well-illuminated and unblurred which may have contributed to the favorable test results.

In addition, with the exception of two labels, No Vehicles & General Caution, with low precision (~88%) and recall (~87%) respectively, the remaining labels have relatively high precision and recall percentages on the test dataset which is consistent with the actual performance on these test images.

*4.3 Model Certainty/Softmax Probabilities*

The images below show the the top 5 probabilities for each of the 13 test signs visualized. The bar chart on the left shows the probability of each of the top 5 predictions with the title of the bar chart as the predicted sign name. The image on the right shows the actual traffic sign and label. The code for making predictions on my final model and visualizing the probabilities are located in the 21st and 23rd cells of the Ipython notebook respectively.

![alt text][image9]

As seen from the images above, the model is very certain of all test images with the exception of the children crossing sign. In this particular case, the model suspects there is just under a 60% probability of this sign being a children crossing sign and just over a 40% possibility of this sign being a bicycles crossing sign. This uncertainty is attributed to the pixelized nature of the downsized image which has reduced the granularity of the details for this sign.

**5. Visualizing the Neural Network**

The two images below show visualizations of the feature maps from the first and second convolution layers in the model. The input to the model is the processed Yield sign image that was the first test image from the set downloaded from the internet.

![alt text][image10]

![alt text][image11]

The feature maps of the first convolution layer seem to imply that the model is identifying and learning the various edges of the Yield sign image. However, since this visualization is grayscale it is difficult to interpret any learning of the color information that is passed to this layer since the training image input has dimensions (32,32,4). The output of the second convolution layer seems to be much more abstract and it is difficult to draw any meaningful conclusions from the visualized images.



