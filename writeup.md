#**Traffic Sign Recognition** 
---
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./Write-up%20Images/DataHistogram.png "Visualization"
[image2]: ./Write-up%20Images/10-image-set.png "31 - Wild Animals Crossing"
[image3]: ./Write-up%20Images/y-flipped-image.png "Image Mirrored along Y-axis"
[image4]: ./Write-up%20Images/AugDataHistogram.png "Data Distribution after Basic Augmentation"
[image5]: ./Write-up%20Images/augmentation.png "Augmentation via Translation/Rotation"
[image6]: ./Write-up%20Images/FinalDataHistogram.png "Data Distribution after Final Augmentation"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
The following sections address the individual [Rubric points](https://review.udacity.com/#!/rubrics/481/view) for this submission. 

---
**1. Submission Files**

This submission consists of:

1. The IPython notebook containing the project code: [Traffic_Sign_Classifier.ipynb](https://github.com/shazraz/P2-TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)
1. An HTML version of the IPython notebook: [Traffic_Sign_Classifier.html](https://github.com/shazraz/P2-TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.html)
1. The trained and saved model: LeNet5_Run3
1. A set of images downloaded from the internet used to test the model: [Images](https://github.com/shazraz/P2-TrafficSignClassifier/tree/master/Images)
 {source: Sonja Krause-Harder for some of these images on the p-traffic-signs Slack channel}

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

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

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
| 32      | End of all speed and passing limits                | Y                  |        |        | X       |         |           |
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

The following histogram shows a result of basic augmentation with a number of under-represented labels being passed in for augmentation. It is seen that while some labels (i.e. 17, 26, 33, 34, 39) are somewhat more represented, additional augmentation is required. 

![alt text][image4]

Consequently, minor random pertubations were applied to the training set images to further balance the dataset. This consisted of translating the images along the both the x and y axes by a random amount within a fixed range or rotating the images about their center by a random amount within a fixed range. This was accomplished by the use of the augment_set() function which was passed a number of parameters including the dataset of images and labels, the labels within the dataset to augment, the minimum threshold quantity of each label and the ranges for the translation and rotation when augmenting the dataset. This led to the introduction of the following hyper-parameters within the model and their corresponding values used for augmentation:

* n_threshold: 1500,  threshold quantity of images within each label
* rt_range: 15, range of rotation for each image
* xlate_range: 5, range of (x,y) translation for each image

The figure below shows a series of augmented images created from a single original image

![alt text][image5]

Once the augmentation was complete, the dataset was much more balanced and ready for further processing prior to use in training. This is shown in the histogram below:

![alt text][image6]

3.1.2 Image Processing

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


