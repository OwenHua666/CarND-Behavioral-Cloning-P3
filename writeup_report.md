# Behavioral Cloning

## Writeup

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Model_Architecture.png "Model Visualization"
[image2]: ./examples/center_driving.PNG "Grayscaling"
[image3]: ./examples/flip.PNG "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5X5 and 3x3 filter sizes and depths between 36 and 64.

The model includes RELU and ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 18). 

Three fully connected layers are added at the end to do the prediction for the steering angle.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers and max polling layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to iteratively test different structures and parameters to achieve a lower validation loss.

My first step was to start with a simple model architecture which only contains one 36-3x3 convolutional layers and three fully connected layers. I also looked at the neural network presented by Nvidia Self-Driving Car group, but I think it is too complicated for this project. I also don't have enough images to train a network with so many parameters. My final version of the model architecture does contain one more convolutional layer and a few more fully connected layers The size and add layers are determined through test.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. It is weird that my first validation loss is small and my first training loss is big. As the training epoch increases, the two errors decrease and converge. In the last few epochs, the validation loss is greater than the training loss. This indicates the trained neural network overfits the training dataset. 

To fight against overfitting, I add Max Pooling layer after first convolutional layer and apply dropout to the first fully connected layer.

The final step was to run the simulator to see how well the car was driving around track one. My first few models drive the vehicle off the trace because I normalized the image twice in the prediction pipeline. After debugging, even the simple model can drive the vehicle within the track at a low speed(9 mi/hr). When I increase the set_speed in drive.py, the vehicle starts to have a zigzagy trajectory around the center lane of the track. When the speed is 15 mi/hr, the vehicle loses its capability to recover to the centerline and falls off the track. To improve the driving behavior in these cases, I use a more complicated neural network (one more convolutional layer and one more fully connected layer). More importantly, I obtain more training data when I drive the vehicle smoothly on the course. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road at a highest speed of 21 mi/hr. 

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.
* ConvNet 36-5x5
* MaxPooling 2x2
* ConvNet 48-3x3
* Flatten
* FC 100
* FC 50
* FC 10

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it goes off the center line of the lane. I also drove the vehicle as smooth as possible. These compose the dataset I obtained from the fouth lap and fifth lap.

To augment the data sat, I also flipped images and angles thinking that this would balance dataset which has more "left turns". For example, here is an image that has then been flipped:

![alt text][image3]

I noticed there are a lot of small turning angles in the dataset because the vehicle is driving straight most of the time. So, I drop 75% image data whose turning angle is smaller than 0.15.

I tried to shift the image left and right and adjusted its turning angle based on the shifting magnitude. A random brighness is also added to the image. However, these two augmentation methods does not improve the driving performance and increases both training and validation errors. Therefore, the code of these two methods are shown in model.py, but they are not called in the training pipeline. 

After the collection process, I had 15368 number of data points. I then preprocessed this data by cropping the irrelevant area like the sky and hood out and resize each image to 64x64x3.  


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8 as evidenced by the training loss and validation loss are not decreasing after 8 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
