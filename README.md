# Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning

DataSet:
 
ChestX-ray14 dataset is used which contains 112,120 frontal-view X-ray images of 30,805 unique patients.
Random splitting of dataset into training (28744 patients, 98637 images), validation (1672 patients, 6351 images), and test (389  patients,  420  images) is done.
There is no patient overlap between the sets.
Each image in the data set contains multiple text-mined labels identifying 14 different pathological conditions.
These in turn can be used by physicians to diagnose 8 different diseases.
We have used this data to develop a model that will provide binary classification predictions for each of the 14 labeled pathologies.
In other words it will predict 'positive' or 'negative' for each of the pathologies.

Code Implementation:

 Preventing Data Leakage:

  It is worth noting that our dataset contains multiple images for each patient. This could be the case, for example, when a patient has taken multiple X-ray images at different times during their hospital visits. In our data splitting, we have ensured that the split is done on the patient level so that there is no data "leakage" between the train, validation, and test datasets.
 
 Preparing Images:

   With our dataset splits ready, we can now proceed with setting up our model to consume them.

    For this we have used the off-the-shelf ImageDataGenerator class from the Keras framework, which allows us to build a "generator" for images specified in a dataframe.
    This class also provides support for basic data augmentation such as random horizontal flipping of images.
    We also use the generator to transform the values in each batch so that their mean is 0 and their standard deviation is 1.
    This will facilitate model training by standardizing the input distribution.
    The generator also converts our single channel X-ray images (gray-scale) to a three-channel format by repeating the values in the image across all channels.
    We wanted this because the pre-trained model that we'll use requires three-channel inputs.

   Since it is mainly a matter of reading and understanding Keras documentation, we have implemented the generator for you. There are a few things to note:

    We normalize the mean and standard deviation of the data
    We shuffle the input after each epoch.
    We set the image size to be 320px by 320px

 Addressing Class Imbalance:

   One of the challenges with working with medical diagnostic datasets is the large class imbalance present in such datasets.
   If we had used normal cross-entropy loss function with a highly unbalanced dataset, then the algorithm would be incentivized to prioritize the majority class (i.e negative in our case), since it contributes more to the loss. 
   Hence we have used the weighted loss technique to prevent class imbalance.

 DenseNet121:

   Next, we have used a pre-trained DenseNet121 model which we have loaded directly from Keras and then add two layers on top of it:
    A GlobalAveragePooling2D layer to get the average of the last convolution layers from DenseNet121.
    A Dense layer with sigmoid activation to get the prediction logits for each of our classes.

  Training:
   The input size for each image was set to 224, as each image had a dimension of 224x224 and the images were normalised based on the mean and standard deviation of images in the ImageNet training set. 
   The Batch Size for training is set to 16. We also augment the training data with random horizontal flipping along with scaling and centre cropping. 
   Instead of training the 121 layered DenseNet end-to-end using Adam with standard parameters (β1 = 0.9 andβ2 = 0.999), we have used a Stochastic Gradient Descent Optimiser with a Momentum of 0.9
   We have used an initial learning rate of 0.01, instead of 0.001 that is decayed by a factor of 10 each time the validation loss plateaus after an epoch, and pick the model with the lowest validation loss.
   After the final dense layer a GlobalAverage Pooling 2D layer is used to get the average of the last convolution layers from DenseNet121.Since we are doing amulti task classificationi.e. we are able to detectmore than 1 disease from the dataset, we use a Dense layer with sigmoid activation to get the prediction logits for each of our 14 classes. Class activation maps are useful for understanding where the model is looking when classifying an image.
   We have used Grad-Cam’s technique to produce a heatmap highlighting the important regions in the image for predicting the pathological condition. This is done by extracting the gradients of each predicted class, flowing into our model’s final convolutional layer.

  Testing and Evaluating:
   Now that we have already trained the model, we evaluate it using our test set. We have conveniently used the predict_generator function to generate the predictions for the images in our test set.
  
  Visualizing Learning with GradCAM:
   One of the most common approaches aimed at increasing the interpretability of models for computer vision tasks is to use Class Activation Maps (CAM).
   Class activation maps are useful for understanding where the model is looking when classifying an image.
   In this section we have used a GradCAM's technique to produce a heatmap highlighting the important regions in the image for predicting the pathological condition.
   This is done by extracting the gradients of each predicted class, flowing into our model's final convolutional layer.
   It is worth mentioning that GradCAM does not provide a full explanation of the reasoning for each classification probability.
   However, it is still a useful tool for debugging our model and augmenting our prediction so that an expert could validate that a prediction is indeed due to the model focusing on the right regions of the image.
