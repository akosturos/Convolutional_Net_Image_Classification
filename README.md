# Image Classification
In this project, I'll classify images from the CIFAR-10 dataset. The dataset consists of airplanes, dogs, cats, and other objects. I'll preprocess the images, then train a convolutional neural network on all the samples. The images will be normalized and the labels one-hot encoded. I'll apply a convolutional net, max pooling, dropout, and fully connected layers. At the end, I'll test the neural network's predictions on the sample images.

##  The Data
The dataset is broken into batches to prevent the machine from running out of memory.  The CIFAR-10 dataset consists of 5 batches, named `data_batch_1`, `data_batch_2`, etc.. Each batch contains the labels and images that are one of the following:
* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

## The Network Components

### Convolution and Max Pooling Layer
Convolution layers have a lot of success with images. For this code cell, I implement the function `conv2d_maxpool` to apply convolution then max pooling:
* Create the weight and bias using `conv_ksize`, `conv_num_outputs` and the shape of `x_tensor`.
* Apply a convolution to `x_tensor` using weight and `conv_strides`.
 * Use 'SAME' padding
* Add bias
* Add a nonlinear activation to the convolution.
* Apply Max Pooling using `pool_ksize` and `pool_strides`.
 * Use 'SAME' padding

### Flatten Layer
The `flatten` function changes the dimension of `x_tensor` from a 4-D tensor to a 2-D tensor.  The output should be the shape (*Batch Size*, *Flattened Image Size*).

### Fully-Connected Layer
Implement a fully_conn function to apply a fully connected layer to x_tensor with the shape (Batch Size, num_outputs).

### Output Layer
Implement the output function to apply a fully connected layer to x_tensor with the shape (Batch Size, num_outputs).

## The Convolutional Model
Implement function `conv_net` to create a convolutional neural network model. The function takes in a batch of images, `x`, and outputs logits.  Use the layers you created above to create this model:

* Apply 1, 2, or 3 Convolution and Max Pool layers
* Apply a Flatten Layer
* Apply 1, 2, or 3 Fully Connected Layers
* Apply an Output Layer
* Return the output
* Apply [TensorFlow's Dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout) to one or more layers in the model using `keep_prob`.

## RESULTS
### Why 50-80% Accuracy?
50% isn't bad for a simple CNN.  Pure guessing would get 10% accuracy. However, you might notice people are getting scores [well above 80%](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130).

##### I will keep training the network to get closer to 90%!!

