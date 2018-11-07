# Fully Convolutional Neural Networks
Contains code and an explanation of the theories and applications of Fully Convolutional Neural Networks

Task is to identify where the hot dog 

fully connected layers do not perserve spatial informnation

integrate convolutions in the layer

where is the object, the perserve the spatial information

will work on images of any siz


Object detection and segmantic segmentation 

Run an inference on an image

real time classification of objects 

FCNs have 3 techniques 

0. replace fully connected layers with 1x1 convolution layers 
1. Up-sampling through the use of transposed convolutional layer
2. Skip connections 

allow network to use information from multple resolution 

structure 

encoder is a series of convolution layers like VGG and ResNet
it works by extracts features from the image 

Decoder upscales the output of the encoder so that it is the same size as the original thus

Segmentation in each individual pixel in the original image. 







Segmentatic segmentaion 
--


# Fully Connected to 1x1 Convolutions

Reduce the dimensionality of the layer. 
The output shape of a convolutional layer is a 4D tensor. 

When we feed the ouput of a convolutional layer into a fully connected layer, it is flattened to a 2D tensor. Now the spatial information is lost. i.e the computer can only tell what the object is, not where it is.

Using 1 x 1 convolutions avoids this by perserving the 4D tensor. The output will contain spatial information allowing the computer to locate the object as well. 

A 1x1 convolution is convolving with a set of filters od dimensions

0. 1x1xfilter_size (HxWxD)
1. Stride = 1
2. zero(same) padding 

Recall: The output of the convolution operation is the result of sweeping the kernal over the input with a sliding window

element wise multiplication and summation 

The number of kernels is equivalent to the number of output in the FCN

The number of weights in each kernel is equavlent to the numeber of inputs in the FCN

Turns convolutions into matrix multiplication with spatial information. 


Changing from fully-connected layers to convolutional layers allows us to feed images (during testing) of any size into the trained network


# Transposed Convolutions

Help in upsampling the previous layer to a desired resolution or dimension. 
example 
a 3x3 input needs to be upsampled to 6x6.

The process
---
0. multiply each pixel of your input with a kernal or filter. 
1. if filter is nxn, the output of this operation will be a weighted kernal of size nxn
NOTE: The weighted kernal defines the output layer

Upsampling is defined by the strides and padding 

for this example using 

tf.layer.conv2d_transpose
(inputs,
    num_outputs,
    kernel_size,
    stride=1,
    padding='SAME',
    data_format=DATA_FORMAT_NHWC,
    activation_fn=tf.nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None
    )

stride = 2
padding = same

results in an output that is 6x6

If we have a 2x2 input and a 3x3 kernel; with "SAME" padding, and a stride of 2 we can expect an output of dimension 4x4.



Create a decoder of FCN 

a reverse convolution where the forward and backward passes are swapped.
NOTE math is the same
NOTE Training is performed exactly the same as other neural nets.

-undoes previous convolution 
math i

differentiabilty



# Skip Connections








# FCNs in the Wild
