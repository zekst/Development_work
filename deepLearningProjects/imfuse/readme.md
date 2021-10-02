




**ABSTRACT** 

This project explores methods for artistic style transfer based on convolutional neural networks. As a technique that combines both artistic aspects and recognition (content) aspects of images, style transfer has always been an interesting topic for researchers in the field of computer vision. With the rapid growth of Deep Convolutional Neural Networks, style transfer can now be accomplished in a speedy manner. Based on papers that relate to fast-style style transfer as well as mixed-style transfer, we have developed our own implementation of style transfer that manages to transfer images with mixed styles or even unseen styles in real-time. We experimented with arbitrary style transfer techniques and studied it’s performance. Moreover, because not everybody can readily access massive compute clusters, we also explored ways to rescale these techniques and apply them to smaller-scale compute infrastructure.
` `PAGE   \\* MERGEFORMAT 1



**CONTENTS**



`                     `Page No.

*Candidate's Declaration	1*

*Abstract	2*

[*Acknowledgement                 i](#_30j0zll)                                                                                       *3*


|**1.**|**INTRODUCTION**||
| - | - | -: |
||<p>1.1 Problem Definition</p><p>&emsp;1.2 Project Overview</p><p>&emsp;1.3 Hardware Specification</p><p>&emsp;1.4 Software Specification</p><p></p>|<p>5</p><p>5</p><p>6</p><p>7</p><p></p>|
||||
|**2.**|**LITERATURE SURVEY**||
||<p>2.1 Existing System</p><p>2.2 Proposed System</p><p>2.3 Feasibility Study</p>|<p>8</p><p>8</p><p>9</p>|
||||
|**3.**|**SYSTEM ANALYSIS & DESIGN**|**  |
||<p>3.1 Flowcharts </p><p>3.2 Design and Test Steps </p><p>3.3 Algorithms and Pseudo Code</p><p>3.4 Testing Process</p>|<p>` `10     </p><p>12</p><p>13</p><p>14</p><p></p><p></p>|
||||
|**4.**|**RESULTS / OUTPUTS**|<p>**15**</p><p></p>|
|**5.**|**CONCLUSIONS** |<p>**17**</p><p></p>|
||**REFERENCES**|**18**|





**INTRODUCTION**

**1.1 Problem Definition**

Neural Style Transfer (NST) is one of the most fun techniques in deep learning. As seen below, it merges two images, namely, a "content" image (C) and a "style" image (S), to create a "generated" image (G). The generated image G combines the "content" of the image C with the "style" of image S.

In this example, you are going to generate an image of the Louvre museum in Paris (content image C), mixed with a painting by Claude Monet, a leader of the impressionist movement (style image S).

As this is a Neural Style Transfer project, one of the main problems encountered here are the models not being very efficient which in turn affects the accuracy of the style transfer from the “style” image to the “content” image. Hence, the output image does not turn out as it should. 

This problem mainly occurs because of the data used to train the model not being adequate. The more data is feeded to it, the more accurately the model gets trained and this can be further determined by plotting graphs to check the efficiency of the trained model.

The other major problem which is encountered is that the interface is very complex for the users so, there is a need to simplify the interface so every user can use the application to carry out neural style transfer with ease.

**1.2 Project Overview**

“imfuse” as a computer vision model, generates the common features in two given pictures from the user and ‘merges’ into a common image.

Our model ‘imfuse’ uses convolutional neural network concept to draw the similarity between images and apply it afterwards.

This is an art-generation creative idea of merging two artifacts into single.

It is an artistic approach to computer vision that shows 

The project can be used to simply apply any design or pattern over any object present in an image which can be very useful in fine arts, fashion and automotive industriues. 









**1.3 Hardware Specification**

The hardware minimum and maximum recommended requirements are listed below:

|**Hardware**|<p>**Minimum Recommended**</p><p>**Requirements**</p>|<p>**Maximum Recommended**</p><p>**Requirements**</p>|
| - | - | - |
|Internal	Memory (RAM)|8.00 GB|16.00 GB or Higher|
|<p>Hard	Disk	Capacity</p><p>(CPU)</p>|60.00GB|80.00GB or Higher|
|Processor|Intel i3 8th gen|<p>Intel(R) Core i5 8th gen</p><p>or Higher</p>|
|Monitor|17” Colored 32bit|18 ” Colored or Higher 64bit|
|VRAM|`       `2GB|4GB|



**1.4 Software Specification**

The software minimum recommended requirements and maximum recommended requirements are listed below:


|**Software**|**Minimum	Recommended Requirements**|**Maximum	Recommended Requirements**|
| - | - | - |
|System type|Microsoft	Win7	or	XP 32bit Operating System|Microsoft	Win10	64bit Operating System|
|Programming Language Compiler|`          `Jupyter Notebook|`           `Jupyter Notebook|















**Literature Survey**

1. **Existing System**

Gatys [1] first introduced in 2015 a deep neural network approach that extracts neural representations to separate and recombine the content and style of arbitrary images. CNNs are some of the most powerful procedures for image processing tasks, and have recently reached human level performance in classification tasks. The neural layers of a CNN can be understood as a set of image filters that extracts higher level features from the pixels of an input image. CNNs develop a representation of the image that makes content information increasingly explicit along the processing hierarchy as they train, such that the input image is transformed into representations that increasingly care about the actual content of the image compared to the individual values of its pixels. Although [1] showed that the style and content of an image can be disentangled and applied independently, the method is computationally expensive. Johnson’s work [2] was able to speed up style transfer by training a feed forward network to replace the optimization-based method of Gatys, and ended up being many times faster, allowing the transformation of video input in real-time. Furthermore, Keiji [5], Huang et al [6] and Ghiasi et al [7] proposed methods that augment Johnson0 s work to take a style choice or style image as inputs to the feed-forward network. Keiji [5] proposed that the network can take an additional conditional vector input indicating the style and the styles can be mixed at test time. Huang et al [6] proposed the network can learn a set of adaptive instance normalization parameters representing a style.

**2.2  Proposed System**

To generalize the style transfer problem, the feedforward network can take the style image as input as well, as illustrated in Figure 3. The style prediction network uses 4 convolutional layers, 3 inception modules and a convolutional layer followed by a global average pooling layer to extract the style as two 256-dimensional vectors feeding into the image transformation network as the scale and off-set to be applied to the response after an instance normalization layer.

For our project, we attempted to replace the pretrained style extraction network with a smaller and trainable network described above because we have much smaller compute infrastructure available. We were able to train the network on 5 styles only. Our implementation was limited in terms of GPU memorysince Tensorflow only supports static computational graphs.![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.001.png)



**2.3 Feasibility Study**

- Technical Feasibility:

This project is made on Jupyter Notebook which is an open-source web application that basically works a complete workstation for developing and running python projects and combines code, rich text, images, videos, animations, mathematical equations, plots, maps, interactive figures and widgets, and graphical user interfaces, into a single document. It’s interface can be used to create an entirely customized experience in the Jupyter Notebook (or another client application such as the console).

` `Jupyter notebooks can help you conduct efficient and reproducible interactive computing experiments with ease. It lets you keep a detailed record of your work. Also, the ease of use of the Jupyter Notebook means that you don’t have to worry about reproducibility; just do all of your interactive work in notebooks, put them under version control, and commit regularly. Don’t forget to refactor your code into independent reusable components. 

- Operational Feasibility

The project works really well in transferring the style from the “style” image to the” content” image to produce the “generated” image if we consider the data set used to train the model which was not very diverse due to hardware restrictions. It works well in detecting the edges of the object in the “content” image and transfers the style very well which can be very useful in the fine arts, fashion as well as automotive industry because it can be used to determine how a particular pattern or design will look on a car or a dress which will in turn save a lot of time which would probably get wasted in making an actual sample of the product with the pattern or design. Hence, it will save a lot of time as well as money.

- Economic Feasibility 

It is extremely feasible economically as you only need a desktop with not very high-end hardware and it is made on a free open-source web application.













**SYSTEM ANALYSIS & DESIGN**

**3.1 Flowcharts**

![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.002.png)

![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.003.png)

![0\_F4F\_8DzBkBFh3XWi.png](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.004.png)








**3.2 Design and Test Steps**

1. Input data

Input the dataset into TensorFlow and set basic parameters learning rate, epochs and batch size (see our guide on hyperparameters  to learn more about these):

1. Convolution layer

Create a function that defines a convolutional layer. In the function, we setup the input shape of the data, initialize weights and bias, create the convolutional layer using the tf.nn.conv2d function, and apply a Relu activation function.

1. Fully connected layers

Flatten the output of the convolutional layers

Setup weights and biases, and create two densely connected layers, with softmax activation, which is appropriate for an output layer that generates probabilities for predictive labels. We’ll use a cross-entropy loss function, built into TensorFlow.

**3.3 Algorithms and Pseudo Code**

In this model we used the “neural style transfer” algorithm. Neural Style Transfer (NST) is one of the most fun techniques in deep learning. As seen below, it merges two images, namely, a "content" image (C) and a "style" image (S), to create a "generated" image (G). The generated image G combines the "content" of the image C with the "style" of image S.

Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. The idea of using a network trained on a different task and applying it to a new task is called transfer learning.

Following the original NST paper (https://arxiv.org/abs/1508.06576), we will use the VGG network. Specifically, we'll use VGG-19, a 19-layer version of the VGG network. This model has already been trained on the very large ImageNet database, and thus has learned to recognize a variety of low-level features (at the earlier layers) and high-level features (at the deeper layers).

**We will build the NST algorithm in three steps:**

1. **Build the content cost function![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.005.png)** 
1. **Build the style cost function ![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.006.png)**
1. **Put it together to get![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.007.png)**
###
### *1 - Computing the content cost:*


`         `We will define as the content cost function as:

`              `![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.008.png)

`            `Here ![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.009.png)and  ![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.010.png) are the height, width and number of channels of the hidden layer you                    have chosen, and appear in a normalization term in the cost. For clarity, note that  ![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.011.png) and                      ![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.012.png) are the volumes corresponding to a hidden layer's activations.

*2- Computing the style cost:*
\*
`      `We will define the style cost as:

`      `![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.013.png)

`            `where ![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.014.png) and ![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.015.png) are respectively the Gram matrices of the "style" image and the                               "generated" image, computed using the hidden layer activations for a particular hidden layer in                   the network.

3- Defining the total cost:

`  	`Create a cost function that minimizes both the style and the content cost. The formula is:

![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.007.png)

- The total cost is a linear combination of the content cost and the style cost 
- alpha and beta are hyperparameters that control the relative weighting between content and style

**3.4 Testing Process**

Step 1:

In the first step, we will perform a GET request to retrieve the image data. To make a GET request, we will need to import request

Now, we set a variable URL and assign the link as a string.

Step 2:

In the next step, we set a variable response whose value will get from the get() method of request. The get() method will consist of two arguments, i.e., URL and stream and stream will be equals to true.

Step 3:

We will use the raw content of our response to obtain the image. For this, we first have to import Image from PIL (Python Image Library) as.

We use the open() method of image and pass the raw content of response as an argument. The value which will be returned from this method will assign to a variable named img

Step 4:

We need to ensure that the image corresponds to what the neural network is trained to learn. Our image is of 1000\*1000 pixels, so we need to make it into 28\*28 grayscale image like ones in the training data. In our trained image dataset, images have a black background and white foreground, and in the above image, there is a white background and black foreground. Now, our first task is to preprocessing this image.

We will use the invert () method of PIL.ImageOps and pass our image as an argument. This method will invert the color of our image.

Step 5:

Now, we will feed this image into our neural network to make predictions. We will add the image into the device and to ensure four-dimensional input for four-dimensional weight. We will un-squeeze the image and assign it to a new variable image

Step 6:

In the next step, we wrap our validation loader. It will create an object which allows us to go through the alterable validation loader one element at a time. We access it one element at a time by calling next on our dataiter. The next () function will grab the first batch of our validate data, and that validate data will be split into images and labels

Step 7:

Now, we will plot the images in the batch along with their corresponding labels. It will be done with the help of figure function of plt and set fig size is equal to the tuple of integers 25\*4, which will specify the width and height of the figure.














**RESULTS/OUTPUTS**

**OUTPUT 1:**

![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.016.png)

**OUTPUT 2:**

![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.017.png)



**OUTPUT 3:**

![](Aspose.Words.a12aea35-b07c-45a6-a1ae-88531ba1bdcd.018.jpeg)






















**CONCLUSIONS**

- Neural Style Transfer is an algorithm that given a content image C and a style image S can generate an artistic image

- It uses representations (hidden layer activations) based on a pretrained ConvNet.

- The content cost function is computed using one hidden layer's activations.

- The style cost function for one layer is computed using the Gram matrix of that layer's activations. The overall style cost function is obtained using several hidden layers.

- Optimizing the total cost function results in synthesizing new images.





























**REFERENCES**

[1] Image Style Transfer Using Convolutional Neural Networks,

Gatys et al, CVPR 2016

[2] Perceptual Losses for Real-Time Style Transfer and Super-

Resolution, Johnson et al, arXiv: 1603.08155

[3] Very Deep Convolutional Networks for Large-Scale Image

Recognition, Simonyan et al, arXiv: 1409.1556.3

[4] Fast Style Transfer, Logan Engstrom, https://github.

com/lengstrom/fast-style-transfer/, 2016

[5] Unseen style transfer based on a conditional fast style transfer

network, Yanai, Keiji, 2017

[6] Arbitrary Style Transfer in Real-time with Adaptive In-

stance Normalization, Xun Huang, Serge Belongie, arXiv:

1703.06868, 2017

[7] Exploring the structure of a real-time, arbitrary neural artistic

stylization network, Ghiasi et al, arXiv: 1705.06830, 2017
