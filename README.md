# SureStartAI-2021
This is a repository for the Spring 2021 AI & Machine Learning virtual trainee program at SureStart, an organization connecting emerging AI talent to comprehensive technical curriculum and industry mentorship.


**Day #6 - February 15th**
Convolutional Neural Networks
- Convolution

Convolution preserves spatial data between pixels and learns image features by "convolving" filters across an image. We slide the filter over the original image by our stride and compute an element wise multiplication. We add the output to the corresponding index of the feature map. In this sense, filters act as "feature detectors" in images.
Different filters can produce different feature maps and help in operations such as edge detection, blur, and sharpening.
In practice, the CNN learns the values of filters during the training process. However, a human still much specify parameters such as the # of filters, filter size, etc. Generally, the more filters we have, the more image features we can extract and the better the system becomes when classifying unseen images.
The size of a feature map is controlled by depth (# of filters; this creates a "volume"), stride (# of pixels we slide the filter matrix over the input matrix), and zero-padding (padding the input matrix with zeroes on the border so we can apply the filter to bordering elements).
Non-linearity

After every convolution operation, a non-linear operation (such as ReLU) is applied. This introduces non-linearity and helps our model become more complex.
Pooling

-Pooling, or downsampling, reduces the dimensionality of the feature map but retains important information. Max pooling takes the largest element from the feature map within that window. After pooling, we are left with a reduced spatial size of the input representation.
Classification

-A fully-connected layer acts as a classifier. The output from the previous convolution and pooling layers represent high-level features of the input image. We use a fully-connecting layer (every neuron in the previous layer is connected to every neuron on the left layer) to use these features for classifying the input image into various classes based on the training dataset. Then backpropogation is used to minimize the output error.


**Day #5 - February 12th**

-Today, we explored different components of neural networks, including concepts such as bias, activation functions, weights, densely connected layers, and epochs/batch-size. Furthermore, we learnt how to use tokenization and padding, text pre-processing, as well as using flattening our model so that it can be well used in other layers of the neural network. We explored how to create a convolutional neural network (CNN) that would use NLP techniques to classify news headlines from a Kaggle dataset as sarcastic or not sarcastic.


**Day #4 - February 11th**

-Deep learning is a subset of machine learning that includes a set of algorithms that mimic the structure and function of the brain in working. It comes in forms such as convolutional neural networks, recurrent neural networks, generative adversarial networks, and recursive neural networks. Deep learning is usually unsupervised or semi-supervised type of ML.

-Deep learning has seen a major advancement over the years. Mostly it has been used in Natural Language Processing. As a matter of fact, a big part of NLP is sentmiment analysis and it is very much used in Netflix to classify movies based on user searches and genres.

One good one is: http://help.sentiment140.com/for-students/
-If I were to develop a model to do this, I would use recurrent neural networks (RNNs) as they are conventionally used in any sort of application regarding language. It is also useful to use bidirectional neural networks for this.


**Day #3 - February 10th**
- **What are "tensors" and how are they used in machine learning?**

- A tensor is a mathematical contruct that allows us to describe physical quantities. Tensors can have various ranks, eg: scalar is basically a tensor of rank 0(with magnitude and direction), vector is basically a tensor of rank 1 (with magnitude and direction), etc. Tensors are dynamically-sized multidimensional data arrays.
- Usually, tensor calculations will provide a meaningful output if they are prompted on iPython or any interactive shells.

- **What did you notice about the computations that you ran in the TensorFlow programs (i.e interactive models) in the tutorial?**

-In TensorFlow, developers are able to create graphs (series of processing nodes). Each node in teh graph represents a mathematical operation, and each edge is a multidimensional data array, or a tensor. TensorFlow supports dataflow programming; dataflow programming models a program as a directed graph of the data flowing between the operations.
- Furthermore, in TensorFlow, developers work to build a computational graph and tyhen execute the graph in order to run the operations.


**Day #2 - February 9th**
- **What is the difference between supervised and unsupervised learning?**

- Supervised learning usually involves feeding the computer a lot of data so that the computer can be able to make decisions. It takes two forms; classification and regression models. The latter includes numerical quantification to data under study whereas classification models solve categorical problems mostly.

- On the other hand, unsupervised learning is aimed at uncovering patterns and relationships that exist within data. In this form of ML, there is no training. However, a common way for this is clustering eg; K-means clustering. where data is grouped into different groups/clusters based on similarity and differences.

- **Does scikit-learn have the power to visualize data by itself?**

-No, scikit learn is mostly responsible for models and datasets. It is not able to do visualization without being complemented by other packages like graphviz, etc.


**Day #1 - February 8th**

Today was an introduction to the Spring 2021 SureStart program, an initiative connecting emerging AI talent to machine and deep learning curriculum. I am currently a sophomore at the University of Rochester majoring in Computer Science. I am interested in learning AI and Machine Learning toolkits and technologies in general so that I can develop apps that utilize AI as well as to create efficient systems that utilize the power of machine learning and AI.
I aim to understand:
- Natural Language Processing
- Deep Learning and Neural Networks
- Unsupervised and Supervised learning techniques

I am mostly interested in App development as well as Systems Programming. I hope to use the knowledge and hands-on experience from this program to make tools that will impact thousands of users using ethical AI development.
