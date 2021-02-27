# SureStartAI-2021
This is a repository for the Spring 2021 AI & Machine Learning virtual trainee program at SureStart, an organization connecting emerging AI talent to comprehensive technical curriculum and industry mentorship.

**Day #18 - February 26th**
Today I understood what autoencoding and upsampling means and what its significance is to machine learning. Upsampling and downsampling are two phenomena that help us manipulate our input data either for the model to use as training(downsampling) or for the sake of forming the inputs based on the sets of features (upsampling).

**Day #17 - February 25th**

- Today I had a hands on experience on how overfitting can be handled with in different ways. Overfitting seems to be a problem whereby the model knows too much hence having very low bias but high variance. We can handle it in different ways: 
- Reduce overfitting by training the network on more examples.
- or, Reduce overfitting by changing the complexity of the network

We can change the complexity of a network by changing its network structure and/or the parameters.
Furthermore, can:
- reduce the network’s capacity by removing layers or reducing the number of elements in the hidden layers, 
- or we can apply regularization, which comes down to adding a cost to the loss function for large weights,
- we can use dropout layers, which will randomly remove certain features by setting them to zero

Reference: https://machinelearningmastery.com/introduction-to-regularization-to-reduce-overfitting-and-improve-generalization-error/

I played with the housing notebook as well to see how it works in different settings. I will play with it again soon.


**Day #16 - February 23rd**

- Activation functions are functions that take the weighted sum input from nodes or data then convert them to some sort of output for another layer or for the output.
- They mostly work to ensure that data is passed on between different nodes across layers. In most situations, there will be lots of neurons in the hidden layers. Activation functions work to take whatever is received from the nodes to make a tangible/usable output out of it.
- A good activation function favors good training of the AI model in general.
- Most common forms of Activation functions are: Rectified Linear Activation (ReLU), Logistic (Sigmoid), and Hyperbolic Tangent (tanh)
- One advantage of the ReLU function is the it is less susceptible to the vanishing gradient problem. It also is less computationally expensive than sigmoid and tan h because of it's simplicty, as a function that outputs the input directly if it is positive and outputs 0 otherwise. This is also important because it means it can output true zero values. ReLU is commonly used in CNN's, so for example it could potentially be used for classification of dog species.
- The Sigmoid activation function takes any form of real value input and then outputs a numnber in the range between 0 and 1. It is same function used in logistic function regression algorithm. It is most likely used in output of the neural network model.
- The function takes any real value as input and outputs values in the range -1 to 1. The larger the input (more positive), the closer the output value will be to 1.0, whereas the smaller the input (more negative), the closer the output will be to -1.0.
- In most cases, this is the rule of thumb when it comes to the most optimal activation functions to use for hidden layers:
  - Multilayer Perceptron (MLP): ReLU activation function.
  - Convolutional Neural Network (CNN): ReLU activation function.
  - Recurrent Neural Network: Tanh and/or Sigmoid activation function.
- for the **output layers**, we have: Linear, Logistic, and Softmax
  - Softmax is mostly used in case we want the model to make a probability vector output.
  - Linear function is considered to be "no activation". This is because the linear activation function does not change the weighted sum of the input in any way and instead returns the value directly.

- You must choose the activation function for your output layer based on the type of prediction problem that you are solving.
  - Binary Classification: One node, sigmoid activation.
  - Multiclass Classification: One node per class, softmax activation.
  - Multilabel Classification: One node per class, sigmoid activation.

- At the end of the day and the type of output we need, the type of network we are using will greatly help us to determine what to use for the Activation function.


**Day #10 - February 19th**

- Today, I worked on building an image classifier with CNNs. I used a dataset that was offered online and then divided the images to classes; Class 0 is dog and class 1 is cat. I used 50 epochs and a batch size of 64 and my model trained as required. Today's work was built on what I did in the previous days with CNNs.

**Day #9 - February 18th**

- Today, I explord the building of a CNN from scratch. I implemented a baseline model for the MNIST database. On top of that, I improved the model using batch normalization as well as depth layer increase. 

- One thing that fascinated me today is that adding more layers does not necessarily make the model more accurate, whereas increasing the number of nodes per layer does. I think it's because in neural networks, one layer's output is the next layer's input. Therefore, adding more layers might just over-flatten the features to the extent that some important features are lost. On the other hand, increasing the number of nodes will increase the ability of feature information being retained, thus, leading to a more efficient and accurate model.


**Day #8 - February 17th**

- Today I learnt about the layers present in Convolutional Neural Networks (CNN). I also explored the difference between fully connected neural networks and CNNs.

**Difference between CNN and fully connected neural networks**

- In practice, it seems that fully connected neural networks have an architecture whereby all neurons connect to all neurons in the next layer. 
That being the case, it seems that these neural networks are generally tedious and hard to make and manage. This is because the architecture comprises of millions of features making it difficult to classify and analyse features to help in computer vision.

**Convolutional Neural Networks**
Again, CNNs are the backbone of machine vision and image recognition. This has been the case due to the fact that they have a very intuitive architecture. This type of neural networks uses convolution and pooling to reduce an image to its essential features and then use them to understand and classify the image accordingly. Of course, the hidden layers that make up CNNs are a "blackbox" since there is so much going on. However, we have three main layers and this is how they are and how they work:

- **Convolution layer**: This layer has filters that scan few pixels at a time and then create a feature map that predicts the class to which every feature belongs to. The result from this layer is a vector with classes of features. Usually, we might use more than one convolution layer when it comes to large datasets.
- **Pooling layer**: The input to this layer is a vector of classes of features. What this layer does is to downsample/flatten the features (break them down by reducing the amount of information in each feature) while maintaining the important information of all the features. Without this feature, we would have slower training since there would be too much information in the feeatures for the model to process. Furthermore, this helps the model to focus on the important information in each feature of the image.
- In a nutshell, the convolution and pooling layers together break the image into features and analyzes these features to ensure efficiency in training the model.

- **Fully connected layer**: This layer gathers all the input from the convolution and the pooling layers then assigns weights to the features (in form of neurons) so that to help predict the correct labels for each feature. The output is a list of probabilities for different possible labels attached to the image. 
- The first fully connected will work as aforementioned. However, the **fully connected output layer** gives the final probabilities for each label.
- The fully connected layer goes through backpropagation process to determine the most accurate weights. Each neuron reecives weights that prioritize the most appropriate label. Finally, the neurons "vote" on each of the labels, and the winner of that vote is the classification decision.

**Day #7 - February 16th**

- Today I learnt about bias in ML and how it can affect the output from ML models. Despite the fact that AI is really helpful, it is not completely fair. Fairness is subjective, however, precision is not. AI models rely heavily on human data input. If the data is skewed or biased, the model will be biased as well. Therefore, the goal of modern day software engineers should be to make models that reflect reality and that are not biased. I believe this could be achieved by using large diverse data sets and focusing on the false negatives in data.

-**How do you think Machine Learning or AI concepts were utilized in the design of this game?**

I think the game used a good amount of previous datasets that were associated with hiring in major firms in America. These datasets showed the long standing systemic racism as well as inequalities that exist in this country. The orange applicants went to better schools (with more reputation) and had higher skill whereas the blue applicants went to schools with lower reputation and had lower skill. I really enjoyed this game since it reflected alot on how hiring is in the modern world.

-**Can you give a real-world example of a biased machine learning model, and share your ideas on how you make this model more fair, inclusive, and
equitable? Please reflect on why you selected this specific biased model.**

- I think a good example is Amazon's Resume Screening Tool that was built in 2014 to automate the hiring process. The model was known to be segregative against women. This is bacause Amazon's computer models were trained to vet applicants by observing patterns in resumes submitted to the company over a 10-year period. Most came from men, a reflection of male dominance across the tech industry. 

**Why I chose this model?**

- I chose this because Amazon is one of the best software companies in the world and a lot of brilliant minds work there. However, it still had world-class software engineers who made a model that was biased. Therefore, I think scientists today should understand that the future of software and AI in general rest in our hands as engineers. Our competence, skill level, and work experience will not determine if our model is biased or smart enough, or not. We need to actively seek to eliminate bias in decision making so that the oour software can make better decisions that make people's lives better, and not the opposite.

Based on this article:https://searchenterpriseai.techtarget.com/feature/6-ways-to-reduce-different-types-of-bias-in-machine-learning ;

These are the ways to reduce bias in ML:

- Identify potential sources if bias. These might be based on historical data, correlation fallacies, etc. 

- Prior to collecting and aggregating data for ML model training, organizations should first try to understand what a representative data set should look like. 

- Evaluate ML models after being placed in operation and check for bias when testing as well. 

- It is also important to allow others to examine when and if the models exhibit any form of bias. This is because transparency allows for root-cause analysis of sources of bias to be eliminated in future model iterations. 


**Day #6 - February 15th**
Convolutional Neural Networks
- Convolution

Convolution preserves spatial data between pixels and learns image features by "convolving" filters across an image. We slide the filter over the original image by our stride and compute an element wise multiplication. We add the output to the corresponding index of the feature map. In this sense, filters act as "feature detectors" in images. It is a way to turn the image to a pixelated image to be processed.

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
