# SureStartAI-2021
This is a repository for the Spring 2021 AI & Machine Learning virtual trainee program at SureStart, an organization connecting emerging AI talent to comprehensive technical curriculum and industry mentorship.

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
