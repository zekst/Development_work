

**ABSTRACT** 

Emojis are small emojis that are commonly included in social media text messages. The combination of visual and textual content in the same message builds up a modern way of communication. Despite being widely used in social media, emojis underlying semantics have received little attention from a Natural Language Processing standpoint. In this project, we investigate the relation between words and emojis, studying the novel task of predicting which emojis are evoked by text-based tweet messages. We experimented variant of word embedding techniques, and train several models based on Multinomial Naive Bayes and LSTMs in this task respectively. Our experimental results show that our model can predict reasonable emoji from tweets.































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

People use emojis every day in their conversation. Emojis have become a new language that can more effectively express an idea or emotion. This visual language is now a standard for online communication, available not only in Twitter, but also in various large online platform such as Facebook and Instagram. Right now, the keyboard on iOS can predict emojis but only base on certain keywords and tags that are associated with emojis. Figure 1: Examples of emoji prediction in iOS13 keyboard Emoji prediction is a fun variant of sentiment analysis. When texting your friends, emoji can make your text messages more expressive. It would be nice if the keyboard can predict emojis based on the emotion and meaning of the whole sentence you typed out. Moreover, despite its status as language form, emojis have been so far scarcely studied from a Natural Language Processing (NLP) standpoint. The interplay between text-based messages and emojis remains virtually unexplored. In this application project, we target to fill this gap by investigating the relation between words and emojis, studying the problem of predicting which emojis are evoked by text-based tweet message. We build classifiers that learns to associate emojis with sentences. The models we explore here is Multinomial Naive Bayes and Bidirectional LSTM. The Standard Bag of Word TF-IDF and pre-trained ‘GLoVe’ model are used as word embedding, respectively. We train a large dataset of sentences with emojis labels aggregated from Twitter messages. In the Inference stage, the trained classifier takes as input a sentence and finds the most appropriate emoji to be used with this sentence.










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

In the traditional architecture there existed only the server and the client. In most cases the server was only a data base server that can only offer data.

Furthermore, Keiji [5], Huang et al [6] and Ghiasi et al [7] proposed methods that augment Johnson0 s work to take a style choice or style emoji as inputs to the feed-forward network. Keiji [5] proposed that the network can take an additional conditional vector input indicating the style and the styles can be mixed at test time. Huang et al [6] proposed the network can learn a set of adaptive instance normalization parameters representing a style.

**2.2  Proposed System**

Have you ever wanted to make your text messages more expressive? predem awill help you do that. So rather than writing "Congratulations on the promotion! Lets get coffee and talk. Love you!" the predem can automatically turn this into "Congratulations on the promotion! 👍 Lets get coffee and talk. ☕️ Love you! ❤️"

We will implement a model which inputs a sentence (such as "Let's go see the baseball game tonight!") and finds the most appropriate emoji to be used with this sentence (⚾️). In many emoji interfaces, you need to remember that ❤️ is the "heart" symbol rather than the "love" symbol. But using word vectors, you'll see that even if training set explicitly relates only a few words to a particular emoji, your algorithm will be able to generalize and associate words in the test set to the same emoji even if those words don't even appear in the training set. This allows you to build an accurate classifier mapping from sentences to emojis, even using a small training set.



**2.3 Feasibility Study**

- Technical Feasibility:

This project is made on Jupyter Notebook which is an open-source web application that basically works a complete workstation for developing and running python projects and combines code, rich text, emojis, videos, animations, mathematical equations, plots, maps, interactive figures and widgets, and graphical user interfaces, into a single document. It’s interface can be used to create an entirely customized experience in the Jupyter Notebook (or another client application such as the console).

` `Jupyter notebooks can help you conduct efficient and reproducible interactive computing experiments with ease. It lets you keep a detailed record of your work. Also, the ease of use of the Jupyter Notebook means that you don’t have to worry about reproducibility; just do all of your interactive work in notebooks, put them under version control, and commit regularly. Don’t forget to refactor your code into independent reusable components. 

- Operational Feasibility

The project works really well in predicting the appropriate emoji for the sentences. The emoji to be displayed after each prediction. The generated” emoji if we consider the data set used to train the model which was not very diverse due to hardware restrictions. Everywhere on social media we can see the use of emoji very often. Emoji makes it easy to express the emotions and makes the conversation more articulative and interesting thus providing significant increment in human interaction.

- Economic Feasibility 

It is extremely feasible economically as you only need a desktop with not very high-end hardware and it is made on a free open-source web application.













**SYSTEM ANALYSIS & DESIGN**

**3.1 Flowcharts**

![Chart, waterfall chart

Description automatically generated](Aspose.Words.d8cb79dc-1a41-427e-9cfd-f8fd44205790.001.png)





![Chart, waterfall chart

Description automatically generated](Aspose.Words.d8cb79dc-1a41-427e-9cfd-f8fd44205790.002.png)




**3.2 Design and Test Steps**

1. Input data

Input the dataset into TensorFlow and set basic parameters━learning rate, epochs and batch size .

1. Creating model

We will create predem-V1 and predem-V2 model. Train them separately with different data.

1. Optimizing the model 

Improving predem-V1 with the help of predem-V2

1. Training and testing the final model.
1. Displaying the prediction.





**3.3 Algorithms and Pseudo Code**

We will start with a baseline model (Predem-V1) using word embeddings, then build a more sophisticated model (Predem-V2) that further incorporates an LSTM.


**1 - Baseline model: Predem-V1**

**1.1 - Dataset EMOJISET**

We will start by building a simple baseline classifier.

We have a tiny dataset (X, Y) where:

- X contains 127 sentences (strings)
- Y contains a integer label between 0 and 4 corresponding to an emoji for each sentence.


###
### 1.2 - Overview of the Predem-V1
In this part, we are going to implement a baseline model called "Predem-v1".

The input of the model is a string corresponding to a sentence (e.g. "I love we). In the code, the output will be a probability vector of shape (1,5), that we then pass in an argmax layer to extract the index of the most likely emoji output.

To get our labels into a format suitable for training a softmax classifier, lets convert ![$Y$](Aspose.Words.d8cb79dc-1a41-427e-9cfd-f8fd44205790.003.png) from its current shape current shape ![$(m, 1)$](Aspose.Words.d8cb79dc-1a41-427e-9cfd-f8fd44205790.003.png) into a "one-hot representation" ![$(m, 5)$](Aspose.Words.d8cb79dc-1a41-427e-9cfd-f8fd44205790.003.png), where each row is a one-hot vector giving the label of one example, We can do so using this next code snipper. Here, Y\_oh stands for "Y-one-hot" in the variable names Y\_oh\_train and Y\_oh\_test

###
### 1.3 - Implementing Predem-V1
As shown in Figure (2), the first step is to convert an input sentence into the word vector representation, which then get averaged together. Similar to the previous exercise, we will use pretrained 50-dimensional GloVe embeddings. Run the following cell to load the word\_to\_vec\_map, which contains all the vector representations.

####
#### Model : 
You now have all the pieces to finish implementing the model() function. After using sentence\_to\_avg() you need to pass the average through forward propagation, compute the cost, and then backpropagate to update the softmax's parameters.

Implement the model() function described in Figure (2). Assuming here that Y one hot is the one-hot encoding of the output labels, the equations you need to implement in the forward pass and to compute the cross-entropy cost are:

`	`![Text, letter

Description automatically generated](Aspose.Words.d8cb79dc-1a41-427e-9cfd-f8fd44205790.004.png)


**1.4 - Examining test set performance**

Random guessing would have had 20% accuracy given that there are 5 classes. This is pretty good performance after training on only 127 examples.
##
## 2 - Predem-V2: Using LSTMs in Keras:
` `Build an LSTM model that takes as input word sequences. This model will be able to take word ordering into account. Predem-V2 will continue to use pre-trained word embeddings to represent words, but will feed them into an LSTM, whose job it is to predict the most appropriate emoji.

### 2.1 - Overview of the model
Here is the Predem-v2 you will implement:

![Diagram, schematic

Description automatically generated](Aspose.Words.d8cb79dc-1a41-427e-9cfd-f8fd44205790.005.png)


**2.2 Keras and mini-batching**

The common solution to this is to use padding. Specifically, set a maximum sequence length, and pad all sequences to the same length. For example, of the maximum sequence length is 20, we could pad every sentence with "0"s so that each input sentence is of length 20. Thus, a sentence "i love you" would be represented as ![$(e\_{i}, e\_{love}, e\_{you}, \vec{0}, \vec{0}, \ldots, \vec{0})$](Aspose.Words.d8cb79dc-1a41-427e-9cfd-f8fd44205790.003.png). In this example, any sentences longer than 20 words would have to be truncated. One simple way to choose the maximum sequence length is to just pick the length of the longest sentence in the training set.

### 2.3 - The Embedding layer

The Embedding() layer takes an integer matrix of size (batch size, max input length) as input. This corresponds to sentences converted into lists of indices (integers), as shown in the figure below.

![Chart, waterfall chart

Description automatically generated](Aspose.Words.d8cb79dc-1a41-427e-9cfd-f8fd44205790.006.png)

The largest integer (i.e. word index) in the input should be no larger than the vocabulary size. The layer outputs an array of shape (batch size, max input length, dimension of word vectors).

##
## 2.4 Building the Predem-V2
Lets now build the Predem-V2 model. You will do so using the embedding layer you have built, and feed its output to an LSTM network.

![Table

Description automatically generated](Aspose.Words.d8cb79dc-1a41-427e-9cfd-f8fd44205790.007.png)





**3.4 Testing Process**

- An Embedding() layer can be initialized with pretrained values. These values can be either fixed or trained further on your dataset. If however your labelled dataset is small, it's usually not worth trying to train a large pre-trained set of embeddings.

- LSTM() has a flag called return\_sequences() to decide if you would like to return every hidden states or only the last one.

- You can use Dropout() right after LSTM() to regularize your network.













**RESULTS/OUTPUTS**

**OUTPUT 1:**

![Text

Description automatically generated](Aspose.Words.d8cb79dc-1a41-427e-9cfd-f8fd44205790.008.png)




























**CONCLUSIONS**



Multinomial naive bayes performed the best among all the methods so far. We know from our problems sets and lectures that multinomial NB is very good for text classification, such as spam classification, so we had high confidence that naive bayes 4324would provide reasonable results. For the deep learning approaches, we most likely could have tried BERT or XLNet[7] to see if it can overcome the weak semantics. More importantly, many of the related topics on this work employs data on the factor of billions to solve this problem, while we only used less than 200,000 data in total. Therefore, the data isn’t generalizable enough for deep learning to outperform traditional methods.

We did discover that stop emoji is essential to handle uneven distribution. When one emoji dominates the dataset, it will make the classifiers to prefer this emoji. Due to the nature of this problem, the emoji and sentence only have weak semantic relations, wherein any examples which share the same emojis which totally expresses opposite emotion. For future improvements, we should employ an output of more than one prediction with decreasing confidence instead of an absolute single emoji. There were often not 1:1 mapping between an emoji and a sentence or expression. Oftentimes, if we have a user decide on which emojis to use for a similar sentence, the emoji selection would vary quite a bit. In addition, when we calculate accuracy, it may be best to have weighted penalties for determining accuracy. The correlation matrix portrays a lot of the emoji overlap with some good reasoning. Currently our accuracy is just based on absolute correctness, which a very hard ask for a model given the weak semantic nature of this problem.

























**REFERENCES**

`     `Research paper written by -

- Chen Huang (Stanford University)
- Chuang Xueying (Shirley) Xie (Stanford University)
- Boyu (Bill) Zhang (Stanford University)

- Arbitrary Style Transfer in Real-time with Adaptive In-

`           `stance Normalization, Xun Huang, Serge Belongie, arXiv:

`         `1703.06868, 2017

