# autocomplete

In this repo we will see how can we predict/complete the next word based on some of the previous words. It's use we can see how we see sugestions on our keyboard when we start typing. 


Here in this work we have used  RNNs (Recurrent Neural Networks). Why RNN here, because they give advantage of one important thing that is MEMORY. In other words RNNs provide to compute the current input but also the input which was provided in the previous step. 

![RNN](https://miro.medium.com/max/1400/1*K6s4Li0fTl1pSX4-WPBMMA.jpeg)

So, here what we will be doing we will pass a sequence of let's say 40 characters and ask the model to predict the next word. Then we will append the new characater and drop the first one and this will go on until we complete a whole word. 

Problems with RNNs -- 

Two major problems with RNNs are vanishing gradient & exploding gradient. In RNNs the gradient can be multiplied a large number of times by the weight matrix. If the weights in the matrix are small, the gradient signal becomes smaller at every training step, thus making learning very slow or even stops it. This is called vanishing gradient. Let's take a look at applying the sigmoid function multiple times, thus simulating the effect of vanishing gradient.

![RNN](https://miro.medium.com/max/1104/1*XbVjM9cPb-BkLrWGNujEQg.png)

And exploding gradient refers to the weights in this matrix so large that it can cause the learning to diverge.

LSTM --

LSTM model is a special kind of RNN, which can learn long-term dependencies. It has different structure than RNNs, the memory cell composed of four elements, input, forget and output gate & a neuron that connects to itself.

![LSTM](https://miro.medium.com/max/1212/1*ZskkUQCNT0i_00shHYSj1A.png)

Here, we will use english NCERT as a training corpus for our model. Thing we need to make -- character to index and intex to character mapping in this. And we cut our corpus into chunks of let's say 40 characters and we will also store the character that is the one we need to predict for every sequence. 

For generating our features and labels we will use previously generated sequences (of 40) and the character we need to predict into one hot encoded vector using character to index dictionary.

-- Making Model

The model we are going to train is single LSTM layer with 128 (taken from a blog), which accepts the size of -- 40 the length of the sequence and x the number of unique characters in our in our dataset. A fully connected layer is added after that (for our output), which has x (same as the number of unique characters) and softmax for our activation function. We are training our model for 20 epochs using RMSProp optimizer and uses 5% of our training dataset for validation.

-- Results

![1](https://github.com/vyaslkv/text-Autocomplete-LSTM-Keras-TF/blob/master/ac1.png)

