import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    for i in range(len(series)-window_size):
        X.append(series[i:i+window_size]);
        y.append(series[i+window_size]);

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential();
    model.add(LSTM(5,input_shape=(window_size,1)));
    model.add(Dense(1,activation="tanh"));
    return model;

### return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    #punctuation = ['!', ',', '.', ':', ';', '?']
    import re;
    return re.sub(r'[^a-z!,.:;? ]',r'',text);

### : fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range( (len(text)-window_size)//step_size + 1 ): #divide text into possible step-sized pieces #added +1
        inputs.append(text[i*step_size:i*step_size+window_size]); #e.g. text[5:100+5]
        outputs.append(text[i*step_size+window_size]); #e.g. text[100+5]

    print("len",len(inputs),len(outputs));
    print("shape",len(inputs[0]),outputs[0:5]);

    return inputs,outputs

#  build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential();
    #layer 1 LSTM
    model.add(LSTM(200,input_shape=(window_size,num_chars)));
    #layer 2 and 3 combined, "linear" (Dense) layer plus the softmax activations layer 
    model.add(Dense(num_chars,activation="softmax"));
    return model;
