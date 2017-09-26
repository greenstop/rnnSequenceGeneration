
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from my_answers import *

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from my_answers import *

dataset = np.loadtxt('datasets/normalized_apple_prices.csv')

plt.plot(dataset)
plt.xlabel('time period')
plt.ylabel('normalized series value')

odd_nums = np.array([1,3,5,7,9,11,13])

window_size = 2

X = []
X.append(odd_nums[0:2])
X.append(odd_nums[1:3])
X.append(odd_nums[2:4])
X.append(odd_nums[3:5])
X.append(odd_nums[4:6])
X.append(odd_nums[5:7])

y = odd_nums[2:]

X = np.asarray(X)
y = np.asarray(y)
y = np.reshape(y, (len(y),1)) #optional

assert(type(X).__name__ == 'ndarray')
assert(type(y).__name__ == 'ndarray')
assert(X.shape == (6,2))
assert(y.shape in [(5,1), (5,)])

print ('--- the input X will look like ----')
print (X)

print ('--- the associated output y will look like ----')
print (y)

from my_answers import window_transform_series

window_transform_series(series = odd_nums,window_size = 2)

window_size = 7
X,y = window_transform_series(series = dataset,window_size = window_size)
print(X);print(y);

train_test_split = int(np.ceil(2*len(y)/float(3)))   # set the split point

X_train = X[:train_test_split,:]
y_train = y[:train_test_split]

X_test = X[train_test_split:,:]
y_test = y[train_test_split:]

X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1)))
X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, 1)))

train_test_split = int(np.ceil(len(y)/float(3)))   # set the split point

X_train = X[train_test_split:,:]
y_train = y[train_test_split:]

X_test = X[:train_test_split,:]
y_test = y[:train_test_split]

X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1)))
X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, 1)))
print(X_train.shape)
print(X_test.shape)
print(train_test_split);

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

np.random.seed(0)

from my_answers import build_part1_RNN
model = build_part1_RNN(window_size)

optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss='mean_squared_error', optimizer=optimizer)

model.summary();

model.fit(X_train, y_train, epochs=1000, batch_size=50, verbose=0)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

training_error = model.evaluate(X_train, y_train, verbose=0)
print('training error = ' + str(training_error))

testing_error = model.evaluate(X_test, y_test, verbose=0)
print('testing error = ' + str(testing_error))

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.plot(dataset,color = 'k')

split_pt = train_test_split + window_size 
plt.plot(np.arange(window_size,split_pt,1),train_predict,color = 'b')

plt.plot(np.arange(split_pt,split_pt + len(test_predict),1),test_predict,color = 'r')

plt.xlabel('day')
plt.ylabel('(normalized) price of Apple stock')
plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.plot(dataset,color = 'k')

plt.plot(np.arange(train_test_split+window_size,window_size+train_test_split+len(train_predict),1),train_predict,color = 'b')

plt.plot(np.arange(window_size,window_size+len(test_predict),1),test_predict,color = 'r')

plt.xlabel('day')
plt.ylabel('(normalized) price of Apple stock')
plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

print(dataset.shape, train_predict.shape, test_predict.shape);
print(np.arange(train_test_split+window_size,window_size+train_test_split+len(train_predict),1))
print(np.arange(window_size,window_size+len(test_predict),1))

text = open('datasets/holmes.txt').read().lower()
print('our original text has ' + str(len(text)) + ' characters')

text[:2000]

text = text[1302:]
text = text.replace('\n',' ')    # replacing '\n' with '' simply removes the sequence
text = text.replace('\r',' ')

text[:1000]

from my_answers import cleaned_text

text = cleaned_text(text)

text = text.replace('  ',' ')

text[:2000]

chars = sorted(list(set(text)))

print ("this corpus has " +  str(len(text)) + " total number of characters")
print ("this corpus has " +  str(len(chars)) + " unique characters")

from my_answers import window_transform_text

window_size = 100
step_size = 5
inputs, outputs = window_transform_text(text,window_size,step_size)

window_size = 100
step_size = 10
inputs, outputs = window_transform_text("a"*192,window_size,step_size)

print('input = ' + inputs[0])
print('output = ' + outputs[0])
print('--------------')
print('input = ' + inputs[1])
print('output = ' + outputs[1])
print('--------------')
print('input = ' + inputs[2])
print('output = ' + outputs[2])
print('--------------')
print('input = ' + inputs[100])
print('output = ' + outputs[100])

chars = sorted(list(set(text)))
print ("this corpus has " +  str(len(chars)) + " unique characters")
print ('and these characters are ')
print (chars)

chars_to_indices = dict((c, i) for i, c in enumerate(chars))  # map each unique character to unique integer

indices_to_chars = dict((i, c) for i, c in enumerate(chars))  # map each unique integer back to unique character

def encode_io_pairs(text,window_size,step_size):
    # number of unique chars
    chars = sorted(list(set(text)))
    num_chars = len(chars)
    
    # cut up text into character input/output pairs
    inputs, outputs = window_transform_text(text,window_size,step_size)
    
    # create empty vessels for one-hot encoded input/output
    X = np.zeros((len(inputs), window_size, num_chars), dtype=np.bool)
    y = np.zeros((len(inputs), num_chars), dtype=np.bool)
    
    # loop over inputs/outputs and tranform and store in X/y
    for i, sentence in enumerate(inputs):
        for t, char in enumerate(sentence):
            X[i, t, chars_to_indices[char]] = 1
        y[i, chars_to_indices[outputs[i]]] = 1
        
    return X,y

window_size = 100
step_size = 5
X,y = encode_io_pairs(text,window_size,step_size)

print(X.shape,y.shape);
print(X[0].shape);
print(y[0].shape);

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import keras
import random

from my_answers import build_part2_RNN

model = build_part2_RNN(window_size, len(chars))

optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary();

Xsmall = X[:10000,:,:]
ysmall = y[:10000,:]

model.fit(Xsmall, ysmall, batch_size=500, epochs=40,verbose = 1)

model.save_weights('model_weights/best_RNN_small_textdata_weights.hdf5')

def predict_next_chars(model,input_chars,num_to_predict):     
    # create output
    predicted_chars = ''
    for i in range(num_to_predict):
        # convert this round's predicted characters to numerical input    
        x_test = np.zeros((1, window_size, len(chars)))
        for t, char in enumerate(input_chars):
            x_test[0, t, chars_to_indices[char]] = 1.

        # make this round's prediction
        test_predict = model.predict(x_test,verbose = 0)[0]

        # translate numerical prediction back to characters
        r = np.argmax(test_predict)                           # predict class of each test input
        d = indices_to_chars[r] 

        # update predicted_chars and input
        predicted_chars+=d
        input_chars+=d
        input_chars = input_chars[1:]
    return predicted_chars

start_inds = [4,7,11]

model.load_weights('model_weights/best_RNN_small_textdata_weights.hdf5')
for s in start_inds:
    start_index = s
    input_chars = text[start_index: start_index + window_size]

    # use the prediction function
    predict_input = predict_next_chars(model,input_chars,num_to_predict = 100)

    # print out input characters
    print('------------------')
    input_line = 'input chars = ' + '\n' +  input_chars + '"' + '\n'
    print(input_line)

    # print out predicted characters
    line = 'predicted chars = ' + '\n' +  predict_input + '"' + '\n'
    print(line)

f = open('my_test_output.txt', 'w')              # create an output file to write too
f.write('this is only a test ' + '\n')           # print some output text
x = 2
f.write('the value of x is ' + str(x) + '\n')    # record a variable value
f.close()     

f = open('my_test_output.txt', 'r')              # create an output file to write too
f.read()

Xlarge = X[:100000,:,:]
ylarge = y[:100000,:]

model.fit(Xlarge, ylarge, batch_size=500, nb_epoch=30,verbose = 1)

model.save_weights('model_weights/best_RNN_large_textdata_weights.hdf5')

start_inds = [1,3,7437]

f = open('text_gen_output/RNN_large_textdata_output.txt', 'w')  # create an output file to write too

model.load_weights('model_weights/best_RNN_large_textdata_weights.hdf5')
for s in start_inds:
    start_index = s
    input_chars = text[start_index: start_index + window_size]

    # use the prediction function
    predict_input = predict_next_chars(model,input_chars,num_to_predict = 100)

    # print out input characters
    line = '-------------------' + '\n'
    print(line)
    f.write(line)

    input_line = 'input chars = ' + '\n' +  input_chars + '"' + '\n'
    print(input_line)
    f.write(input_line)

    # print out predicted characters
    predict_line = 'predicted chars = ' + '\n' +  predict_input + '"' + '\n'
    print(predict_line)
    f.write(predict_line)
f.close()






