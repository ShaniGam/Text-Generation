# coding=utf8
from __future__ import print_function
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop

import numpy as np
import codecs

# load the files
train = codecs.open('train.txt', 'r', encoding='utf8').read()
chars = set(train)
test = codecs.open('test.txt', 'r', encoding='utf8').read()

# Char to index
char_indices = dict((c, i) for i, c in enumerate(chars))

# Index to Char
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 50

#split the data into sentences and create one hot vector for each character
def read_data(text):
    X, Y = [], []
    text_len = int(len(text) / maxlen)
    for i in range(text_len):
        X.append(text[i * maxlen:(i + 1) * maxlen])
        Y.append(text[i * maxlen + 1:(i + 1) * maxlen + 1])

    x = np.zeros((len(X), maxlen, len(char_indices)))
    for i in range(len(X)):
        for j in range(len(X[i])):
            idx = char_indices[X[i][j]]
            x[i][j][idx] = 1.

    y = np.zeros((len(X), maxlen, len(char_indices)))
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            idx = char_indices[Y[i][j]]
            y[i][j][idx] = 1.
    return x, y

train_x, train_y = read_data(train)
test_x, test_y = read_data(test)


# create a model with LSTM and Dropout
def buildModel():
    print('Build model...')
    model = Sequential()
    model.add(LSTM(512, input_shape=(None, len(char_indices)), return_sequences=True, implementation=2))
    model.add(Dropout(0.2))
    model.add(Dense(len(char_indices)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model


# train the model, calculate the accuracy and generate new text
def train_model(model):
    # For each epoch
    for iteration in range(20):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        # Runs one epoch
        model.fit(train_x, train_y, batch_size=64, epochs=1, shuffle=False)

    generate_text(model, iteration)
    calculate_accuracy(iteration, model)


# generate new text
def generate_text(model, iteration, generate_size=10000):
    # characters to start the text with
    start_test = train[:32]
    sentence = [char_indices[c] for c in start_test]
    generated_text = start_test
    X = np.zeros((1, generate_size, len(char_indices)))
    for i in range(generate_size):
        X[0, i, :][sentence[-1]] = 1
        start = max(0, i - maxlen)
        preds = model.predict(X[:, start:i + 1, :])[0][-1]
        sentence = np.random.choice(len(char_indices), 1, p=preds)
        generated_text += indices_char[sentence[-1]]
    # write the generate text into a file
    with codecs.open("Epoch" + str(iteration) + ".txt", "w+", encoding='utf8') as f:
        f.write(generated_text[:10000])


def calculate_accuracy(iteration, model):
    # Initializes
    cross_entropy = 0.0
    success = 0.0
    test_length = len(test_x)
    predictions = model.predict(test_x)
    counter = 0

    # check the accuracy and cross entropy
    for i in range(test_length):
        prediction = predictions[i]
        for j in range(maxlen):
            pred = np.argmax(prediction[j])
            correct_char = np.argmax(test_y[i][j])
            cross_entropy -= np.log2(prediction[j][correct_char])
            if test_y[i][j][pred] == 1:
                success += 1.
            counter += 1

    success /= counter
    cross_entropy /= counter
    # write the results into a file
    print("Success: " + str(success))
    print("Cross entropy: " + str(cross_entropy))
    resultTest = open("TestResult" + str(iteration) + "_" + str(success) + "_" + str(cross_entropy) + ".txt", "w+")
    resultTest.write("Success: " + str(success) + "\n")
    resultTest.write("Cross entropy: " + str(cross_entropy) + "\n")

    # close the results file
    resultTest.close()

    # save the model
    model.save("model.h5")


def main():
    # Build the model
    model = buildModel()
    # Start the training
    train_model(model)


if __name__ == "__main__":
    main()
