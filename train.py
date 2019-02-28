import os
import pandas as pd
from collections import Counter

import keras_metrics
import numpy as np
import nltk
from nltk.corpus import stopwords
import os
import re
import sys
import unicodedata
from tqdm import tqdm
import gensim
import string

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Bidirectional
from keras.optimizers import RMSprop
from keras.models import Model

def punc(sentences):
    # import ipdb; ipdb.set_trace()
    new_sentences=[]
    exclude = list(set(string.punctuation))
    exclude.extend(["’", "‘"])
    for sentence in tqdm(sentences):
        s = ''.join(ch for ch in sentence if ch not in exclude)
        new_sentences.append(s)
    
    return new_sentences

def tokenize(sentences):
    
    tokens=[]
    unique_tokens=[]
    
    for sentence in sentences:
        sentence=nltk.word_tokenize(sentence)
        tokens.append(sentence)
        unique_tokens.extend(sentence)
    
    return tokens, unique_tokens

def read_data(filename):
    cls=[]
    text=[]

    with open(os.path.join('data', filename+'.txt'), 'r') as f:
        for line in f:
            cls.append(line.split('||')[0])
            text.append(line.split('||')[1])
    
    return cls, text

def RNN():
    max_len=15
    max_words=20000
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words, 150,input_length=max_len)(inputs)
    layer = Bidirectional(LSTM(64, return_sequences=True))(layer)
    layer = Bidirectional(LSTM(64))(layer)
    layer = Dense(512,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dense(128,name='FC2')(layer)
    layer = Activation('relu')(layer)
    # layer = Dropout(0.2)(layer)
    layer = Dense(5,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

def remove_common(sentences, unique_tokens, top):
    # import ipdb; ipdb.set_trace()
    new_sentences=[]
    common=Counter(unique_tokens).most_common(top)
    common=list(list(zip(*common))[0])
    print(common)

    for sentence in tqdm(sentences):
        words=[word for word in sentence if word not in common]
        new_sentences.append(words)
    
    return new_sentences

def split(df):
    cols=df.cls.drop_duplicates().values
    from sklearn.model_selection import train_test_split
    import sklearn

    train=pd.DataFrame()
    test=pd.DataFrame()

    for col in cols:
        split_df=df[df.cls == col]
        train_df, test_df= train_test_split(split_df, test_size=0.2)
        train=[train, train_df]
        test=[test, test_df]
        train=pd.concat(train)
        test=pd.concat(test)
    
    return sklearn.utils.shuffle(train), sklearn.utils.shuffle(test)


def main():
    # import ipdb; ipdb.set_trace()
    cls, text=read_data('classification')
    
    yo=read_data('anandabazar_classification')
    cls.extend(yo[0])
    text.extend(yo[1])

    yo=read_data('ebala_classification')
    cls.extend(yo[0])
    text.extend(yo[1])
    # import ipdb; ipdb.set_trace()
    yo=list(set(zip(cls, text)))
    df=pd.DataFrame(yo, columns=['cls', 'text'])
    df=df.replace(['international', 'sport', 'nation'], ['world', 'sports', 'national'])
    df=df[df.cls!='travel']
    df=df[df.cls!='world']
    print(Counter(df.cls.values))

    train_df, test_df=split(df)

    train_comments=train_df.text.values
    train_labels=train_df.cls.values
    train_comments=punc(train_comments)
    # comments=remove_stopwords(comments)
    train_comments, unique_tokens=tokenize(train_comments)
    train_comments=remove_common(train_comments, unique_tokens, 20)

    test_comments=test_df.text.values
    test_labels=test_df.cls.values
    test_comments=punc(test_comments)
    # comments=remove_stopwords(comments)
    test_comments, _ =tokenize(test_comments)
    test_comments=remove_common(test_comments, unique_tokens, 20)

    print(f"Most Common Words: {Counter(unique_tokens).most_common(10)}")
    print(f"{len(list(zip(unique_tokens)))}")

    # print(comments[:5])

    max_words = 20000
    max_len = 15
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(train_comments)
    sequences = tok.texts_to_sequences(train_comments)
    test_sequences=tok.texts_to_sequences(test_comments)
    print(sequences[0])
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
    test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)
    print(test_sequences_matrix[0])
    train_labels = pd.Series(train_labels).str.get_dummies()
    test_labels=pd.Series(test_labels).str.get_dummies()
    train_labels=np.asarray(train_labels)
    test_labels=np.asarray(test_labels)

    import pickle
    with open('tok.pkl', 'wb') as f:
        pickle.dump(tok, f, pickle.HIGHEST_PROTOCOL)
    

    model = RNN()
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
    filepath=f'model.hdf5'
    callback=keras.callbacks.ModelCheckpoint(filepath, monitor='acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model_hist=model.fit(sequences_matrix,train_labels, validation_data=(test_sequences_matrix, test_labels), batch_size=128, epochs=10, shuffle=True, callbacks=[callback])
    model.save('models/model1.h5')

    precision = model_hist.history['precision'][0]
    recall = model_hist.history['recall'][0]
    f_score = (2.0 * precision * recall) / (precision + recall)
    print('F1-SCORE {}'.format(f_score))

    model.evaluate(test_sequences_matrix, test_labels, verbose=0)

    


if __name__ == "__main__":
    main()