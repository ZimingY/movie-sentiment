#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[17]:


folder = 'aclImdb'
labels = {'pos':1, 'neg':0}
df = pd.DataFrame()
for f in ('test','train'):
    for l in ('pos','neg'):
        path = os.path.join(folder,f,l)
        for file in os.listdir(path):
            if file.endswith(".txt"):
                with open(os.path.join(path, file),'r',encoding = 'utf-8') as infile: # use 'rb' since it is text
                    txt = infile.read()
                df = df.append( [[txt, labels[l]] ], ignore_index = True)

df.columns = ['review', 'sentiment']


# In[18]:


# save to csv
df.to_csv('movie_data.csv', index = False, encoding = 'utf-8')


# In[19]:


df.head()


# In[20]:


df = pd.DataFrame()
df = pd.read_csv('movie_data.csv', encoding = 'utf-8')
df.head(3)


# In[21]:


df.shape


# In[ ]:


### naive bayes model


# In[ ]:


#------
X_train = df.loc[:24999,'review'].values
y_train = df.loc[:24999,'sentiment'].values
X_test = df.loc[25000:,'review'].values
y_test = df.loc[25000:,'sentiment'].values
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
print(train_vectors.shape,test_vectors.shape)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_vectors,y_train)
from sklearn.metrics import accuracy_score
predicted = clf.predict(test_vectors)
print(accuracy_score(y_test,predicted))

test1 = 'this movie is bad! really hate it'
test2 = 'i love the movie'
test3 = 'not to my taste, will skip'
test4 = 'if you like action, then this movie might be good for you'
test_samples = [test1,test2,test3,test4]
test_sample_vector = vectorizer.transform(test_samples)
clf.predict(test_sample_vector)
#------


# In[ ]:





# In[ ]:


# 


# In[22]:


X_train = df.loc[:24999,'review'].values
y_train = df.loc[:24999,'sentiment'].values
X_test = df.loc[25000:,'review'].values
y_test = df.loc[25000:,'sentiment'].values

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

total_reviews = X_train +X_test
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(total_reviews)

max_length = max( [len(s.split()) for s in df['review'].values.tolist() ] )

vocab_size = len(tokenizer_obj.word_index) + 1
X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)
X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_tokens, maxlen = max_length,padding = 'post')

X_test_pad = pad_sequences(X_test_tokens, maxlen = max_length,padding = 'post')


# In[49]:





# In[55]:





# In[50]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[23]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
embedding_dim = 100
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.backend import set_session
tf.keras.backend.clear_session()  # For easy reset of notebook state.

config_proto = tf.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.graph_options.rewrite_options.arithmetic_optimization = off
session = tf.Session(config=config_proto)
set_session(session)


model = Sequential()
model.add( Embedding(vocab_size, embedding_dim, input_length = max_length))
model.add( GRU(units = 32, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(1,activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[24]:


model.summary()


# In[25]:


model.fit(X_train_pad,y_train, batch_size=128, 
          epochs=25,validation_data= (X_test_pad,y_test), verbose = 2)


# In[ ]:


test1 = 'this movie is fantastic! really like it'
test2 = 'good movie'
test3 = 'not to my taste, will skip'
test4 = 'if you like action, then this movie might be good for you'
test_samples = [test1,test2,test3,test4]
test_sample_token = tokenizer_obj.texts_to_sequences(test_samples)
test_pad = pad_sequences(test_sample_token, maxlen = max_length)
model.predict(x = test_pad)


# In[ ]:





# In[ ]:





# In[ ]:





# In[26]:


# learn word embedding via word2 vec
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


stop_words = set( stopwords.words('english') )
lines = df['review'].values.tolist()
translator = str.maketrans('','',string.punctuation  )
review_line = []
for line in lines:
    tokens = word_tokenize(line) # tokenize
    tokens = [w.lower() for w in tokens] # lower case
    stripped = [w.translate(translator) for w in tokens] # remove punctuation
    
    words = [word for word in stripped if word.isalpha()] # remove non alphabetic
    words = [w for w in words if not w in stop_words]
    review_line.append(tokens)

    


# In[28]:


len(review_line)


# In[29]:


# train word2vec
import gensim
embed_dim = 100
model = gensim.models.Word2Vec( sentences = review_line, size =  embed_dim, window=5, min_count=1, workers=4)
# vocab size
words = list(model.wv.vocab)
print(len(words))


# In[30]:


# save model
model.wv.save_word2vec_format('w2v.txt', binary = False)


# In[31]:


f = open('w2v.txt')
embeddings = {}
with open('w2v.txt', encoding = 'utf-8') as f:
#     next(f)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray( values[1:] )
        embeddings[ word ] = coefs
f.close()


# In[ ]:





# In[32]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(review_line)


# In[33]:


sequences = tokenizer_obj.texts_to_sequences(review_line) # to tensor
word_index = tokenizer_obj.word_index # get the index of text 
max_length = max( [len(line.split()) for line in df['review'].values.tolist()] )
pad = pad_sequences(sequences,maxlen = max_length) # pad text to max length


# In[34]:


print('unique token',len(word_index))
print('shape of review tensor',pad.shape)
embedding_matrix = np.zeros( (len(word_index)+1,embed_dim ) )
for word,i in word_index.items():
    if i>len(word_index)+1:
        continue 
    if embeddings[word] is not None:
        embedding_matrix[i] = embeddings[word]
        


# In[35]:


from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.initializers import Constant
model = Sequential()
embedding_layer = Embedding( len(word_index)+1,
                            embed_dim, 
                            embeddings_initializer = Constant(embedding_matrix),
                            input_length = max_length,
                           trainable = False)
model.add(embedding_layer)
model.add(GRU(units = 32, dropout = 0.2, recurrent_dropout=0.2))
model.add(Dense(1,activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics = ['accuracy'])


# In[36]:


model.summary()


# In[37]:


validation_split = 0.2
num_validation = int( validation_split * pad.shape[0] )
sentiment = df['sentiment'].values
# shuffle the index
index = np.arange( pad.shape[0] )
np.random.shuffle(index)
pad = pad[index]
sentiment = sentiment[index]

x_train = pad[:-num_validation]
y_train = sentiment[:-num_validation]
x_test = pad[-num_validation:]
y_test = sentiment[-num_validation:]


# In[38]:


print(y_test.shape)
print(x_train.shape)


# In[39]:


model.fit(x_train,y_train,batch_size=128, 
          epochs = 25, validation_data=(x_test, y_test),
         verbose = 2)


# In[ ]:


test1 = 'this movie is fantastic! really like it'
test2 = 'good movie'
test3 = 'not to my taste, will skip'
test4 = 'if you like action, then this movie might be good for you'
test_samples = [test1,test2,test3,test4]
test_sample_token = tokenizer_obj.texts_to_sequences(test_samples)
test_pad = pad_sequences(test_sample_token, maxlen = max_length)
model.predict(x = test_pad)


# In[ ]:




