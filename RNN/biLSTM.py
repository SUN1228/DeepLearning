'''bidirection LSTM'''
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential

max_features=10000 #字典大小
maxlen=500         #序列最大长度

#Load data
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)

#Pad sequences
x_train=sequence.pad_sequences(x_train,maxlen=maxlen)
x_test=sequence.pad_sequences(x_test,maxlen=maxlen)

#Build model
model=Sequential()
model.add(layers.Embedding(max_features,128))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(1,activation='sigmoid'))

#Compile model
model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['acc'])

history=model.fit(x_train,y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

