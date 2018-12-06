'''simpleRNN for IMDB segtiment analysis val acc 82.4%'''

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import SimpleRNN
from keras.layers import Dense,Embedding

#Preparing the IMDB data
max_features=10000 #字典大小
maxlen=500
batch_size=32

(input_train,y_train),(input_test,y_test)=imdb.load_data(num_words=max_features)
input_train=sequence.pad_sequences(input_train,maxlen=maxlen) #填充处理
input_test=sequence.pad_sequences(input_test,maxlen=maxlen)

#Training the model with Embedding and SimpleRNN layers
model=Sequential()
model.add(Embedding(max_features,32))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(input_train,y_train,
            epochs=10,
            batch_size=128,
            validation_split=0.2)

#Plotting results
import matplotlib.pyplot as plt 

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Val acc')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Val loss')
plt.legend()

plt.show()