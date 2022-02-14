


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv(r"C:\Users\Roshan Kumar\Downloads\Datasets\MBTI 500.csv\MBTI 500.csv")




df





label = df.type.factorize()
label




text = df.posts.values





from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(text)





encoded_docs = tokenizer.texts_to_sequences(text)
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_sequence = pad_sequences(encoded_docs, maxlen=50)




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding




embedding_vector_length = 32
vocab_size=1000
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=50))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(16, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
             metrics=["accuracy"])
print(model.summary)




print(model.summary())





history = model.fit(padded_sequence, label[0], validation_split=0.2, epochs=5, batch_size=32)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

plt.savefig("Accuracy plot.jpg")

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.savefig("Loss plt.jpg")

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=50)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ", label[1][prediction])
test_sentence1 = "I enjoyed staying alone."
predict_sentiment(test_sentence1)
test_sentence2 = "I like to kick the shit out of people"
predict_sentiment(test_sentence2)










