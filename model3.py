import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from keras.constraints import maxnorm
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

f = open("characters_dataset", "rb")
X_train = np.load(f)
y_train = np.load(f)
X_test = np.load(f)
y_test = np.load(f)
label_names = np.load(f)
f.close()

"""### Rescaling and Reshape Data"""

# Melakukan reshape data menjadi 4 dimensi (batch size, width, weight, dan channel)
# X_train = X_train.reshape(X_train.shape[0], 96, 96, 1)
# X_test = X_test.reshape(X_test.shape[0], 96, 96, 1)

# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')

# X_train /= 255
# X_test /= 255

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

print('x_train shape : ', X_train.shape)
print('Banyak gambar di x_train : ', X_train.shape[0])

print('Banyak gambar di x_test', X_test.shape[0])
print('Banyak gambar di y_test', y_train.shape[0])

NUMBER_OF_CLASSES = len(label_names)
input_shape = (96,96,1)
MAX_NORM = 4
INITIAL_ADAM_LEARNING_RATE = 0.01

depth = 30
model = Sequential()
model.add(Convolution2D(
    depth, 5, 5, border_mode='same',
    W_constraint=maxnorm(MAX_NORM),
    init='he_normal',
    input_shape=input_shape))
model.add(BatchNormalization())
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

depth *= 2
model.add(Convolution2D(
    depth, 5, 5, init='he_normal',
    border_mode='same', W_constraint=maxnorm(MAX_NORM)))
model.add(BatchNormalization())
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

depth *= 2
model.add(Convolution2D(
    depth, 5, 5, init='he_normal',
    border_mode='same', W_constraint=maxnorm(MAX_NORM)))
model.add(BatchNormalization())
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(2000, init='he_normal', W_constraint=maxnorm(MAX_NORM)))
model.add(BatchNormalization())
model.add(PReLU())
model.add(Dropout(0.5))

model.add(Dense(2000, init='he_normal', W_constraint=maxnorm(MAX_NORM)))
model.add(BatchNormalization())
model.add(PReLU())
model.add(Dropout(0.5))

model.add(Dense(NUMBER_OF_CLASSES, W_constraint=maxnorm(MAX_NORM)))
model.add(Activation('softmax'))

adam = Adam(lr=INITIAL_ADAM_LEARNING_RATE)
model.compile(
    loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

"""
# Membuat model
model = Sequential()

model.add(Conv2D(filters=64, 
                 kernel_size=2, 
                 padding='same', 
                 activation='relu', 
                 strides=1, 
                 input_shape=input_shape))
# Conv2D
# - filters = jumlah node. mirip seperti Dense
# - kernel_size = jumlah node yang bergeser. 2 artinya matrik 2x2
# - strides = bergesernya pixel ke kanan sebanyak 1 kolom
# - padding = merupakan garis putus (sisa/tambahan pixel dari hasil stride) untuk membantu proses perhitungan
#       => value 'same' artinya padding akan diberikan sebagaimana hasil proses perhitungan sama dengan input 
#       => input 28x28x1 maka menjadi 28x28x64. 28x28 tidak berubah karena 'same'

model.add(MaxPooling2D(pool_size=2))
# MaxPooling2D
# - untuk memperkecil ukuran dari Conv2D dengan cara mengambil angka tersebar untuk setiap setingan pixel matrik dalam hal ini 2x2 dalam setiap proses
# - dapat mencegah overfitting

model.add(Flatten())
# Flatten
# - meratakan semua output dari layer sebelumnya sehingga bisa diproses menggunakan Dense
# - hasil output MaxPool adalah 14x14x64 dan flatten akan meratakan menjadi 12.544 sel
model.add(Dense(20, activation='softmax'))
"""

"""### Fitting Data"""

#adam = Adam(lr=0.001)
n_epochs = 50
n_bs = 96

# - menggunakan loss function 'categorical_crossentropy' untuk multikategori,
# - adam sebagai optimizer karena performanya yang baik dengan learning rate digunakan 0.001,
# - dan menambahkan paramater metrics 'accuracy' untuk menilai performa model

model.compile(
    optimizer = adam,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)


# Melakukan training sebanyak 50 kali dengan pembagian validasi data training-test 90%-10% dan batch size 256
history = model.fit(
    X_train, 
    y_train, 
    epochs=n_epochs, 
    validation_split=0.1, 
    batch_size=n_bs)

plt.figure(figsize=(15,6))

plt.subplot(1,2,2)
plt.title('Loss Test')
plt.plot(history.history['loss'], label = "Loss")
plt.plot(history.history['val_loss'], label = "Validation Loss")
plt.ylabel("Loss")
plt.xlabel("Number of Epochs")
plt.legend()

# Menampilkan perbedaan akurasi model dengan akurasi tes dalam bentuk grafik
plt.subplot(1,2,1)
plt.title('Accuracy Test')
plt.plot(history.history['acc'], label = "Accuracy")
plt.plot(history.history['val_acc'], label = "Validation Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Number of Epochs")
plt.legend()
plt.show()

# Melihat akurasi setelah melakukan fitting data
print("Accuracy after fitting: {:.2f}%".format(history.history['acc'][-1]*100))

# Melakukan evaluasi akurasi data dari data test
score = model.evaluate(X_test, y_test)
print('\nTest Accurary : {:.2f}%'.format(score[1]*100))
print('Test Lost : {:.3f}'.format(score[0]) )