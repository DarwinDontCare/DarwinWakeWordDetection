import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pylab as plt
from os import listdir
from os.path import isdir, join
import itertools
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

dataset_path = 'data'
ataset_path = r"data/audio_data.npz"
all_targets = all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
print(all_targets)

# Settings
feature_sets_path = 'data'
feature_sets_filename = 'audio_data.npz'
wake_word = 'wake_word'

# Load feature sets
feature_sets = np.load(join(feature_sets_path, feature_sets_filename))
print(feature_sets.files)

# Assign feature sets
X_train = feature_sets['x_train']
y_train = feature_sets['y_train']
X_val = feature_sets['x_val']
y_val = feature_sets['y_val']
X_test = feature_sets['x_test']
y_test = feature_sets['y_test']
# Look at tensor dimensions

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

# Peek at labels
print(y_val)

# Convert ground truth arrays to one wake word (1) and 'other' (0)
wake_word_index = all_targets.index(wake_word)
y_train = np.equal(y_train, wake_word_index).astype('float64')
y_val = np.equal(y_val, wake_word_index).astype('float64')
y_test = np.equal(y_test, wake_word_index).astype('float64')

# Peek at labels after conversion
print(y_val)

# What percentage of 'stop' appear in validation labels
print(sum(y_val) / len(y_val))
print(1 - sum(y_val) / len(y_val))

# View the dimensions of our input data
print(X_train.shape)

# CNN for TF expects (batch, height, width, channels)
# So we reshape the input tensors with a "color" channel of 1
X_train = X_train.reshape(X_train.shape[0], 
                          X_train.shape[1], 
                          X_train.shape[2], 
                          1)
X_val = X_val.reshape(X_val.shape[0], 
                      X_val.shape[1], 
                      X_val.shape[2], 
                      1)
X_test = X_test.reshape(X_test.shape[0], 
                        X_test.shape[1], 
                        X_test.shape[2], 
                        1)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

# Input shape for CNN is size of MFCC of 1 sample
sample_shape = X_test.shape[1:]
print(sample_shape)

# Normalização de Dados
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

model = models.Sequential()
model.add(layers.Conv2D(32, 
                        (2, 2), 
                        activation='relu',
                        input_shape=sample_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Classifier
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# Learning Callbacks
initial_learning_rate = 0.001
lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=10000, decay_rate=0.9)
optimizer = 'rmsprop'

model.compile(loss='binary_crossentropy', 
              optimizer=optimizer, 
              metrics=['acc'])

# Callbacks
checkpoint = ModelCheckpoint("saved_model/best_score_model.h5", save_best_only=True)
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

print("Model summary:")
print(model.summary())

print("Model score:")

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stopping]
)

tensorflow.keras.models.save_model(filepath="saved_model/WakeWordModel.pb", model=model)

print("Model score:")

score = model.evaluate(X_test, y_test)
print(score)

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


#print("Model Classification Report:")
#y_pred = np.argmax(model.predict(X_test), axis=1)
#cm = confusion_matrix(np.argmax(y_test), y_pred)
#print(classification_report(np.argmax(y_test, axis=1), y_pred))
    
#plot_confusion_matrix(cm, classes=["Does not have Wake Word", "Has Wake Word"])