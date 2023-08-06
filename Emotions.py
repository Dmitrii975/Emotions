from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LeakyReLU, Dropout, MaxPooling2D
colors = np.random.randint(0, 255, (10, 3))

def prepare_image(der):
    global colors
    read_image = cv2.imread(der)
    image_gray = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image_gray, (200, 200))
    
    model = KMeans(n_clusters = 5, n_init='auto')
    model.fit(np.reshape(image, (40000, 3)))
    
    prediction = model.predict(np.reshape(image, (40000, 3)))
    
    arr = np.zeros((200,200), dtype=int)
    the_popular = np.bincount(prediction).argmax()
    
    prediction = np.reshape(prediction, (200, 200))
    
    for i in range(200):
        for j in range(200):
            cluster = prediction[i][j]
            if cluster == the_popular:
                arr[i][j] = 1
    return arr
x_dataset = []
y_dataset = []
der = 'C:\\Users\\Dmitry\\Desktop\\data\\Angry\\'
for i in os.listdir(der):
    x_dataset.append(prepare_image(der + i))
    y_dataset.append([1,0,0])

der = 'C:\\Users\\Dmitry\\Desktop\\data\\Sad\\'
for i in os.listdir(der):
    x_dataset.append(prepare_image(der + i))
    y_dataset.append([0,1,0])

der = 'C:\\Users\\Dmitry\\Desktop\\data\\Happy\\'
for i in os.listdir(der):
    x_dataset.append(prepare_image(der + i))
    y_dataset.append([0,0,1])

x_dataset = np.array(x_dataset)
np.random.shuffle(x_dataset)
y_dataset = np.array(y_dataset)

#model = Sequential()
#model.add(Conv2D(128, (3,3), input_shape=(200,200,1), activation='relu'))
#model.add(MaxPooling2D((2,2)))

#model.add(Dropout(0.2))
#model.add(Flatten())

#model.add(Dense(128, activation='relu'))
#model.add(LeakyReLU())

#model.add(Dropout(0.2))

#model.add(Dense(64, activation='relu'))
#model.add(LeakyReLU())

#model.add(Dense(3, activation='softmax'))

#model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#model.build((200,200,1))

#model.summary()
#model.fit(x_dataset, y_dataset, epochs=10, batch_size = 32)
#model.save('C:\\Users\\Dmitry\\Desktop\\cv_model.keras')

model = load_model('C:\\Users\\Dmitry\\Desktop\\cv_model.keras')
data = prepare_image('C:\\Users\\Dmitry\\Desktop\\1.jpg')
data = np.expand_dims(data, 0)
print(model.predict(data))
