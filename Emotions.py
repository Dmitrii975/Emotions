from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2

def prepare_image(der):
    read_image = cv2.imread(der)
    image_gray = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image_gray, (200, 200))
    
    model = KMeans(n_clusters = 5)
    model.fit(np.reshape(image, (40000, 3)))
    
    prediction = model.predict(np.reshape(image, (40000, 3)))
    
    colors = np.random.randint(0, 255, (10, 3))
    
    arr = np.zeros((200,200,3), dtype=int)
    the_popular = np.bincount(prediction).argmax()
    
    prediction = np.reshape(prediction, (200, 200))
    
    for i in range(200):
        for j in range(200):
            cluster = prediction[i][j]
            if cluster == the_popular:
                arr[i][j] = colors[cluster]
    
    return arr