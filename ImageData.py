import numpy as np 
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D, concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping


class ImageData:
    def __init__(self, train_size: float, gen_new_images: bool=False) -> None:

        self.train_size = train_size

        # original data
        self.train = pd.read_json("./data/train.json")
        self.test = pd.read_json("./data/test.json")

        self.missing_vals()

        self.X_train = self.reshape_images(self.train)
        self.X_test = self.reshape_images(self.test)
        self.y_train = np.array(self.train['is_iceberg'])
        self.X_angle_train = np.array(self.train.inc_angle)
        self.X_angle_test = np.array(self.test.inc_angle)


        if gen_new_images:

            self.X_train = self.gen_new_images(self.X_train)
            self.y_train = np.concatenate((self.y_train, self.y_train, self.y_train))
            self.X_angle_train = np.concatenate((self.X_angle_train, self.X_angle_train, 
                                                 self.X_angle_train))

            # specifies image transformations performed during training using generator function
            self.gen = ImageDataGenerator(horizontal_flip = True,
                                      vertical_flip = True,
                                      width_shift_range = 0.1,
                                      height_shift_range = 0.1,
                                      zca_whitening=False,
                                      featurewise_center=False,
                                      featurewise_std_normalization=False)
        else:
            self.gen = ImageDataGenerator(horizontal_flip = False,
                                      vertical_flip = False,
                                      width_shift_range = 0,
                                      height_shift_range = 0,
                                      zca_whitening=False,
                                      featurewise_center=False,
                                      featurewise_std_normalization=False)

        #self.plot_random_images(self.X_train)
        
        self.X_train, self.X_valid, self.X_angle_train, \
        self.X_angle_valid, self.y_train, self.y_valid = self.split()

    def plot_random_images(self, images: np.array) -> None:
        fig = plt.figure(200, figsize=(7, 7))
        random_indicies = np.random.choice(range(len(images)), 9, False)
        subset = images[random_indicies]
        for i in range(9):
            ax = fig.add_subplot(3, 3, i + 1)
            ax.imshow(subset[i], cmap='viridis')
        plt.show()
        
    def split(self) -> np.array:
        return train_test_split(self.X_train, self.X_angle_train, self.y_train,
                                random_state=7, train_size=self.train_size)
    
    def gen_flow_(self, X1, X2, y, batch_size: int):
        # generator function for ImageDataGenerator 

        genX1 = self.gen.flow(X1, y, batch_size=batch_size, seed=42)
        genX2 = self.gen.flow(X1, X2, batch_size=batch_size, seed=42)
        while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i[1]], X1i[1]
    
   
    def missing_vals(self) -> None:
        # missing values for incendence angle substituted with the avg for train and test

        self.train.inc_angle = self.train.inc_angle.replace('na', np.nan)
        avg_train_angle = self.train.inc_angle[self.train.inc_angle != np.nan].mean()
        self.train.inc_angle = self.train.inc_angle.astype(np.float32).fillna(avg_train_angle)
        avg_test_angle = self.test.inc_angle[self.test.inc_angle != np.nan].mean()
        self.test.inc_angle = self.test.inc_angle.astype(np.float32).fillna(avg_test_angle)

    def reshape_images(self, images: pd.DataFrame) -> np.array:
        # images are reshapes for training to 75x75x3 where the third band is the average of original 2
        # each image is scaled with a man/max scaling 
        imgs = []
        
        for i, row in images.iterrows():
            band_1 = np.array(row['band_1']).reshape(75, 75)
            band_2 = np.array(row['band_2']).reshape(75, 75)
            band_3 = (band_1 + band_2) / 2
            
            a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
            b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
            c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

            imgs.append(np.dstack((a, b, c)))
        return np.array(imgs)

    def gen_new_images(self, imgs: np.array) -> np.array:
        # new images are generated from originals using vertical and horizontal flips

        more_images, vert_flip_imgs, hori_flip_imgs = [], [], []
        
        for i in range(0, imgs.shape[0]):
            a, b, c = imgs[i,:,:,0], imgs[i,:,:,1], imgs[i,:,:,2]
            av = cv2.flip(a, 1)
            ah = cv2.flip(a, 0)
            bv = cv2.flip(b, 1)
            bh = cv2.flip(b, 0)
            cv = cv2.flip(c, 1)
            ch = cv2.flip(c, 0)
            
            vert_flip_imgs.append(np.dstack((av, bv, cv)))
            hori_flip_imgs.append(np.dstack((ah, bh, ch)))
        
        v = np.array(vert_flip_imgs)
        h = np.array(hori_flip_imgs)
        more_images = np.concatenate((imgs, v, h))
        
        return more_images

