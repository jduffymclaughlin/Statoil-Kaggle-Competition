import numpy as np 
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D, concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping


class ImageData:
    def __init__(self, test_size: float, gen_new_images: bool=False) -> None:
        self.test_size = test_size

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

        self.X_train, self.X_valid, self.X_angle_train, \
        self.X_angle_valid, self.y_train, self.y_valid = self.split()

        self.gen = ImageDataGenerator(horizontal_flip = True,
                                      vertical_flip = True,
                                      width_shift_range = 0.1,
                                      height_shift_range = 0.1,
                                      zca_whitening=False,
                                      featurewise_center=False,
                                      featurewise_std_normalization=False)

    def split(self) -> np.array:
        return train_test_split(self.X_train, self.X_angle_train, self.y_train,
                                random_state=3, train_size=self.test_size)
    
    def gen_flow_(self, X1, X2, y, batch_size: int):
        genX1 = self.gen.flow(X1, y, batch_size=batch_size, seed=42)
        genX2 = self.gen.flow(X1,X2, batch_size=batch_size, seed=42)
        while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i[1]], X1i[1]
    
   
    def missing_vals(self) -> None:
        self.train.inc_angle = self.train.inc_angle.replace('na', np.nan)
        avg_train_angle = self.train.inc_angle[self.train.inc_angle != np.nan].mean()
        self.train.inc_angle = self.train.inc_angle.astype(np.float32).fillna(avg_train_angle)
        avg_test_angle = self.test.inc_angle[self.test.inc_angle != np.nan].mean()
        self.test.inc_angle = self.test.inc_angle.astype(np.float32).fillna(avg_test_angle)
        #weights = np.fromiter((angle / total for angle in train.inc_angle), np.float32)

    def reshape_images(self, images: pd.DataFrame) -> np.array:
        band_1 = np.array([np.array(band).astype(np.float32)
                          .reshape(75, 75) for band in images["band_1"]])
        band_2 = np.array([np.array(band).astype(np.float32)
                          .reshape(75, 75) for band in images["band_2"]])
        return np.concatenate([band_1[:, :, :, np.newaxis], band_2[:, :, :, np.newaxis],
                              ((band_1 + band_2) / 2)[:, :, :, np.newaxis]], axis=-1)

    def gen_new_images(self, imgs: np.array) -> np.array:
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


def main():

    ig = ImageData(.75, True)

if __name__ == '__main__':
    main()
