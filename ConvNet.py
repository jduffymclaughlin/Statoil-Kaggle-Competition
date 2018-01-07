import numpy as np
import pandas as pd
from keras.models import Sequential, model_from_json, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D, concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from ImageData import ImageData


class ConvNet:
    def __init__(self, cross_validating: bool, model_num: int, conv_layers: tuple, dense_layers: tuple, epochs: int,
                 learning_rate: float, dropout: float, patience: int=1000, batch_size: int=32) -> None:
        
        self.cross_validating = cross_validating
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.model_num = model_num
        self.weights_path = './model_weights_' + str(model_num) + '.hdf5'
        self.patience = patience
        self.batch_size = batch_size

        self.callbacks = self.get_callbacks()
        self.model = self.get_model()
    
    def get_model(self) -> Sequential:
        pic_input = Input(shape=(75, 75, 3))
        ang_input = Input(shape=(1,))
        cnn = BatchNormalization()(pic_input)

        for c in self.conv_layers:
            cnn = Conv2D(c, kernel_size = (3,3), activation='relu')(cnn)
            cnn = MaxPooling2D((2,2))(cnn)
            cnn = Dropout(self.dropout)(cnn)

        cnn = GlobalMaxPooling2D()(cnn)
        cnn = concatenate([cnn,ang_input])

        for d in self.dense_layers:
            cnn = Dense(d, activation='relu')(cnn)
            cnn = Dropout(self.dropout)(cnn)

        cnn = Dense(1, activation = 'sigmoid')(cnn)
        model = Model(inputs=[pic_input, ang_input], outputs=cnn)
        opt = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
        return model

    def get_callbacks(self) -> list:
        if self.cross_validating:
            es = EarlyStopping('val_loss', patience=self.patience, mode="min")
            msave = ModelCheckpoint(self.weights_path, monitor='val_loss', save_best_only=True)
        else:
            es = EarlyStopping('loss', patience=self.patience, mode="min")
            msave = ModelCheckpoint(self.weights_path, monitor='loss', save_best_only=True)
        return [es, msave]

    def fit(self, data: ImageData) -> None:
        generator = data.gen_flow_(data.X_train, data.X_angle_train, data.y_train, self.batch_size)
        batch_size = 32

        self.model.fit_generator(generator, validation_data=([data.X_valid, data.X_angle_valid], data.y_valid),
                    steps_per_epoch=len(data.X_train) / self.batch_size, epochs=self.epochs,
                    callbacks=self.callbacks, verbose=1)
        self.save_model()

    def save_model(self) -> None:
        model_json = self.model.to_json()
        with open("./model_" + str(self.model_num) + ".json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model_" + str(self.model_num) + ".h5")

    def evaluate(self, data: ImageData) -> pd.DataFrame:
        self.model.load_weights(filepath=self.weights_path)
        score = self.model.evaluate([data.X_valid, data.X_angle_valid], data.y_valid, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        predicted_test = self.model.predict([data.X_test, data.X_angle_test])
        submission = pd.DataFrame()
        submission['id'] = data.test['id']
        submission['is_iceberg'] = predicted_test.reshape((predicted_test.shape[0]))
        submission.to_csv('sub_' + str(self.model_num) + '.csv', index=False)
        return submission


def main():

    dat1 = ImageData()

    cn1 = ConvNet(model_num=16, conv_layers=(32, 64, 128, 256), dense_layers=(256, 128), 
                 epochs=120, learning_rate=0.0001, patience=175) 
    cn1.model.summary()
    cn1.fit(dat1)
    sub16 = cn1.evaluate(dat1)


    print(sub16)

if __name__ == '__main__':
    main()
