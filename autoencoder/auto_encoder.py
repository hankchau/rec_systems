import os
import csv
import pandas as pd
import numpy as np

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense
from keras import Input, regularizers, callbacks


VAL_SPLIT = 0.2
# USER_DIM =
# ITEM_DIM =
FEATURE_DIM = 4180
LATENT_DIM = 256
LEARN_RATE = 0.001

EPOCHS = 50
BATCH_SIZE = 2048
PATIENCE = 5

class AutoEncoder:

    def build_model(self):
        # Input user
        self.user_input = Input(shape=(FEATURE_DIM,), name='User_Input')

        # Encoding
        self.encoding1 = Dense(1024, kernel_initializer='lecun_normal', name='Encoding1')(self.user_input)

        # Embedding
        self.embedding = Dense(LATENT_DIM, name='Embedding')(self.encoding1)

        inputs = [self.user_input]
        self.encoder = Model(inputs=inputs, outputs=self.embedding, name='Encoder')
        print(self.encoder.summary())

        # Decoder
        self.decoder_input = Input(shape=(LATENT_DIM,), name='Decoder_Input')
        self.decoding1 = Dense(1024, kernel_initializer='lecun_normal', name='Decoding1')(self.decoder_input)
        self.output = Dense(FEATURE_DIM, activation='sigmoid', name='Output')(self.decoding1)

        self.decoder = Model(inputs=self.decoder_input, outputs=self.output, name='Decoder')
        print(self.decoder.summary())

        # autoencoder
        self.ae = Model(inputs=inputs, outputs=self.decoder(self.encoder(inputs)),
                                 name='AutoEncoder')
        print(self.ae.summary())

        # compile model
        self.ae.compile(
            loss='binary_crossentropy',
            optimizer=Adam(lr=LEARN_RATE),
            metrics=['acc', 'binary_crossentropy']
        )

    def fit(self, data, outpath):
        self.ae.fit(
            x=[data.train_data],
            y=data.train_data,
            validation_data=(data.val_data, data.val_data),
            shuffle=True,
            verbose=2,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                # callbacks.EarlyStopping(
                    # monitor='val_loss', verbose=2, mode='min', patience=PATIENCE),
                callbacks.ModelCheckpoint(
                    filepath=(outpath + '/ae_model_save.hdf5'),
                    monitor='val_loss', verbose=2,
                    save_best_only=True, mode='min', period=1)
            ]
        )


class DataParse:

    def read_data(self, datafile):
        print('reading data from ' + datafile)
        df = pd.read_csv(datafile, sep=',')

        self.data = {}
        self.users = []
        for rindex, row in df.iterrows():
            user = str(int(row[0]))
            self.users.append(user)
            self.data[user] = row[1:]

    def val_split(self):
        print('splitting data ...')
        # find split index
        split_indx =  int((1-VAL_SPLIT) * len(self.data))
        self.train_users = self.users[:split_indx]
        self.val_users = self.users[split_indx:]

        # find train and val data
        self.train_data = []
        for user in self.train_users:
            self.train_data.append(self.data[user])

        self.val_data = []
        for user in self.val_users:
            self.val_data.append(self.data[user])

        # convert to np arrays
        self.train_data = np.array(self.train_data)
        self.val_data = np.array(self.val_data)

        print('total     data: ' + str(len(self.users)) +
              'training   set: ' + str(self.train_data.shape[0]) +
              '\nvalidation set: ' + str(self.val_data.shape[0]))


def main(datafile, outpath):
    dp = DataParse()
    dp.read_data(datafile)
    dp.val_split()

    ae = AutoEncoder()
    ae.build_model()
    ae.fit(dp, outpath)


if __name__ == '__main__':
    datafile = os.getcwd() + '/output.csv'
    outpath = os.getcwd() + '/autoencoder'
    main(datafile, outpath)

