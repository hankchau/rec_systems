import os
import sys
import csv
import numpy as np
import keras_metrics as km
import matplotlib.pyplot as plt

import Data

from math import inf
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Embedding, Flatten, Dot, Multiply, Dense, Dropout
from keras import Input, regularizers, callbacks

# HYPER-PARAMETERS
LATENT_DIM = 64
REG_SCALE = 0.000001

BATCH_SIZE = 4096
LEARN_RATE = 0.001

EPOCHS = 1
ITERATIONS = 500
PATIENCE = 20


class GMF:
    def __init__(self):
        print('pre-training GMF ...')
        self.history = {}

    # default method uses dot product for combining layer
    def build_model(self, cust_dim, fund_dim, latent_dim, lr, dot=False, mul=True):
        print('building GMF model ...')

        # customer input embedding layer
        self.GMF_custInputLayer = Input(shape=(1,), dtype='float', name='GMF_cust_Input')
        cust_embedding = Embedding(
            trainable=True,
            input_dim=cust_dim,
            output_dim=latent_dim,
            embeddings_regularizer=regularizers.l2(REG_SCALE),
            name='GMF_cust_Embedding'
        )
        # fund input embedding layer
        self.GMF_fundInputLayer = Input(shape=(1,), dtype='float', name='GMF_fund_Input')
        fund_embedding = Embedding(
            trainable=True,
            input_dim=fund_dim,
            output_dim=latent_dim,
            embeddings_regularizer=regularizers.l2(REG_SCALE),
            name='GMF_fund_Embedding'
        )
        # Flatten input latent layers
        self.userLatentLayer = Flatten(name='GMF_cust_Latent')(cust_embedding(self.GMF_custInputLayer))
        self.fundLatentLayer = Flatten(name='GMF_fund_Latent')(fund_embedding(self.GMF_fundInputLayer))

        # check which combination to use
        if dot:
            self.dot = Dot(axes=-1, name='Dot_Layer')([self.userLatentLayer, self.fundLatentLayer])
            self.predictions = Dense(1, activation='sigmoid', name='prediction')(self.dot)
        elif mul:
            # element-wise multiplication
            self.mul = Multiply(name='Mul_Layer')([self.userLatentLayer, self.fundLatentLayer])
            self.drop = Dropout(0.5)(self.mul)
            self.predictions = Dense(1, activation='sigmoid', name='prediction')(self.drop)

        # build model
        self.gmf_model = Model(
            inputs=[self.GMF_custInputLayer, self.GMF_fundInputLayer],
            output=self.predictions
        )
        print('GMF Model Architecture: ')
        print(self.gmf_model.summary())

        self.metrics = [km.binary_precision(0), km.binary_recall(0)]

        print('compiling GMF model ...')
        # compile GMF model
        self.gmf_model.compile(
            optimizer=Adam(lr=lr),
            loss='binary_crossentropy',
            metrics=self.metrics
        )

        # update history dict
        self.history['precision'] = []
        self.history['recall'] = []
        self.history['val_precision'] = []
        self.history['val_recall'] = []
        self.history['loss'] = []
        self.history['val_loss'] = []

    def fit(self, data, outpath, perm_train, perm_val, batch_size=BATCH_SIZE, epochs=EPOCHS):
        cust_train = np.array(data.train_data['CST_ID'])[perm_train]
        fund_train = np.array(data.train_data['FND_ID'])[perm_train]
        pred_train = np.array(data.train_data['Rating'])[perm_train]

        cust_val = np.array(data.val_data['CST_ID'])[perm_val]
        fund_val = np.array(data.val_data['FND_ID'])[perm_val]
        pred_val = np.array(data.val_data['Rating'])[perm_val]

        print('fitting GMF model on train data ...')
        inputs_train = {
            'GMF_cust_Input': cust_train,
            'GMF_fund_Input': fund_train
        }

        inputs_val = {
            'GMF_cust_Input': cust_val,
            'GMF_fund_Input': fund_val
        }

        self.result = self.gmf_model.fit(
            x=inputs_train,
            y=pred_train,
            validation_data=(inputs_val, pred_val),
            # class_weight={1: 0.9, 0: 0.1},
            # class_weight='auto',
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True

            # callbacks=[
            # callbacks.EarlyStopping(monitor='val_loss', verbose=2, mode='min', patience=3),
            # callbacks.ModelCheckpoint(
            # filepath=(outpath + '/gmf_model_save.hdf5'),
            # monitor='val_loss', verbose=2,
            # save_best_only=True, mode='min', period=1)
            # ]

        )

        # update history
        self.history['loss'].extend(self.result.history['loss'])
        self.history['val_loss'].extend(self.result.history['val_loss'])
        self.history['precision'].extend(self.result.history['precision'])
        self.history['val_precision'].extend(self.result.history['val_precision'])
        self.history['recall'].extend(self.result.history['recall'])
        self.history['val_recall'].extend(self.result.history['val_recall'])

    def plot_history(self):
        plt.subplot(2, 1, 1)
        plt.plot(self.history['precision'])
        plt.plot(self.history['recall'])
        plt.plot(self.history['val_precision'])
        plt.plot(self.history['val_recall'])
        plt.title('Model Performance')
        plt.ylabel('Percentage')
        plt.legend(['train precision', 'train recall',
                    'val precision', 'val recall'], loc='lower right')
        # plt.show()
        # plt.savefig(os.getcwd() + '/gmf_acc_history.png')

        plt.subplot(2, 1, 2)
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        # plt.title('Model Loss (Binary Crossentropy)')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train loss', 'val loss'], loc='upper right')
        # fig2.show()
        plt.savefig(os.getcwd() + '/gmf_history.png')

    def predict(self, data):
        pred_custs = data.neg_data['CST_ID']
        pred_funds = data.neg_data['FND_ID']

        self.rates = self.gmf_model.predict(
            x=[np.array(pred_custs), np.array(pred_funds)],
            batch_size=4096
        )

        self.predictions = {}
        for indx in range(len(pred_custs)):
            custID = data.train_custs[pred_custs[indx]]
            fundID = data.funds[pred_funds[indx]]
            if custID not in self.predictions:
                self.predictions[custID] = []

            rate = self.rates[indx]
            # condition
            if rate >= 0.5:
                self.predictions[custID].append(fundID)

    def save_predict(self, outpath):
        outpath += '/gmf_predictions.csv'
        with open(outpath, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['CST_ID', 'Fund_ID'])

            for key, value in self.predictions.items():
                line = [key] + value
                writer.writerow(line)


def get_data(datafile):
    # get data
    data = Data.DataParse()
    data.read_train_file(datafile)
    data.val_split()
    data.get_negative_sample()
    data.get_train_data()
    data.get_val_data()

    return data

def main(outpath, datafile=None, data=None):
    if datafile:
        data = get_data(datafile)

    gmf = GMF()
    gmf.build_model(
        cust_dim=data.train_custs_len,
        fund_dim=data.funds_len,
        latent_dim=LATENT_DIM,
        lr=LEARN_RATE,
        mul=True
    )

    best_loss = inf
    counter = 0
    for i in range(ITERATIONS):
        print('Iteration: ' + str(i))
        perm_train = np.random.permutation(len(data.train_data['CST_ID']))
        perm_val = np.random.permutation(len(data.val_data['CST_ID']))
        gmf.fit(data, outpath, perm_train=perm_train, perm_val=perm_val, batch_size=BATCH_SIZE, epochs=EPOCHS)

        # Early stopping
        if best_loss >= gmf.history['val_loss'][-1]:
            gmf.gmf_model.save(outpath + '/gmf_model_save.hdf5')
            original_loss = best_loss
            best_loss = gmf.history['val_loss'][-1]
            counter = 0
            print('best val_loss improved from ' + str(original_loss) +
                  ' to ' + str(best_loss) + '\nsaving model ...')
        else:
            counter += 1
            if counter >= PATIENCE:
                print('val_loss did not improve after ' + str(counter) + ' iterations')
                print('earlystopping at iteration: ' + str(i))
                break
        # resample negative instances
        data.get_train_data()

    gmf.plot_history()
    gmf.predict(data)
    gmf.save_predict(outpath)

    return gmf

if __name__ == '__main__':
    cd = os.getcwd()
    datafile = cd + '/output.csv'
    outpath = cd + '/results'
    main(outpath, datafile=datafile)
