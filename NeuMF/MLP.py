"""
CF Recommender System
By Hank Chau

Model credits to:
Neural Collaborative Filtering (2017)
https://arxiv.org/pdf/1708.05031.pdf

"""
import os
import csv
import sys
from math import inf
import numpy as np
import keras_metrics as km
import matplotlib.pyplot as plt

import Data

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Embedding, Flatten, Concatenate, Dense, Dropout
from keras import Input, regularizers, callbacks


# HYPER-PARAMETERS
THRESHOLD = 0.6
LATENT_DIM = 128
REG_SCALE = 0.000001

BATCH_SIZE = 4096
LEARN_RATE = 0.001

EPOCHS = 1
ITERATIONS = 500
PATIENCE = 20


class NeuralCF:

    def __init__(self):
        print('building Neural CF model ...')
        self.history = {}

    def build_model(self, cust_dim, fund_dim, latent_dim=LATENT_DIM, lr=LEARN_RATE):
        self.build_embedding(cust_dim, fund_dim, latent_dim)
        self.build_mlp_layers()

        print('finished building model')
        metrics = [km.binary_precision(0), km.binary_recall(0)]

        # compile neural CF model
        self.ncf_model.compile(
            optimizer=Adam(lr=lr),
            loss='binary_crossentropy',
            metrics=metrics
        )

        # update history dict
        self.history['precision'] = []
        self.history['recall'] = []
        self.history['val_precision'] = []
        self.history['val_recall'] = []
        self.history['loss'] = []
        self.history['val_loss'] = []

        print('compiling Neural CF model ...')
        print(self.ncf_model.summary())


    def build_embedding(self, cust_dim, fund_dim, latent_dim):
        # customer input embedding layer
        self.MLP_custInputLayer = Input(shape=(1,), dtype='float', name='MLP_cust_Input')
        cust_embedding = Embedding(
            input_dim=cust_dim,
            output_dim=latent_dim,
            embeddings_regularizer=regularizers.l2(REG_SCALE),
            name='MLP_cust_Embedding'
        )

        # hedgefund input embedding layer
        self.MLP_fundInputLayer = Input(shape=(1,), dtype='float', name='MLP_fund_Input')
        fund_embedding = Embedding(
            input_dim=fund_dim,
            output_dim=latent_dim,
            embeddings_regularizer=regularizers.l2(REG_SCALE),
            name='MLP_fund_Embedding'
        )

        # flatten embedding layers
        self.userLatentLayer = Flatten(name='MLP_cust_latent')(cust_embedding(self.MLP_custInputLayer))
        self.fundLatentLayer = Flatten(name='MLP_fund_latent')(fund_embedding(self.MLP_fundInputLayer))


    def build_mlp_layers(self):
        # Neural CF Layers (MLP)
        # mlp input layer
        self.mlpInputLayer = Concatenate(axis=-1, name='MLP_input')([self.userLatentLayer, self.fundLatentLayer])

        # drop layer
        self.dropLayer = Dropout(0.5, name='MLP_Drop')(self.mlpInputLayer)

        # hidden layer 0
        self.hiddenLayer0 = Dense(LATENT_DIM*2, activation='relu', name='MLP_Hidden0')(self.dropLayer)

        # hidden layer 1
        self.hiddenLayer1 = Dense(LATENT_DIM, activation='relu', name='MLP_Hidden1')(self.hiddenLayer0)

        # hidden layer 2
        self.hiddenLayer2 = Dense(int(LATENT_DIM/2), activation='relu', name='MLP_Hidden2')(self.hiddenLayer1)

        # hidden Layer 3
        self.hiddenLayer3 = Dense(int(LATENT_DIM/4), activation='relu', name='MLP_Hidden3')(self.hiddenLayer2)

        # output layer
        self.predictLayer = Dense(1, activation='sigmoid', name='Prediction')(self.hiddenLayer3)

        self.ncf_model = Model(
            inputs=[self.MLP_custInputLayer, self.MLP_fundInputLayer],
            outputs=[self.predictLayer]
        )

    def fit(self, data, outpath, perm_train, perm_val, batch_size=BATCH_SIZE, epochs=EPOCHS):

        cust_train = np.array(data.train_data['CST_ID'])[perm_train]
        fund_train = np.array(data.train_data['FND_ID'])[perm_train]
        pred_train = np.array(data.train_data['Rating'])[perm_train]

        cust_val = np.array(data.val_data['CST_ID'])[perm_val]
        fund_val = np.array(data.val_data['FND_ID'])[perm_val]
        pred_val = np.array(data.val_data['Rating'])[perm_val]

        print('fitting NCF model on train data ...')
        inputs_train = {
            'MLP_cust_Input': cust_train,
            'MLP_fund_Input': fund_train
        }
        inputs_val = {
            'MLP_cust_Input': cust_val,
            'MLP_fund_Input': fund_val
        }

        self.result = self.ncf_model.fit(
            x=inputs_train,
            y=pred_train,
            # class_weight={1: 0.75, 0: 0.25},
            # class_weight='auto',
            validation_data = (inputs_val, pred_val),
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True
            # callbacks=[
            #     callbacks.EarlyStopping(
            #         monitor='val_loss', verbose=2,mode='min', patience=5),
            #     callbacks.ModelCheckpoint(
            #         filepath=(outpath + '/mlp_model_save.hdf5'),
            #         monitor='val_loss', verbose=2,
            #         save_best_only=True, mode='min', period=1)
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
        plt.title('MLP Model Performance')
        plt.ylabel('Percentage')
        plt.legend(['train precision', 'train recall',
                    'val precision', 'val recall'], loc='lower right')

        plt.subplot(2, 1, 2)
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train loss', 'val loss'], loc='upper right')
        plt.savefig(os.getcwd() + '/mlp_history.png')

    def predict(self, data):
        pred_custs = data.neg_data['CST_ID']
        pred_funds = data.neg_data['FND_ID']

        self.rates = self.ncf_model.predict(
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
            if rate >= THRESHOLD:
                    self.predictions[custID].append(fundID)


    def save_predict(self, outpath):
        outpath += '/mlp_predictions_' + str(THRESHOLD) + '.csv'
        with open(outpath, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['CST_ID', 'Fund_ID'])

            for key, value in self.predictions.items():
                line = [key] + value
                writer.writerow(line)


def get_data(datafile):
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

    ncf = NeuralCF()
    ncf.build_model(
        cust_dim=data.train_custs_len,
        fund_dim=data.funds_len,
        latent_dim=LATENT_DIM,
        lr=LEARN_RATE,
    )

    best_loss = inf
    counter = 0
    for i in range(ITERATIONS):
        print('Iteration: '+ str(i))
        perm_train = np.random.permutation(len(data.train_data['CST_ID']))
        perm_val = np.random.permutation(len(data.val_data['CST_ID']))
        ncf.fit(data, batch_size=BATCH_SIZE, epochs=EPOCHS, outpath=outpath, perm_train=perm_train, perm_val=perm_val)

        # Early stopping
        if best_loss >= ncf.history['val_loss'][-1]:
            ncf.ncf_model.save(outpath + '/mlp_model_save.hdf5')
            original_loss = best_loss
            best_loss = ncf.history['val_loss'][-1]
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

    ncf.plot_history()
    ncf.predict(data)
    ncf.save_predict(outpath)

    return ncf

if __name__ == '__main__':
    cd = os.getcwd()
    datafile = cd + '/output.csv'
    outpath = cd + '/results'
    main(outpath, datafile=datafile)
