"""
CF Recommender System
By Hank Chau

Model credits to:
Neural Collaborative Filtering (2017)
https://arxiv.org/pdf/1708.05031.pdf

"""
import csv
import sys
import numpy as np
import keras_metrics as km

import Data

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Embedding, Flatten, Concatenate, Dense
from keras import Input, initializers, regularizers, callbacks

LATENT_DIM = 128
LEARN_RATE = 0.0005

BATCH_SIZE = 256
EPOCHS = 10
ITERATIONS = 3

REG_SCALE = 0.01


class NeuralCF:

    def __init__(self):
        print('building Neural CF model ...')


    def build_model(self, cust_dim, fund_dim, latent_dim=LATENT_DIM, lr=LEARN_RATE):
        self.build_embedding(cust_dim, fund_dim, latent_dim)
        self.build_mlp_layers()

        print('finished building model')

        # compile neural CF model
        self.ncf_model.compile(
            optimizer=Adam(lr=lr),
            loss='binary_crossentropy',
            metrics=['acc', km.binary_precision(0), km.binary_recall(0)]
        )
        print('compiling Neural CF model ...')
        print(self.ncf_model.summary())


    def build_embedding(self, cust_dim, fund_dim, latent_dim):
        # customer input embedding layer
        self.MLP_custInputLayer = Input(shape=(1,), dtype='float', name='MLP_cust_Input')
        cust_embedding = Embedding(
            trainable=True,
            input_dim=cust_dim,
            output_dim=latent_dim,
            embeddings_initializer=initializers.RandomNormal(mean=0, stddev=0.01),
            embeddings_regularizer=regularizers.l2(REG_SCALE),
            name='MLP_cust_Embedding'
        )

        # hedgefund input embedding layer
        self.MLP_fundInputLayer = Input(shape=(1,), dtype='float', name='MLP_fund_Input')
        fund_embedding = Embedding(
            trainable=True,
            input_dim=fund_dim,
            output_dim=latent_dim,
            embeddings_initializer=initializers.RandomNormal(mean=0, stddev=0.01),
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

        # hidden layer 1
        self.hiddenLayer1 = Dense(LATENT_DIM*2, activation='relu', name='MLP_Hidden1')(self.mlpInputLayer)

        # hidden layer 2
        self.hiddenLayer2 = Dense(LATENT_DIM, activation='relu', name='MLP_Hidden2')(self.hiddenLayer1)

        # hidden layer 3
        self.hiddenLayer3 = Dense(int(LATENT_DIM/2), activation='relu', name='MLP_Hidden3')(self.hiddenLayer2)

        # output layer
        self.predictLayer = Dense(1, activation='sigmoid', name='Prediction')(self.hiddenLayer3)

        self.ncf_model = Model(
            inputs=[self.MLP_custInputLayer, self.MLP_fundInputLayer],
            outputs=[self.predictLayer]
        )

    def fit(self, data, outpath, perm, batch_size=BATCH_SIZE, epochs=EPOCHS):

        cust_train = np.array(data.train_data['CST_ID'])[perm]
        fund_train = np.array(data.train_data['FND_ID'])[perm]
        pred_train = np.array(data.train_data['Rating'])[perm]

        print('fitting NCF model on train data ...')
        inputs = {
            'cust_Input': cust_train,
            'fund_Input': fund_train
        }

        self.history = self.ncf_model.fit(
            x=inputs,
            y=pred_train,
            validation_split=0.2,
            class_weight={1: 0.75, 0: 0.25},
            # class_weight='auto',
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_loss', mode='min', patience=3),
                callbacks.ModelCheckpoint(
                    filepath=(outpath + '/unweighted_model_save.hdf5'),
                    monitor='val_loss',
                    save_best_only=True, mode='min', period=1)
            ]
        )

        self.ncf_model.save_weights(outpath + '/unweighted_model_weights.hdf5')

        return self.history

    def evaluate(self, test_data, batch_size):
        print('\nevaluating NCF model on test data ...')

        cust_test = np.array(test_data['CST_ID'])
        fund_test = np.array(test_data['FND_ID'])
        pred_test = np.array(test_data['Rating'])

        self.ncf_model.evaluate(
            x=[cust_test, fund_test],
            y=pred_test,
            batch_size=batch_size
        )

    def predict(self, data):
        pred_custs = data.neg_data['CST_ID']
        pred_funds = data.neg_data['FND_ID']

        self.rates = self.ncf_model.predict(
            x=[np.array(pred_custs), np.array(pred_funds)],
            batch_size=256
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
        outpath += '/unweighted_MLP_predictions.csv'
        with open(outpath, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['CST_ID', 'Fund_ID'])

            for key, value in self.predictions.items():
                line = [key] + value
                writer.writerow(line)

def main(datafile, outpath):
    data = Data.DataParse()
    data.read_train_file(datafile)
    data.get_train_data()
    # data.get_test_data(testfile)
    perm = np.random.permutation(len(data.train_data['CST_ID']))

    ncf = NeuralCF()
    ncf.build_model(
        cust_dim=data.train_custs_len,
        fund_dim=data.funds_len,
        latent_dim=LATENT_DIM,
        lr=LEARN_RATE,
    )

    for i in range(ITERATIONS):
        print('Iteration: '+ str(i))
        ncf.fit(data, batch_size=BATCH_SIZE, epochs=EPOCHS, outpath=outpath, perm=perm)
        data.get_train_data()

    # res = ncf.evaluate(data.test_data, BATCH_SIZE)
    # print(res)
    ncf.predict(data)

    ncf.save_predict(outpath)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
