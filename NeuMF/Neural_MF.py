import sys
import csv
import os
import numpy as np
import keras_metrics as km

import Data
import GMF
import Neural_CF as MLP

from keras import callbacks
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Concatenate, Dense


# HYPER-PARAMETERS
ALPHA = 0.5 # weighting for MLP
THRESHOLD = 0.6

BATCH_SIZE = 256
LEARN_RATE = 0.1

EPOCHS = 20
ITERATIONS = 5


class Recommender:

    def __init__(self):
        print('preparing Neural MF model ...')

    def build_model(self, pretrain, set_weights=False):

        # setting up gmf / mlp
        gmf = pretrain.gmf
        mlp = pretrain.ncf

        # build model concat
        self.NeuMFLayer = Concatenate(axis=-1, name='NeuMF_Layer')([gmf.mul, mlp.hiddenLayer3])
        self.predictLayer = Dense(1, activation='sigmoid', name='Prediction')(self.NeuMFLayer)

        # build
        self.NeuMF = Model(
            inputs=[gmf.GMF_custInputLayer, gmf.GMF_fundInputLayer,
                    mlp.MLP_custInputLayer, mlp.MLP_fundInputLayer],
            outputs=[self.predictLayer]
        )

        if set_weights:
            self.set_initial(pretrain)

        gmf.gmf_model.layers.pop()
        mlp.ncf_model.layers.pop()


    def set_initial(self, pretrain):
        print()
        # setting up gmf
        gmf = pretrain.gmf
        gmf_weights = gmf.gmf_model.layers[-1].get_weights()
        gmf_weights = [w * (1 - ALPHA) for w in gmf_weights]
        gmf.gmf_model.layers.pop()

        # setting up mlp
        mlp = pretrain.ncf
        mlp_weights = mlp.ncf_model.layers[-1].get_weights()
        mlp_weights = [w * ALPHA for w in mlp_weights]

        predict_weights = np.concatenate((gmf_weights[0], mlp_weights[0]), axis=0)
        predict_biases = gmf_weights[1] + gmf_weights[1]

        self.NeuMF.get_layer('Prediction').set_weights([predict_weights, predict_biases])


    def fit(self, data, outpath, perm, batch_size=BATCH_SIZE, epochs=EPOCHS):
        cust_train = np.array(data.train_data['CST_ID'])[perm]
        fund_train = np.array(data.train_data['FND_ID'])[perm]
        pred_train = np.array(data.train_data['Rating'])[perm]

        print('fitting NeuMF model on train data ...')
        inputs = {
            'GMF_cust_Input': cust_train,
            'GMF_fund_Input': fund_train,
            'MLP_cust_Input': cust_train,
            'MLP_fund_Input': fund_train
        }

        self.history = self.NeuMF.fit(
            x=inputs,
            y=pred_train,
            validation_split=0.2,
            class_weight={1: 0.75, 0: 0.25},
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_loss', mode='min', patience=3),
                callbacks.ModelCheckpoint(
                    filepath=(outpath + '/NeuMF_model_save.hdf5'),
                    monitor='val_loss',
                    save_best_only=True, mode='min', period=1)
            ]
        )

        return self.history

    def predict(self, data):
        pred_custs = data.neg_data['CST_ID']
        pred_funds = data.neg_data['FND_ID']

        self.rates = self.NeuMF.predict(
            x=[np.array(pred_custs), np.array(pred_funds),
               np.array(pred_custs), np.array(pred_funds)],
            batch_size=256
        )

        self.rating_table = []
        self.predictions = {}
        for indx in range(len(data.neg_data['CST_ID'])):
            custID = data.train_custs[pred_custs[indx]]
            fundID = data.funds[pred_funds[indx]]
            if custID not in self.predictions:
                self.predictions[custID] = []

            rate = self.rates[indx][0]

            # update rating table
            row = [custID, fundID, str(int(rate*10000)/100)+'%']
            self.rating_table.append(row)

            # condition
            if rate >= THRESHOLD:
                # update predictions
                self.predictions[custID].append(fundID)

    def save_predict(self, outpath):
        outpath += '/NeuMF_predictions_' + str(THRESHOLD) + '.csv'
        with open(outpath, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['CST_ID', 'Fund_ID'])

            for key, value in self.predictions.items():
                line = [key] + value
                writer.writerow(line)

    def save_pred_table(self, outpath):
        outpath = outpath + '/NeuMF_ratings_table_' + str(THRESHOLD) + '.csv'
        with open(outpath, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['CST_ID', 'Fund_ID', 'Rating'])

            for row in self.rating_table:
                writer.writerow(row)

    def load_NeuMF(self, savefile):
        print('loading NeuMF model weights from ' + savefile)
        self.NeuMF.load_weights(savefile)


class PreTrainer:
    def __init__(self, data):
        print('preparing for pre-train ...')
        print('using default hyperparameters ...')
        self.build_MLP(data)
        self.build_GMF(data)

    # pretrain entire model
    def get_pretrain(self, data, outpath):
        self.perm = np.random.permutation(len(data.train_data['CST_ID']))
        self.train_MLP(data, outpath)
        self.train_GMF(data, outpath)


    def load_weights(self, mlp_file, gmf_file):
        self.ncf.ncf_model.load_weights(mlp_file)
        self.gmf.gmf_model.load_weights(gmf_file)

    def build_MLP(self, data):
        print('pretraining MLP model ...')
        self.ncf = MLP.NeuralCF()

        self.ncf.build_model(
            cust_dim=data.train_custs_len,
            fund_dim=data.funds_len,
            latent_dim=MLP.LATENT_DIM,
            lr=MLP.LEARN_RATE
        )

    def build_GMF(self, data):
        print('pretraining GMF model')
        self.gmf = GMF.GMF()

        self.gmf.build_model(
            cust_dim=data.train_custs_len,
            fund_dim=data.funds_len,
            latent_dim=GMF.LATENT_DIM,
            lr=GMF.LEARN_RATE,
            mul=True
        )

    def train_MLP(self, data, outpath, iteration=MLP.ITERATIONS):
        for i in range(iteration):
            self.ncf.fit(data=data, outpath=outpath, perm=self.perm)
            data.get_train_data()
        self.ncf.predict(data)
        self.ncf.save_predict(outpath)

    def train_GMF(self, data, outpath, iteration=GMF.ITERATIONS):
        for i in range(iteration):
            self.gmf.fit(data=data, outpath=outpath, perm=self.perm)
            data.get_train_data()
        self.gmf.predict(data)
        self.gmf.save_predict(outpath)


def main(datafile, outpath, gmf_weights=None, mlp_weights=None, NeuMF_weights=None):
    data = Data.DataParse()
    data.read_train_file(datafile)
    data.get_train_data()

    pt = PreTrainer(data)

    rc = Recommender()

    # if NeuMF weights provided
    if NeuMF_weights:
        print('loading weights for NeuMF Model ...')
        rc.build_model(pretrain=pt)
        rc.load_NeuMF(NeuMF_weights)
    else:
        # load pretrained model weights
        if gmf_weights and mlp_weights:
            print('loading weights for GMF and MLP models ...')
            pt.load_weights(mlp_file=mlp_weights, gmf_file=gmf_weights)
        # pretrain from scratch
        else:
            print('pretraining from scratch ...')
            pt.get_pretrain(data, outpath)

        rc.build_model(pretrain=pt, set_weights=True)

    rc.NeuMF.compile(
        optimizer=SGD(lr=LEARN_RATE),
        loss='binary_crossentropy',
        metrics=['acc', km.binary_precision(0), km.binary_recall(0)]
    )
    print('compiling NeuMF Model ...')
    print(rc.NeuMF.summary())

    for i in range(ITERATIONS):
        print('Iteration: ' + str(i))
        perm = np.random.permutation(len(data.train_data['CST_ID']))

        rc.fit(data, outpath, perm=perm, batch_size=BATCH_SIZE, epochs=EPOCHS)
        data.get_train_data()

    rc.predict(data)
    rc.save_pred_table(outpath)
    rc.save_predict(outpath)


if __name__ == '__main__':
    cd = os.getcwd()
    datafile = cd + '/cst_fund_chart.csv'
    outpath = cd
    # gmf_weights = cd + '/gmf_model_save.hdf5'
    # mlp_weights = cd + '/mlp_model_save.hdf5'

    main(datafile, outpath)
