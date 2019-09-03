import sys
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import keras_metrics as km

import Data
import GMF
import MLP

from math import inf
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Concatenate, Dense

# HYPER-PARAMETERS
ALPHA = 0.5 # weighting for MLP
THRESHOLD = 0.6

BATCH_SIZE = 4096
LEARN_RATE = 0.0005

EPOCHS = 1
ITERATIONS = 500
PATIENCE = 20


class Recommender:

    def __init__(self):
        print('preparing Neural MF model ...')
        self.history = {}

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

        print(self.NeuMF.summary())

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

    def compile_model(self):
        metrics = [km.binary_precision(0), km.binary_recall(0)]

        self.NeuMF.compile(
            optimizer=SGD(lr=LEARN_RATE),
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

        print('compiling NeuMF Model ...')

    def fit(self, data, outpath, perm_train, perm_val, batch_size=BATCH_SIZE, epochs=EPOCHS):
        cust_train = np.array(data.train_data['CST_ID'])[perm_train]
        fund_train = np.array(data.train_data['FND_ID'])[perm_train]
        pred_train = np.array(data.train_data['Rating'])[perm_train]

        cust_val = np.array(data.val_data['CST_ID'])[perm_val]
        fund_val = np.array(data.val_data['FND_ID'])[perm_val]
        pred_val = np.array(data.val_data['Rating'])[perm_val]

        print('fitting NeuMF model on train data ...')
        inputs_train = {
            'GMF_cust_Input': cust_train,
            'GMF_fund_Input': fund_train,
            'MLP_cust_Input': cust_train,
            'MLP_fund_Input': fund_train
        }

        inputs_val = {
            'GMF_cust_Input': cust_val,
            'GMF_fund_Input': fund_val,
            'MLP_cust_Input': cust_val,
            'MLP_fund_Input': fund_val
        }

        self.result = self.NeuMF.fit(
            x=inputs_train,
            y=pred_train,
            # class_weight={1: 0.75, 0: 0.25},
            validation_data=(inputs_val, pred_val),
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True

            # callbacks=[
            #     callbacks.EarlyStopping(
            #         monitor='val_loss', verbose=2, mode='min', patience=5),
            #     callbacks.ModelCheckpoint(
            #         filepath=(outpath + '/NeuMF_model_save.hdf5'),
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

    def plot_history(self, outpath):
        plt.subplot(2, 1, 1)
        plt.plot(self.history['precision'])
        plt.plot(self.history['recall'])
        plt.plot(self.history['val_precision'])
        plt.plot(self.history['val_recall'])
        plt.title('NeuMF Model Performance')
        plt.ylabel('Percentage')
        plt.legend(['train precision', 'train recall',
                    'val precision', 'val recall'], loc='lower right')

        plt.subplot(2, 1, 2)
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train loss', 'val loss'], loc='upper right')
        plt.savefig(outpath + '/NeuMF_history.png')
        plt.close()

    def predict(self, data):
        pred_custs = data.neg_data['CST_ID']
        pred_funds = data.neg_data['FND_ID']

        self.rates = self.NeuMF.predict(
            x=[np.array(pred_custs), np.array(pred_funds),
               np.array(pred_custs), np.array(pred_funds)],
            batch_size=4096
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
    def __init__(self):
        print('preparing for pre-train ...')
        print('using default hyperparameters ...')

    def load_weights(self, mlp_file, gmf_file):
        self.ncf.ncf_model.load_weights(mlp_file)
        self.gmf.gmf_model.load_weights(gmf_file)

    # pretrain entire model
    def get_pretrain(self, outpath, data):
        self.perm_train = np.random.permutation(len(data.train_data['CST_ID']))
        self.perm_val = np.random.permutation(len(data.val_data['CST_ID']))
        self.train_MLP(outpath, data)
        self.train_GMF(outpath, data)

    def train_MLP(self, outpath, data):
        print('pretraining GMF model')
        self.ncf = MLP.main(outpath, data=data)

    def train_GMF(self, outpath, data):
        print('pretraining MLP model ...')
        self.gmf = GMF.main(outpath, data=data)


def main(outpath, datafile, gmf_weights=None, mlp_weights=None, NeuMF_weights=None):
    data = Data.DataParse()
    data.read_train_file(datafile)
    data.val_split()
    data.get_negative_sample()
    data.get_train_data()
    data.get_val_data()

    pt = PreTrainer()

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
            pt.get_pretrain(outpath, data)
        # build NeuMF model
        rc.build_model(pretrain=pt, set_weights=True)

    # compile model
    rc.compile_model()

    best_loss = inf
    counter = 0
    for i in range(ITERATIONS):
        print('Iteration: ' + str(i))
        perm_train = np.random.permutation(len(data.train_data['CST_ID']))
        perm_val = np.random.permutation(len(data.val_data['CST_ID']))
        rc.fit(data, outpath, perm_train=perm_train, perm_val=perm_val, batch_size=BATCH_SIZE, epochs=EPOCHS)

        # Early stopping
        if best_loss >= rc.history['val_loss'][-1]:
            rc.NeuMF.save(outpath + '/NeuMF_model_save.hdf5')
            original_loss = best_loss
            best_loss = rc.history['val_loss'][-1]
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

    rc.plot_history(outpath)
    rc.predict(data)
    rc.save_pred_table(outpath)
    rc.save_predict(outpath)


if __name__ == '__main__':
    cd = os.getcwd()
    datafile = cd + '/output.csv'
    outpath = cd + '/results'
    # gmf_weights = cd + '/results/gmf_model_save.hdf5'
    # mlp_weights = cd + '/results/mlp_model_save.hdf5'
    NeuMF_weights = cd + '/results/NeuMF_model_save.hdf5'

    main(outpath, datafile=datafile, NeuMF_weights=NeuMF_weights)
