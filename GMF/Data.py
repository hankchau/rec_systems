import sys
import random
import pandas as pd

"""
Using all positive instances
"""

VAL_RATIO = 0.3
NEG_RATIO = 3

class DataParse:

    def get_train_data(self):
        self.get_negative_sample()

        self.train_data = {'CST_ID': [], 'FND_ID': [], 'Rating': []}
        self.train_data['CST_ID'] = self.pos_train['CST_ID'] + (self.neg_train['CST_ID'])
        self.train_data['FND_ID'] = self.pos_train['FND_ID'] + (self.neg_train['FND_ID'])
        self.train_data['Rating'] = self.pos_train['Rating'] + (self.neg_train['Rating'])

        print('finished generating training samples')

    def get_val_data(self):
        self.val_data = {'CST_ID': [], 'FND_ID': [], 'Rating': []}
        self.val_data['CST_ID'] = self.pos_val['CST_ID'] + (self.neg_val['CST_ID'])
        self.val_data['FND_ID'] = self.pos_val['FND_ID'] + (self.neg_val['FND_ID'])
        self.val_data['Rating'] = self.pos_val['Rating'] + (self.neg_val['Rating'])

        print('finished generating validation samples')

    def read_train_file(self, trainfile):
        print('reading training data ...')

        # customer rating matrix
        df = pd.read_csv(trainfile, sep=',')
        df = df.transpose()

        # positive samples
        self.pos_data = {'CST_ID': [], 'FND_ID': []}
        # negative samples
        self.neg_data = {'CST_ID': [], 'FND_ID': []}
        # index to customer dictionary
        self.train_custs = {}
        self.funds = {}

        fund_indx = -1
        # get data
        for rindex, row in df.iterrows():
            if not fund_indx == -1:
                self.funds[fund_indx] = rindex

            for cindex, f in row.iteritems():
                if fund_indx == -1:
                    self.train_custs[cindex] = f
                    continue
                if f == 1:
                    self.pos_data['CST_ID'].append(cindex)
                    self.pos_data['FND_ID'].append(fund_indx)
                elif f == 0:
                    self.neg_data['CST_ID'].append(cindex)
                    self.neg_data['FND_ID'].append(fund_indx)
            fund_indx += 1

        self.funds_len = len(self.funds)
        self.train_custs_len = len(self.train_custs)
        # get lengths
        self.neg_len = len(self.neg_data['CST_ID'])
        self.pos_len = len(self.pos_data['CST_ID'])
        # self.NEG_SAMPLE_RATIO = int(self.neg_len/self.pos_len)
        self.NEG_SAMPLE_RATIO = NEG_RATIO

    def val_split(self):
        # split pos data
        ratio = (1 - VAL_RATIO)
        split_indx = int(ratio * len(self.pos_data['CST_ID']))
        self.pos_train = {'CST_ID': [], 'FND_ID': [], 'Rating':[]}
        self.pos_train['CST_ID'] = (self.pos_data['CST_ID'])[:split_indx]
        self.pos_train['FND_ID'] = (self.pos_data['FND_ID'])[:split_indx]
        self.pos_train_len = len(self.pos_train['CST_ID'])
        self.pos_train['Rating'] = [1] * self.pos_train_len

        self.pos_val = {'CST_ID': [], 'FND_ID': [], 'Rating':[]}
        self.pos_val['CST_ID'] = (self.pos_data['CST_ID'])[split_indx:]
        self.pos_val['FND_ID'] = (self.pos_data['FND_ID'])[split_indx:]
        self.pos_val_len = len(self.pos_val['CST_ID'])
        self.pos_val['Rating'] = [1] * self.pos_val_len

        # split neg data
        split_indx = int(ratio * len(self.neg_data['CST_ID']))
        self.neg_train_pool = {'CST_ID': [], 'FND_ID': [], 'Rating':[]}
        self.neg_train_pool['CST_ID'] = (self.neg_data['CST_ID'])[:split_indx]
        self.neg_train_pool['FND_ID'] = (self.neg_data['FND_ID'])[:split_indx]
        self.neg_train_pool['Rating'] = [0] * len(self.neg_train_pool['CST_ID'])

        self.neg_val = {'CST_ID': [], 'FND_ID': [], 'Rating':[]}
        self.neg_val['CST_ID'] = (self.neg_data['CST_ID'])[split_indx:]
        self.neg_val['FND_ID'] = (self.neg_data['FND_ID'])[split_indx:]
        self.neg_val['Rating'] = [0] * len(self.neg_val['CST_ID'])

    # get negative instances from sampling
    def get_negative_sample(self):

        print('data: ' + str(self.pos_len + self.neg_len) +
              '\n   positive: '+ str(self.pos_len) +
              '\n   negative: ' + str(self.neg_len) +
              '\n   sparsity: ' + str(self.neg_len/(self.neg_len + self.pos_len)) +
              '\n   negative sample ratio: 1:' + str(self.NEG_SAMPLE_RATIO) +
              '\n   validation ratio: ' + str(VAL_RATIO))

        # negative instance index
        self.pool_len = len(self.neg_train_pool['CST_ID'])
        neg_trainIndx = random.sample(range(0, self.pool_len),
                                      int(self.pos_train_len * self.NEG_SAMPLE_RATIO))
        self.neg_train = {'CST_ID': [], 'FND_ID': []}

        # negative instance samples
        for indx in neg_trainIndx:
            self.neg_train['CST_ID'].append(self.neg_train_pool['CST_ID'][indx])
            self.neg_train['FND_ID'].append(self.neg_train_pool['FND_ID'][indx])

        # get ratings
        self.neg_train_len = len(self.neg_train['CST_ID'])
        self.neg_train['Rating'] = [0] * self.neg_train_len

        print('training sampled: ' + str(self.pos_train_len + self.neg_train_len) +
              '\n   positive: ' + str(self.pos_train_len) +
              '\n   negative: ' + str(self.neg_train_len))


def main(trainfile):
    data = DataParse()
    data.read_train_file(trainfile)
    data.val_split()
    data.get_train_data()
    data.get_val_data()

    print('finished reading data')


if __name__ == '__main__':
    main(sys.argv[1])
