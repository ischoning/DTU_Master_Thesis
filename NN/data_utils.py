from Constants import *
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Linear, LSTM, GRU, Conv1d, Conv2d, Dropout, MaxPool2d, BatchNorm1d, BatchNorm2d, CrossEntropyLoss, BCELoss
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from sklearn.preprocessing import LabelEncoder
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    #from sklearn.cross_validation import StratifiedShuffleSplit
    # cross_validation -> now called: model_selection
    # https://stackoverflow.com/questions/30667525/importerror-no-module-named-sklearn-cross-validation
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[int(row), int(col)] = 1
    return out


class load_data():
    # input are pandas dataframes
    def __init__(self, x_train, y_train, x_test, y_test):
        self.image_shape = IMAGE_SHAPE[:2]
        self._load(x_train, y_train, x_test, y_test)
        
    def _load(self, x_train, y_train, x_test, y_test):
        #print("Loading training data")
        train_data = self._make_dataset(x_train, y_train)
        #print("Loading test data")
        test_data = self._make_dataset(x_test, y_test)        
        # need to reformat the train for validation split reasons in the batch_generator
        self.train = self._format_dataset(train_data)
        self.test = self._format_dataset(test_data)
        
        
    def _make_dataset(self, df, y=None):
        seq_length=WINDOW
        # make dataset
        data = dict()

        for i, dat in enumerate(df.iterrows()):
            index, row = dat
            sample = dict()
            features = row.values
            #for j in range(len(FEATS)):
            #    sample[FEATS[j]] = features[WINDOW*j:WINDOW*(j+1)]
            image = np.reshape(features, self.image_shape)
            image = np.expand_dims(image, axis=2)
            sample['img'] = image
            sample['t'] = np.asarray(y.loc[index], dtype='int32')
            data[index] = sample
            #if i % 200 == 0:
                #print("\t%d of %d" % (i, len(df)))
        
        return data

    def _format_dataset(self, d):
        # making arrays with all data in, is nessesary when doing validation split
        data = dict()
        value = list(d.values())[0]
        img_tot_shp = tuple([len(d)] + list(value['img'].shape))
        data['img'] = np.zeros(img_tot_shp, dtype='float32')
        #feature_tot_shp = (len(d), WINDOW)
        #for feat in FEATS:
        #    data[feat] = np.zeros(feature_tot_shp, dtype='float32')
        data['ts'] = np.zeros((len(d),), dtype='int32')
        data['ids'] = np.zeros((len(d),), dtype='int32')
        for i, pair in enumerate(d.items()):
            key, value = pair
            #for feat in FEATS:
            #    data[feat][i] = value[feat]
            data['img'][i] = value['img']
            data['ts'][i] = value['t']
            data['ids'][i] = key
        
        return data

    
class batch_generator():
    def __init__(self, data, batch_size=BATCH_SIZE, num_classes=NUM_CLASSES,
                 num_iterations=5e3, num_features=WINDOW, seed=42, val_size=0.1):
        self._train = data.train
        self._test = data.test
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._num_iterations = num_iterations
        self._num_features = num_features
        # get image size
        value = self._train['img'][0]
        self._image_shape = list(value.shape)
        self._seed = seed
        self._val_size = val_size
        self._valid_split()
        
    def _valid_split(self):
        # Updated to use: model_selection
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self._val_size,
            random_state=self._seed
        ).split(
            np.zeros(self._train['ts'].shape),
            self._train['ts']
        )
        self._idcs_train, self._idcs_valid = next(iter(sss))
        
    def _shuffle_train(self):
        np.random.shuffle(self._idcs_train)

    def _batch_init(self, purpose):
        assert purpose in ['train', 'valid', 'test']
        batch_holder = dict()
        #for feat in FEATS:
        #    batch_holder[feat] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        batch_holder['img'] = np.zeros(tuple([self._batch_size] + list(self._image_shape)), dtype='float32')
        batch_holder['ts'] = np.zeros((self._batch_size, self._num_classes), dtype='float32')          
        batch_holder['ids'] = []
        return batch_holder

    def gen_valid(self):
        batch = self._batch_init(purpose='valid')
        i = 0
        for idx in self._idcs_valid:
            #for feat in FEATS:
            #    batch[feat][i] = self._train[feat][idx]
            batch['img'][i] = self._train['img'][idx]
            batch['ts'][i] = onehot(np.asarray([self._train['ts'][idx]], dtype='float32'), self._num_classes)
            i += 1
            if i >= self._batch_size:
                yield batch, i
                batch = self._batch_init(purpose='valid')
                i = 0
        if i != 0:
            batch['ts'] = onehot(np.asarray([self._train['ts'][idx]], dtype='float32'), self._num_classes)
            #for feat in FEATS:
            #    batch[feat] = batch[feat][:i]
            batch['img'] = batch['img'][:i]
            yield batch, i

    def gen_test(self):
        batch = self._batch_init(purpose='test')
        i = 0
        for idx in range(len(self._test['ids'])):
            #for feat in FEATS:
            #    batch[feat][i] = self._test[feat][idx]
            batch['img'][i] = self._test['img'][idx]
            batch['ts'][i] = onehot(np.asarray([self._test['ts'][idx]], dtype='float32'), self._num_classes)
            batch['ids'].append(self._test['ids'][idx])
            i += 1
            if i >= self._batch_size:
                yield batch, i
                batch = self._batch_init(purpose='test')
                i = 0
        if i != 0:
            yield batch, i     

    def gen_train(self):
        batch = self._batch_init(purpose='train')
        iteration = 0
        i = 0
        while iteration < self._num_iterations:
            # shuffling all batches
            self._shuffle_train()
            for idx in self._idcs_train:
                # extract data from dict
                #for feat in FEATS:
                #    batch[feat][i] = self._train[feat][idx]
                batch['img'][i] = self._train['img'][idx]
                batch['ts'][i] = onehot(np.asarray([self._train['ts'][idx]], dtype='float32'), self._num_classes)
                i += 1
                if i >= self._batch_size:
                    yield batch
                    batch = self._batch_init(purpose='train')
                    i = 0
                    iteration += 1
#                     if iteration >= self._num_iterations:
#                         break
                    