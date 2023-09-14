# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
SNLI - Entailment
'''
from __future__ import absolute_import, division, unicode_literals

import codecs
import os
import io
import copy
import logging
import numpy as np
import pickle

from senteval.tools.validation import SplitClassifier


class SNLIEval(object):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : SNLI Entailment*****\n\n')
        self.task_name = os.path.basename(taskpath)
        self.seed = seed
        train1 = self.loadFile(os.path.join(taskpath, 's1.train'))
        train2 = self.loadFile(os.path.join(taskpath, 's2.train'))

        trainlabels = io.open(os.path.join(taskpath, 'labels.train'),
                              encoding='utf-8').read().splitlines()

        valid1 = self.loadFile(os.path.join(taskpath, 's1.dev'))
        valid2 = self.loadFile(os.path.join(taskpath, 's2.dev'))
        validlabels = io.open(os.path.join(taskpath, 'labels.dev'),
                              encoding='utf-8').read().splitlines()

        test1 = self.loadFile(os.path.join(taskpath, 's1.test'))
        test2 = self.loadFile(os.path.join(taskpath, 's2.test'))
        testlabels = io.open(os.path.join(taskpath, 'labels.test'),
                             encoding='utf-8').read().splitlines()
        testindexes = list(range(len(testlabels)))

        # sort data (by s2 first) to reduce padding
        sorted_train = sorted(zip(train2, train1, trainlabels),
                              key=lambda z: (len(z[0]), len(z[1]), z[2]))
        train2, train1, trainlabels = map(list, zip(*sorted_train))

        sorted_valid = sorted(zip(valid2, valid1, validlabels),
                              key=lambda z: (len(z[0]), len(z[1]), z[2]))
        valid2, valid1, validlabels = map(list, zip(*sorted_valid))

        sorted_test = sorted(zip(test2, test1, testlabels, testindexes),
                             key=lambda z: (len(z[0]), len(z[1]), z[2], z[3]))
        test2, test1, testlabels, testindexes = map(list, zip(*sorted_test))

        self.samples = train1 + train2 + valid1 + valid2 + test1 + test2
        self.data = {'train': (train1, train2, trainlabels, None),
                     'valid': (valid1, valid2, validlabels, None),
                     'test': (test1, test2, testlabels, testindexes)
                     }

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath):
        with codecs.open(fpath, 'rb', 'latin-1') as f:
            return [line.split() for line in
                    f.read().splitlines()]

    def run(self, params, batcher):
        self.X, self.y, self.index = {}, {}, {}
        dico_label = {'entailment': 0,  'neutral': 1, 'contradiction': 2}

        if params.save_emb is not None:
            data_filename = '_'.join(params.save_emb.split('_')[:-1]) + '_' + self.task_name + '.pkl'
            if os.path.isfile(data_filename):
                logging.info('Loading sentence embeddings')
                # loaded_data = np.load(data_filename)
                with open(data_filename) as f:
                    loaded_data = pickle.load(f)
                self.X, self.y, self.index = loaded_data['X'], loaded_data['y'], loaded_data['index']
                logging.info('Generated sentence embeddings')
            else:
                for key in self.data:
                    logging.info('Computing embedding for {0}'.format(key))
                    if key not in self.X:
                        self.X[key] = []
                    if key not in self.y:
                        self.y[key] = []

                    input1, input2, mylabels, myindexes = self.data[key]
                    enc_input = []
                    n_labels = len(mylabels)
                    for ii in range(0, n_labels, params.batch_size):
                        batch1 = input1[ii:ii + params.batch_size]
                        batch2 = input2[ii:ii + params.batch_size]

                        if len(batch1) == len(batch2) and len(batch1) > 0:
                            enc1 = batcher(params, batch1)
                            enc2 = batcher(params, batch2)
                            enc_input.append(np.hstack((enc1, enc2, enc1 * enc2,
                                                        np.abs(enc1 - enc2))))
                        if (ii*params.batch_size) % (20000*params.batch_size) == 0:
                            logging.info("PROGRESS (encoding): %.2f%%" %
                                         (100 * ii / n_labels))
                    self.X[key] = np.vstack(enc_input)
                    self.y[key] = np.array([dico_label[y] for y in mylabels])
                    self.index[key] = np.array(myindexes)
                    logging.info('Computed {0} embeddings'.format(key))
                logging.info('Saving sentence embeddings')
                # np.savez(data_filename, X=self.X, y=self.y, index=self.index)
                with open(data_filename, 'wb') as f:
                    pickle.dump({'X':self.X, 'y':self.y, 'index':self.index}, f, protocol=4)
        else:
            for key in self.data:
                logging.info('Computing embedding for {0}'.format(key))
                if key not in self.X:
                    self.X[key] = []
                if key not in self.y:
                    self.y[key] = []

                input1, input2, mylabels, myindexes = self.data[key]
                enc_input = []
                n_labels = len(mylabels)
                for ii in range(0, n_labels, params.batch_size):
                    batch1 = input1[ii:ii + params.batch_size]
                    batch2 = input2[ii:ii + params.batch_size]

                    if len(batch1) == len(batch2) and len(batch1) > 0:
                        enc1 = batcher(params, batch1)
                        enc2 = batcher(params, batch2)
                        enc_input.append(np.hstack((enc1, enc2, enc1 * enc2,
                                                    np.abs(enc1 - enc2))))
                    if (ii*params.batch_size) % (20000*params.batch_size) == 0:
                        logging.info("PROGRESS (encoding): %.2f%%" %
                                     (100 * ii / n_labels))
                self.X[key] = np.vstack(enc_input)
                self.y[key] = np.array([dico_label[y] for y in mylabels])
                self.index[key] = np.array(myindexes)
                logging.info('Computed {0} embeddings'.format(key))

        config = {'nclasses': 3, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'nhid': params.nhid, 'noreg': True}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier

        clf = SplitClassifier(self.X, self.y, config)
        devacc, testacc, predictions = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1} for SNLI\n'
                      .format(devacc, testacc))

        a, b, trgts = self.data['test']
        idxs = []
        n = 0
        for line in a:
            idxs.append(' '.join(line) + ' --> ' + ' '.join(b[n]))
            n+=1
           
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0]),
                'indexes': idxs,
                'targets': trgts,
               'predictions': predictions}
