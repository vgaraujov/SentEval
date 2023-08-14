# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
TREC question-type classification
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np

from senteval.tools.validation import KFoldClassifier


class TRECEval(object):
    def __init__(self, task_path, seed=1111):
        logging.info('***** Transfer task : TREC *****\n\n')
        self.seed = seed
        self.train = self.loadFile(os.path.join(task_path, 'train_5500.label'))
        self.test = self.loadFile(os.path.join(task_path, 'TREC_10.label'))
        self.task_name = os.path.basename(task_path)

    def do_prepare(self, params, prepare):
        samples = self.train['X'] + self.test['X']
        return prepare(params, samples)

    def loadFile(self, fpath):
        trec_data = {'X': [], 'y': []}
        tgt2idx = {'ABBR': 0, 'DESC': 1, 'ENTY': 2,
                   'HUM': 3, 'LOC': 4, 'NUM': 5}
        with io.open(fpath, 'r', encoding='latin-1') as f:
            for line in f:
                target, sample = line.strip().split(':', 1)
                sample = sample.split(' ', 1)[1].split()
                assert target in tgt2idx, target
                trec_data['X'].append(sample)
                trec_data['y'].append(tgt2idx[target])
        return trec_data

    def run(self, params, batcher):
        train_embeddings, test_embeddings = [], []

        # Sort to reduce padding
        sorted_corpus_train = sorted(zip(self.train['X'], self.train['y']),
                                     key=lambda z: (len(z[0]), z[1]))
        train_samples = [x for (x, y) in sorted_corpus_train]
        train_labels = [y for (x, y) in sorted_corpus_train]

        test_indexes = list(range(len(self.test['y'])))
        sorted_corpus_test = sorted(zip(self.test['X'], self.test['y'], test_indexes),
                                    key=lambda z: (len(z[0]), z[1], z[2]))
        test_samples = [x for (x, y, i) in sorted_corpus_test]
        test_labels = [y for (x, y, i) in sorted_corpus_test]
        test_indexes = [i for (x, y, i) in sorted_corpus_test]

        if params.save_emb is not None:
            data_filename = '_'.join(params.save_emb.split('_')[:-1]) + '_' + self.task_name + '.npz'
            if os.path.isfile(data_filename):
                logging.info('Loading sentence embeddings')
                loaded_data = np.load(data_filename)
                train_embeddings, test_embeddings = loaded_data['train_emb'], loaded_data['test_emb']
                logging.info('Generated sentence embeddings')
            else:
                # Get train embeddings
                for ii in range(0, len(train_labels), params.batch_size):
                    batch = train_samples[ii:ii + params.batch_size]
                    embeddings = batcher(params, batch)
                    train_embeddings.append(embeddings)
                train_embeddings = np.vstack(train_embeddings)
                logging.info('Computed train embeddings')

                # Get test embeddings
                for ii in range(0, len(test_labels), params.batch_size):
                    batch = test_samples[ii:ii + params.batch_size]
                    embeddings = batcher(params, batch)
                    test_embeddings.append(embeddings)
                test_embeddings = np.vstack(test_embeddings)
                logging.info('Computed test embeddings')
                logging.info('Saving sentence embeddings')
                np.savez(data_filename, train_emb=train_embeddings, test_emb=test_embeddings)
            
        else:
            # Get train embeddings
            for ii in range(0, len(train_labels), params.batch_size):
                batch = train_samples[ii:ii + params.batch_size]
                embeddings = batcher(params, batch)
                train_embeddings.append(embeddings)
            train_embeddings = np.vstack(train_embeddings)
            logging.info('Computed train embeddings')

            # Get test embeddings
            for ii in range(0, len(test_labels), params.batch_size):
                batch = test_samples[ii:ii + params.batch_size]
                embeddings = batcher(params, batch)
                test_embeddings.append(embeddings)
            test_embeddings = np.vstack(test_embeddings)
            logging.info('Computed test embeddings')

        config_classifier = {'nclasses': 6, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier,
                             'kfold': params.kfold}
        clf = KFoldClassifier({'X': train_embeddings,
                               'y': np.array(train_labels)},
                              {'X': test_embeddings,
                               'y': np.array(test_labels)},
                              config_classifier)
        devacc, testacc, tgts, preds = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} \
            for TREC\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.train['X']), 'ntest': len(self.test['X']), 
                'indexes': np.array(test_indexes), 'targets': tgts, 'predictions': preds}
