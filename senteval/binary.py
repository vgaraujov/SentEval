# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
Binary classifier and corresponding datasets : MR, CR, SUBJ, MPQA
'''
from __future__ import absolute_import, division, unicode_literals

import io
import os
import numpy as np
import logging

from senteval.tools.validation import InnerKFoldClassifier


class BinaryClassifierEval(object):
    def __init__(self, pos, neg, seed=1111):
        self.seed = seed
        self.samples, self.labels = pos + neg, [1] * len(pos) + [0] * len(neg)
        self.indexes = list(range(len(self.samples)))
        self.n_samples = len(self.samples)

    def do_prepare(self, params, prepare):
        # prepare is given the whole text
        return prepare(params, self.samples)
        # prepare puts everything it outputs in "params" : params.word2id etc
        # Those output will be further used by "batcher".

    def loadFile(self, fpath):
        with io.open(fpath, 'r', encoding='latin-1') as f:
            return [line.split() for line in f.read().splitlines()]

    def run(self, params, batcher):
        enc_input = []
        # Sort to reduce padding
        sorted_corpus = sorted(zip(self.samples, self.labels, self.indexes),
                               key=lambda z: (len(z[0]), z[1], z[2]))
        sorted_samples = [x for (x, y, i) in sorted_corpus]
        sorted_labels = [y for (x, y, i) in sorted_corpus]
        sorted_indexes = [i for (x, y, i) in sorted_corpus]

        if params.save_emb is not None:
            data_filename = '_'.join(params.save_emb.split('_')[:-1]) + '_' + self.task_name + '.npy'
            if os.path.isfile(data_filename):
                logging.info('Loading sentence embeddings')
                enc_input = np.load(data_filename)
                logging.info('Generated sentence embeddings')
            else:
                logging.info('Generating sentence embeddings')
                for ii in range(0, self.n_samples, params.batch_size):
                    batch = sorted_samples[ii:ii + params.batch_size]
                    embeddings = batcher(params, batch)
                    enc_input.append(embeddings)
                enc_input = np.vstack(enc_input)
                logging.info('Generated sentence embeddings')
                logging.info('Saving sentence embeddings')
                np.save(data_filename, enc_input)
        else:
            logging.info('Generating sentence embeddings')
            for ii in range(0, self.n_samples, params.batch_size):
                batch = sorted_samples[ii:ii + params.batch_size]
                embeddings = batcher(params, batch)
                enc_input.append(embeddings)
            enc_input = np.vstack(enc_input)
            logging.info('Generated sentence embeddings')

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}
        clf = InnerKFoldClassifier(enc_input, np.array(sorted_labels), np.array(sorted_indexes), config)
        devacc, testacc, idxs, tgts, preds = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1}\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc, 'ndev': self.n_samples,
                'ntest': self.n_samples, 'indexes': idxs,'targets': tgts, 'predictions': preds}


class CREval(BinaryClassifierEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : CR *****\n\n')
        self.task_name = os.path.basename(task_path)
        pos = self.loadFile(os.path.join(task_path, 'custrev.pos'))
        neg = self.loadFile(os.path.join(task_path, 'custrev.neg'))
        super(self.__class__, self).__init__(pos, neg, seed)


class MREval(BinaryClassifierEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : MR *****\n\n')
        self.task_name = os.path.basename(task_path)
        pos = self.loadFile(os.path.join(task_path, 'rt-polarity.pos'))
        neg = self.loadFile(os.path.join(task_path, 'rt-polarity.neg'))
        super(self.__class__, self).__init__(pos, neg, seed)


class SUBJEval(BinaryClassifierEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : SUBJ *****\n\n')
        self.task_name = os.path.basename(task_path)
        obj = self.loadFile(os.path.join(task_path, 'subj.objective'))
        subj = self.loadFile(os.path.join(task_path, 'subj.subjective'))
        super(self.__class__, self).__init__(obj, subj, seed)


class MPQAEval(BinaryClassifierEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : MPQA *****\n\n')
        self.task_name = os.path.basename(task_path)
        pos = self.loadFile(os.path.join(task_path, 'mpqa.pos'))
        neg = self.loadFile(os.path.join(task_path, 'mpqa.neg'))
        super(self.__class__, self).__init__(pos, neg, seed)
