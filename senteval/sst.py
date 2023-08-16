# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
SST - binary classification
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np

from senteval.tools.validation import SplitClassifier


class SSTEval(object):
    def __init__(self, task_path, nclasses=2, seed=1111):
        self.seed = seed

        # binary of fine-grained
        assert nclasses in [2, 5]
        self.nclasses = nclasses
        self.task_name = 'Binary' if self.nclasses == 2 else 'Fine-Grained'
        logging.debug('***** Transfer task : SST %s classification *****\n\n', self.task_name)

        train = self.loadFile(os.path.join(task_path, 'sentiment-train'))
        dev = self.loadFile(os.path.join(task_path, 'sentiment-dev'))
        test = self.loadFile(os.path.join(task_path, 'sentiment-test'))
        self.sst_data = {'train': train, 'dev': dev, 'test': test}

    def do_prepare(self, params, prepare):
        samples = self.sst_data['train']['X'] + self.sst_data['dev']['X'] + \
                  self.sst_data['test']['X']
        return prepare(params, samples)

    def loadFile(self, fpath):
        sst_data = {'X': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if self.nclasses == 2:
                    sample = line.strip().split('\t')
                    sst_data['y'].append(int(sample[1]))
                    sst_data['X'].append(sample[0].split())
                elif self.nclasses == 5:
                    sample = line.strip().split(' ', 1)
                    sst_data['y'].append(int(sample[0]))
                    sst_data['X'].append(sample[1].split())
        assert max(sst_data['y']) == self.nclasses - 1
        return sst_data

    def run(self, params, batcher):
        sst_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size

        if params.save_emb is not None:
            data_filename = '_'.join(params.save_emb.split('_')[:-1]) + '_SST' + self.task_name + '.npy'
            if os.path.isfile(data_filename):
                logging.info('Loading sentence embeddings')
                sst_embed = np.load(data_filename, allow_pickle = True)
                logging.info('Generated sentence embeddings')
            else:
                for key in self.sst_data:
                    logging.info('Computing embedding for {0}'.format(key))
                    # Sort to reduce padding
                    indexes = list(range(len(self.sst_data[key]['y'])))
                    sorted_data = sorted(zip(self.sst_data[key]['X'],
                                             self.sst_data[key]['y'],
                                             indexes),
                                         key=lambda z: (len(z[0]), z[1], z[2]))
                    self.sst_data[key]['X'], self.sst_data[key]['y'], self.sst_data[key]['idx'] = map(list, zip(*sorted_data))

                    sst_embed[key]['X'] = []
                    for ii in range(0, len(self.sst_data[key]['y']), bsize):
                        batch = self.sst_data[key]['X'][ii:ii + bsize]
                        embeddings = batcher(params, batch)
                        sst_embed[key]['X'].append(embeddings)
                    sst_embed[key]['X'] = np.vstack(sst_embed[key]['X'])
                    sst_embed[key]['y'] = np.array(self.sst_data[key]['y'])
                    sst_embed[key]['idx'] = np.array(self.sst_data[key]['idx'])
                    logging.info('Computed {0} embeddings'.format(key))
                logging.info('Saving sentence embeddings')
                np.save(data_filename, sst_embed)
        else:
            for key in self.sst_data:
                logging.info('Computing embedding for {0}'.format(key))
                # Sort to reduce padding
                indexes = list(range(len(self.sst_data[key]['y'])))
                sorted_data = sorted(zip(self.sst_data[key]['X'],
                                         self.sst_data[key]['y'],
                                         indexes),
                                     key=lambda z: (len(z[0]), z[1], z[2]))
                self.sst_data[key]['X'], self.sst_data[key]['y'], self.sst_data[key]['idx'] = map(list, zip(*sorted_data))

                sst_embed[key]['X'] = []
                for ii in range(0, len(self.sst_data[key]['y']), bsize):
                    batch = self.sst_data[key]['X'][ii:ii + bsize]
                    embeddings = batcher(params, batch)
                    sst_embed[key]['X'].append(embeddings)
                sst_embed[key]['X'] = np.vstack(sst_embed[key]['X'])
                sst_embed[key]['y'] = np.array(self.sst_data[key]['y'])
                sst_embed[key]['idx'] = np.array(self.sst_data[key]['idx'])
                logging.info('Computed {0} embeddings'.format(key))

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier}

        clf = SplitClassifier(X={'train': sst_embed['train']['X'],
                                 'valid': sst_embed['dev']['X'],
                                 'test': sst_embed['test']['X']},
                              y={'train': sst_embed['train']['y'],
                                 'valid': sst_embed['dev']['y'],
                                 'test': sst_embed['test']['y']},
                              config=config_classifier)

        devacc, testacc, predictions = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} for \
            SST {2} classification\n'.format(devacc, testacc, self.task_name))
        
        preds = {}
        n = 0
        for line in self.sst_data['test']['X']:
            preds[n]=[' '.join(line), self.sst_data['test']['y'][n], predictions[n]]
            n+=1

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(sst_embed['dev']['X']),
                'ntest': len(sst_embed['test']['X']),
                'indexes': [' '.join(line) for line in self.sst_data['test']['X']],
                'targets': self.sst_data['test']['y'],
               'predictions': predictions}
