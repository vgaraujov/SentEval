# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals
from pudb import set_trace
from senteval import utils
from senteval.binary import CREval, MREval, MPQAEval, SUBJEval
from senteval.snli import SNLIEval
from senteval.trec import TRECEval
from senteval.sick import SICKRelatednessEval, SICKEntailmentEval
from senteval.mrpc import MRPCEval
from senteval.sts import STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, STSBenchmarkEval
from senteval.sst import SSTEval
from senteval.rank import ImageCaptionRetrievalEval
from senteval.probing import *

class SE(object):
    def __init__(self, params, batcher, prepare=None):
        # parameters
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                           'SICKRelatedness', 'SICKEntailment', 'STSBenchmark',
                           'SNLI', 'ImageCaptionRetrieval', 'STS12', 'STS13',
                           'STS14', 'STS15', 'STS16',
                           'Length', 'WordContent', 'Depth', 'TopConstituents',
                           'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                           'OddManOut', 'CoordinationInversion', 'Mr_Aspect', 'Mr_Case', 'Mr_Deixis',
                           'Mr_Gender', 'Mr_Number', 'Mr_Person', 'Mr_Polarity',
                           'Mr_PronType', 'Mr_Tense', 'Mr_VerbForm',
                           "Ar_Aspect", "Ar_Case", "Ar_Definite", "Ar_Gender", "Ar_Mood", "Ar_Number", "Ar_NumForm",
                           "Ar_NumValue", "Ar_Person", "Ar_PronType", "Ar_Voice",
                           "Zh_Aspect", "Zh_NumType", "Zh_Person", "Zh_Voice",
                           "He_Case", "He_Definite", "He_Gender", "He_HebBinyan", "He_Number", "He_Person",
                           "He_Polarity", "He_PronType", "He_Tense", "He_VerbForm", "He_Voice",
                           "Hi_Aspect", "Hi_Case", "Hi_Gender", "Hi_Mood", "Hi_Number", "Hi_NumType", "Hi_Person",
                           "Hi_PronType", "Hi_Tense", "Hi_VerbForm", "Hi_Voice",
                           "Ru_Animacy", "Ru_Aspect", "Ru_Case", "Ru_Degree", "Ru_Gender", "Ru_Mood", "Ru_Number",
                           "Ru_Person", "Ru_Tense", "Ru_VerbForm", "Ru_Voice",
                           "Ta_Case", "Ta_Gender", "Ta_Mood", "Ta_Number", "Ta_NumType", "Ta_Person", "Ta_PunctType",
                           "Ta_Tense", "Ta_VerbForm"
                           ]

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        # Original SentEval tasks
        if name == 'CR':
            self.evaluation = CREval(tpath + '/downstream/CR', seed=self.params.seed)
        elif name == 'MR':
            self.evaluation = MREval(tpath + '/downstream/MR', seed=self.params.seed)
        elif name == 'MPQA':
            self.evaluation = MPQAEval(tpath + '/downstream/MPQA', seed=self.params.seed)
        elif name == 'SUBJ':
            self.evaluation = SUBJEval(tpath + '/downstream/SUBJ', seed=self.params.seed)
        elif name == 'SST2':
            self.evaluation = SSTEval(tpath + '/downstream/SST/binary', nclasses=2, seed=self.params.seed)
        elif name == 'SST5':
            self.evaluation = SSTEval(tpath + '/downstream/SST/fine', nclasses=5, seed=self.params.seed)
        elif name == 'TREC':
            self.evaluation = TRECEval(tpath + '/downstream/TREC', seed=self.params.seed)
        elif name == 'MRPC':
            self.evaluation = MRPCEval(tpath + '/downstream/MRPC', seed=self.params.seed)
        elif name == 'SICKRelatedness':
            self.evaluation = SICKRelatednessEval(tpath + '/downstream/SICK', seed=self.params.seed)
        elif name == 'STSBenchmark':
            self.evaluation = STSBenchmarkEval(tpath + '/downstream/STS/STSBenchmark', seed=self.params.seed)
        elif name == 'SICKEntailment':
            self.evaluation = SICKEntailmentEval(tpath + '/downstream/SICK', seed=self.params.seed)
        elif name == 'SNLI':
            self.evaluation = SNLIEval(tpath + '/downstream/SNLI', seed=self.params.seed)
        elif name in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            fpath = name + '-en-test'
            self.evaluation = eval(name + 'Eval')(tpath + '/downstream/STS/' + fpath, seed=self.params.seed)
        elif name == 'ImageCaptionRetrieval':
            self.evaluation = ImageCaptionRetrievalEval(tpath + '/downstream/COCO', seed=self.params.seed)

        # Probing Tasks
        elif name == 'Length':
                self.evaluation = LengthEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'WordContent':
                self.evaluation = WordContentEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'Depth':
                self.evaluation = DepthEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'TopConstituents':
                self.evaluation = TopConstituentsEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'BigramShift':
                self.evaluation = BigramShiftEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'Tense':
                self.evaluation = TenseEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'SubjNumber':
                self.evaluation = SubjNumberEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'ObjNumber':
                self.evaluation = ObjNumberEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'OddManOut':
                self.evaluation = OddManOutEval(tpath + '/probing', seed=self.params.seed)
        elif name == 'CoordinationInversion':
                self.evaluation = CoordinationInversionEval(tpath + '/probing', seed=self.params.seed)

        # Multilingual Probing Tasks
        elif name == 'Mr_Aspect':
            self.evaluation = Multi_Aspect(tpath + '/probing/marathi', seed=self.params.seed)
        elif name == 'Mr_Case':
            self.evaluation = Multi_Case(tpath + '/probing/marathi', seed=self.params.seed)
        elif name == 'Mr_Deixis':
            self.evaluation = Multi_Deixis(tpath + '/probing/marathi', seed=self.params.seed)
        elif name == 'Mr_Gender':
            self.evaluation = Multi_Gender(tpath + '/probing/marathi', seed=self.params.seed)
        elif name == 'Mr_Number':
            self.evaluation = Multi_Number(tpath + '/probing/marathi', seed=self.params.seed)
        elif name == 'Mr_Person':
            self.evaluation = Multi_Person(tpath + '/probing/marathi', seed=self.params.seed)
        elif name == 'Mr_Polarity':
            self.evaluation = Multi_Polarity(tpath + '/probing/marathi', seed=self.params.seed)
        elif name == 'Mr_PronType':
            self.evaluation = Multi_PronType(tpath + '/probing/marathi', seed=self.params.seed)
        elif name == 'Mr_Tense':
            self.evaluation = Multi_Tense(tpath + '/probing/marathi', seed=self.params.seed)
        elif name == 'Mr_VerbForm':
            self.evaluation = Multi_VerbForm(tpath + '/probing/marathi', seed=self.params.seed)

        # Arabic Probing Tasks
        elif name == 'Ar_Aspect':
            self.evaluation = Multi_Aspect(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name == 'Ar_Case':
            self.evaluation = Multi_Case(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name == "Ar_Definite":
            self.evaluation = Multi_Definite(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name == 'Ar_Gender':
            self.evaluation = Multi_Gender(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name == "Ar_Mood":
            self.evaluation = Multi_Mood(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name == "Ar_Number":
            self.evaluation = Multi_Number(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name == "Ar_NumForm":
            self.evaluation = Multi_NumForm(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name == "Ar_NumValue":
            self.evaluation = Multi_NumValue(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name == "Ar_Person":
            self.evaluation = Multi_Person(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name == 'PronType':
            self.evaluation = Multi_PronType(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name == 'Ar_Voice':
            self.evaluation = Multi_Voice(tpath + '/probing/Arabic', seed=self.params.seed)

        # Chinese Probing Tasks
        elif name == 'Zh_Aspect':
            self.evaluation = Multi_Aspect(tpath + '/probing/Chinese', seed=self.params.seed)
        elif name == 'Zh_NumType':
            self.evaluation = Multi_NumType(tpath + '/probing/Chinese', seed=self.params.seed)
        elif name == 'Zh_Person':
            self.evaluation = Multi_Person(tpath + '/probing/Chinese', seed=self.params.seed)
        elif name == 'Zh_Voice':
            self.evaluation = Multi_Voice(tpath + '/probing/Chinese', seed=self.params.seed)

        # Hebrew Probing Tasks
        elif name == "He_Case":
            self.evaluation = Multi_Case(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == "He_Definite":
            self.evaluation = Multi_Definite(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == "He_Gender":
            self.evaluation = Multi_Gender(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == 'He_HebBinyan':
            self.evaluation = Multi_HebBinyan(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == "He_Number":
            self.evaluation = Multi_Number(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == "He_Person":
            self.evaluation = Multi_Person(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == "He_Polarity":
            self.evaluation = Multi_Polarity(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == "He_PronType":
            self.evaluation = Multi_PronType(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == "He_Tense":
            self.evaluation = Multi_Tense(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == "He_VerbForm":
            self.evaluation = Multi_VerbForm(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == 'He_Voice':
            self.evaluation = Multi_Voice(tpath + '/probing/Hebrew', seed=self.params.seed)

        # Hindi Probing Tasks
        elif name == 'Hi_Aspect':
            self.evaluation = Multi_Aspect(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == 'Hi_Case':
            self.evaluation = Multi_Case(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == 'Hi_Gender':
            self.evaluation = Multi_Gender(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == "Hi_Mood":
            self.evaluation = Multi_Mood(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == "Hi_Number":
            self.evaluation = Multi_Number(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == "Hi_NumType":
            self.evaluation = Multi_NumType(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == "Hi_Person":
            self.evaluation = Multi_Person(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == "Hi_PronType":
            self.evaluation = Multi_PronType(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == "Hi_Tense":
            self.evaluation = Multi_Tense(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == "Hi_VerbForm":
            self.evaluation = Multi_VerbForm(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == 'Hi_Voice':
            self.evaluation = Multi_Voice(tpath + '/probing/Hindi', seed=self.params.seed)

        #Russian Probing Tasks
        elif name == 'Ru_Animacy':
            self.evaluation = Multi_Animacy(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == 'Ru_Aspect':
            self.evaluation = Multi_Aspect(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == 'Ru_Case':
            self.evaluation = Multi_Case(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == 'Ru_Degree':
            self.evaluation = Multi_Degree(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == 'Ru_Gender':
            self.evaluation = Multi_Gender(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == 'Ru_Mood':
            self.evaluation = Multi_Mood(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == 'Ru_Number':
            self.evaluation = Multi_Number(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == 'Ru_Person':
            self.evaluation = Multi_Person(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == 'Ru_Tense':
            self.evaluation = Multi_Tense(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == 'Ru_VerbForm':
            self.evaluation = Multi_VerbForm(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == 'Ru_Voice':
            self.evaluation = Multi_Voice(tpath + '/probing/Russian', seed=self.params.seed)

        #Tamil Probing Tasks
        elif name == 'Ta_Case':
            self.evaluation = Multi_Case(tpath + '/probing/Tamil', seed=self.params.seed)
        elif name == "Ta_Gender":
            self.evaluation = Multi_Gender(tpath + '/probing/Tamil', seed=self.params.seed)
        elif name == "Ta_Mood":
            self.evaluation = Multi_Mood(tpath + '/probing/Tamil', seed=self.params.seed)
        elif name == "Ta_Number":
            self.evaluation = Multi_Number(tpath + '/probing/Tamil', seed=self.params.seed)
        elif name == "Ta_NumType":
            self.evaluation = Multi_NumType(tpath + '/probing/Tamil', seed=self.params.seed)
        elif name == "Ta_Person":
            self.evaluation = Multi_Person(tpath + '/probing/Tamil', seed=self.params.seed)
        elif name == 'Ta_PunctType':
            self.evaluation = Multi_PunctType(tpath + '/probing/Tamil', seed=self.params.seed)
        elif name == "Ta_Tense":
            self.evaluation = Multi_Tense(tpath + '/probing/Tamil', seed=self.params.seed)
        elif name == "Ta_VerbForm":
            self.evaluation = Multi_VerbForm(tpath + '/probing/Tamil', seed=self.params.seed)
        elif name == 'Ta_Voice':
            self.evaluation = Multi_Voice(tpath + '/probing/Tamil', seed=self.params.seed)

        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
