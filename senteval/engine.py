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
                           'OddManOut', 'CoordinationInversion',
                           "Ar_SubjNumber", "Ar_ObjNumber", "Ar_Aspect", "Ar_Mood", "Ar_Voice", "Ar_PronType",
                           "Ar_SubjDefinite",
                           "Ar_ObjDefinite",
                           "Zh_Aspect", "Zh_Voice",
                           "He_Tense", "He_SubjNumber", "He_ObjNumber", "He_Voice", "He_VerbForm", "He_PronType",
                           "He_SubjGender",
                           "He_ObjGender", "He_SubjDefinite", "He_ObjDefinite",
                           "Hi_Tense", "Hi_SubjNumber", "Hi_ObjNumber", "Hi_Aspect", "Hi_Mood", "Hi_Voice",
                           "Hi_VerbForm", "Hi_PronType",
                           "Hi_SubjGender", "Hi_ObjGender",
                           "Ru_Tense", "Ru_SubjNumber", "Ru_ObjNumber", "Ru_Aspect", "Ru_Mood", "Ru_Voice",
                           "Ru_VerbForm",
                           "Ru_SubjGender", "Ru_ObjGender",
                           "Ta_Tense", "Ta_SubjNumber", "Ta_ObjNumber", "Ta_Mood", "Ta_VerbForm",
                           "Cop_PronType", "Cop_SubjDefinite", "CopObjDefinite",
                           "Sa_Tense", "Sa_SubjNumber", "Sa_ObjNumber", "Sa_Mood", "Sa_VerbForm", "Sa_SubjGender",
                           "Sa_ObjGender",
                           "En_Tense", "En_SubjNumber", "En_ObjNumber", "En_Mood", "En_VerbForm", "En_PronType",
                           'X_Length', 'X_WordContent', 'X_Depth',
                           'X_BigramShift', 'X_Tense', 'X_SubjNumber', 'X_ObjNumber',
                           'X_OddManOut', 'X_CoordinationInversion'
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

        #Xprobe Russian
        elif name == 'X_Length':
            self.evaluation = LengthEval(tpath + '/probing/Xprobe', seed=self.params.seed)
        elif name == 'X_WordContent':
            self.evaluation = WordContentEval(tpath + '/probing/Xprobe', seed=self.params.seed)
        elif name == 'X_Depth':
            self.evaluation = DepthEval(tpath + '/probing/Xprobe', seed=self.params.seed)
        elif name == 'X_BigramShift':
            self.evaluation = BigramShiftEval(tpath + '/probing/Xprobe', seed=self.params.seed)
        elif name == 'X_Tense':
            self.evaluation = TenseEval(tpath + '/probing/Xprobe', seed=self.params.seed)
        elif name == 'X_SubjNumber':
            self.evaluation = SubjNumberEval(tpath + '/probing/Xprobe', seed=self.params.seed)
        elif name == 'X_ObjNumber':
            self.evaluation = ObjNumberEval(tpath + '/probing/Xprobe', seed=self.params.seed)
        elif name == 'X_OddManOut':
            self.evaluation = OddManOutEval(tpath + '/probing/Xprobe', seed=self.params.seed)
        elif name == 'X_CoordinationInversion':
            self.evaluation = CoordinationInversionEval(tpath + '/probing/Xprobe', seed=self.params.seed)


        # Arabic Probing Tasks
        elif name == "Ar_SubjNumber":
            self.evaluation = Multi_SubjNumber(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name ==  "Ar_ObjNumber":
            self.evaluation = Multi_ObjNumber(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name == "Ar_Aspect":
            self.evaluation = Multi_Aspect(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name == "Ar_Mood":
            self.evaluation = Multi_Mood(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name == "Ar_Voice":
            self.evaluation = Multi_Voice(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name == "Ar_PronType":
            self.evaluation = Multi_PronType(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name == "Ar_SubjDefinite":
            self.evaluation = Multi_SubjDefinite(tpath + '/probing/Arabic', seed=self.params.seed)
        elif name == "Ar_ObjDefinite":
            self.evaluation = Multi_ObjDefinite(tpath + '/probing/Arabic', seed=self.params.seed)

        # Chinese Probing Tasks
        elif name == 'Zh_Aspect':
            self.evaluation = Multi_Aspect(tpath + '/probing/Chinese', seed=self.params.seed)
        elif name == 'Zh_Voice':
            self.evaluation = Multi_Voice(tpath + '/probing/Chinese', seed=self.params.seed)

        # Hebrew Probing Tasks
        elif name == "He_PronType":
            self.evaluation = Multi_PronType(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == "He_Tense":
            self.evaluation = Multi_Tense(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == "He_VerbForm":
            self.evaluation = Multi_VerbForm(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == 'He_Voice':
            self.evaluation = Multi_Voice(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == 'He_SubjNumber':
            self.evaluation = Multi_SubjNumber(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == 'He_ObjNumber':
            self.evaluation = Multi_ObjNumber(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == "He_SubjGender":
            self.evaluation = Multi_SubjGender(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == "He_ObjGender":
            self.evaluation = Multi_ObjGender(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == "He_SubjDefinite":
            self.evaluation = Multi_SubjDefinite(tpath + '/probing/Hebrew', seed=self.params.seed)
        elif name == "He_ObjDefinite":
            self.evaluation = Multi_ObjDefinite(tpath + '/probing/Hebrew', seed=self.params.seed)

        # Hindi Probing Tasks
        elif name == "Hi_Tense":
            self.evaluation = Multi_Tense(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == "Hi_SubjNumber":
            self.evaluation = Multi_SubjNumber(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == "Hi_ObjNumber":
            self.evaluation = Multi_ObjNumber(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == "Hi_Aspect":
            self.evaluation = Multi_Aspect(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == "Hi_Mood":
            self.evaluation = Multi_Mood(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == "Hi_Voice":
            self.evaluation = Multi_Voice(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == "Hi_VerbForm":
            self.evaluation = Multi_VerbForm(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == "Hi_PronType":
            self.evaluation = Multi_PronType(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == "SubjGender":
            self.evaluation = Multi_SubjGender(tpath + '/probing/Hindi', seed=self.params.seed)
        elif name == "ObjGender":
            self.evaluation = Multi_ObjGender(tpath + '/probing/Hindi', seed=self.params.seed)


        #Russian Probing Tasks
        elif name == "Ru_SubjNumber":
            self.evaluation = Multi_SubjNumber(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == "Ru_ObjNumber":
            self.evaluation = Multi_ObjNumber(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == "Ru_Tense":
            self.evaluation = Multi_Tense(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == "Ru_Aspect":
            self.evaluation = Multi_Aspect(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == "Ru_Mood":
            self.evaluation = Multi_Mood(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == "Ru_Voice":
            self.evaluation = Multi_Voice(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == "Ru_VerbForm":
            self.evaluation = Multi_VerbForm(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == "Ru_SubjGender":
            self.evaluation = Multi_SubjGender(tpath + '/probing/Russian', seed=self.params.seed)
        elif name == "Ru_ObjGender":
            self.evaluation = Multi_ObjGender(tpath + '/probing/Russian', seed=self.params.seed)

        #Tamil Probing Tasks
        elif name == "Ta_SubjNumber":
            self.evaluation = Multi_SubjNumber(tpath + '/probing/Tamil', seed=self.params.seed)
        elif name == "Ta_ObjNumber":
            self.evaluation = Multi_ObjNumber(tpath + '/probing/Tamil', seed=self.params.seed)
        elif name == "Ta_Tense":
            self.evaluation = Multi_Tense(tpath + '/probing/Tamil', seed=self.params.seed)
        elif name == "Ta_Mood":
            self.evaluation = Multi_Mood(tpath + '/probing/Tamil', seed=self.params.seed)
        elif name == "Ta_VerbForm":
            self.evaluation = Multi_VerbForm(tpath + '/probing/Tamil', seed=self.params.seed)

        #Coptic Probing Tasks
        elif name == "Cop_PronType":
            self.evaluation = Multi_PronType(tpath + '/probing/Coptic', seed=self.params.seed)
        elif name == "Cop_SubjDefinite":
            self.evaluation = Multi_SubjDefinite(tpath + '/probing/Coptic', seed=self.params.seed)
        elif name == "Cop_ObjDefinite":
            self.evaluation = Multi_ObjDefinite(tpath + '/probing/Coptic', seed=self.params.seed)

        #Sanskrit Probing Tasks
        elif name == "Sa_SubjNumber":
            self.evaluation = Multi_SubjNumber(tpath + '/probing/Sanskrit', seed=self.params.seed)
        elif name == "Sa_ObjNumber":
            self.evaluation = Multi_ObjNumber(tpath + '/probing/Sanskrit', seed=self.params.seed)
        elif name == "Sa_Tense":
            self.evaluation = Multi_Tense(tpath + '/probing/Sanskrit', seed=self.params.seed)
        elif name == "Sa_Mood":
            self.evaluation = Multi_Mood(tpath + '/probing/Sanskrit', seed=self.params.seed)
        elif name == "Sa_VerbForm":
            self.evaluation = Multi_VerbForm(tpath + '/probing/Sanskrit', seed=self.params.seed)
        elif name == "Sa_SubjGender":
            self.evaluation = Multi_SubjGender(tpath + '/probing/Sanskrit', seed=self.params.seed)
        elif name == "Sa_ObjGender":
            self.evaluation = Multi_ObjGender(tpath + '/probing/Sanskrit', seed=self.params.seed)

        #English UD Probing Tasks
        elif name == "En_SubjNumber":
            self.evaluation = Multi_SubjNumber(tpath + '/probing/English', seed=self.params.seed)
        elif name == "En_ObjNumber":
            self.evaluation = Multi_ObjNumber(tpath + '/probing/English', seed=self.params.seed)
        elif name == "En_Tense":
            self.evaluation = Multi_Tense(tpath + '/probing/English', seed=self.params.seed)
        elif name == "En_Mood":
            self.evaluation = Multi_Mood(tpath + '/probing/English', seed=self.params.seed)
        elif name == "En_VerbForm":
            self.evaluation = Multi_VerbForm(tpath + '/probing/English', seed=self.params.seed)
        elif name == "En_PronType":
            self.evaluation = Multi_PronType(tpath + '/probing/English', seed=self.params.seed)



        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
