# -*- coding: utf-8 -*-
import subprocess
import pkg_resources

from ..utils.misc import listify
from .metric import Metric

import platform

BLEU_SCRIPT = pkg_resources.resource_filename('nmtpytorch',
                                              'lib/multi-bleu.perl')


class BLEUScorer:
    """BLEUScorer class."""
    def __init__(self):
        # For multi-bleu.perl we give the reference(s) files as argv,
        # while the candidate translations are read from stdin.
        self.__cmdline = [BLEU_SCRIPT]

    def compute(self, refs, hyps, filename= None, language=None, lowercase=False):
        if (platform.system() == "Linux"):
            cmdline = self.__cmdline[:]

            if lowercase:
                cmdline.append("-lc")

            # Make reference files a list
            cmdline.extend(listify(refs))

            if isinstance(hyps, str):
                hypstring = open(hyps).read().strip()
            elif isinstance(hyps, list):
                hypstring = "\n".join(hyps)

            score = subprocess.run(cmdline, stdout=subprocess.PIPE,
                               input=hypstring,
                               universal_newlines=True).stdout.splitlines()
        elif (platform.system() == "Windows"):
            cmdline = self.__cmdline[:]
            if lowercase:
                cmdline.append("-lc")

            cmdline.extend(listify(refs))

            # if isinstance(hyps, str):
            #     hypstring = open(hyps).read().strip()
            # elif isinstance(hyps, list):
            #     hypstring = "\n".join(hyps)       
            
            # command = "perl ./nmtpytorch/lib/multi-bleu.perl " + refs[0] + " < " + hypstring

            print(filename)

            # command = "perl ./nmtpytorch/lib/multi-bleu.perl {0} < ./{1}".format(refs[0], filename) 
            command = "perl ./nmtpytorch/lib/multi-bleu.perl {}".format(refs[0])
            f = open(filename)

            # pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)#, universal_newlines=True)
            score = subprocess.Popen(command, stdin=f, stdout=subprocess.PIPE).communicate()
            # score = pipe.communicate()
            print(score)
            score = [x if x == None else x.decode('UTF-8') for x in score]
            print(score)

        if len(score) == 0:
            return Metric('BLEU', 0, "0.0")
        else:
            score = score[0].strip()
            print(score)
            float_score = float(score.split()[2][:-1])
            print(float_score)
            verbose_score = score.replace('BLEU = ', '')
            return Metric('BLEU', float_score, verbose_score)
