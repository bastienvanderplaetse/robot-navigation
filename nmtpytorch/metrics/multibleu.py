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

    def compute(self, refs, hyps, language=None, lowercase=False):
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
            command = "perl ./nmtpytorch/lib/multi-bleu.perl " + refs[0]
            pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            score = pipe.communicate()
            score = [x if x == None else x.decode('UTF-8') for x in score]

        if len(score) == 0:
            return Metric('BLEU', 0, "0.0")
        else:
            score = score[0].strip()
            float_score = float(score.split()[2][:-1])
            verbose_score = score.replace('BLEU = ', '')
            return Metric('BLEU', float_score, verbose_score)
