#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"

import numpy as np

from inverse_cochlea import mlp_reconstr


from inverse_cochlea import (
    ANF,
    Signal
)


def test_make_mlp_sets():

    anf_data = np.arange(10).reshape((5,2))
    anf = ANF(data=anf_data, cfs=[1,2], fs=1, type='hsr')

    signal_data = np.arange(5)
    signal = Signal(data=signal_data, fs=1)

    input_set, target_set = mlp_reconstr._make_mlp_sets(
        win_len=2,
        fs=1,
        anf=anf,
        signal=signal
    )
