#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"

import numpy as np
from numpy.testing import assert_array_almost_equal

from inverse_cochlea import mlp_reconstr


from inverse_cochlea import (
    ANF,
    Signal
)


def test_make_mlp_sets():

    anf_data = np.arange(10.).reshape((5,2))
    anf = ANF(data=anf_data, cfs=[1,2], fs=1, type='hsr')

    signal_data = np.arange(5.)
    signal = Signal(data=signal_data, fs=1)

    input_set, target_set = mlp_reconstr._make_mlp_sets(
        anf=anf,
        signal=signal,
        win_len=2,
        fs_net=1,
    )



    assert_array_almost_equal(
        input_set,
        [[ 0.,  1.,  2.,  3.],
         [ 2.,  3.,  4.,  5.],
         [ 4.,  5.,  6.,  7.],
         [ 6.,  7.,  8.,  9.]]
    )

    assert_array_almost_equal(
        target_set,
        [[ 0.], [ 1.], [ 2.], [ 3.]]
    )
