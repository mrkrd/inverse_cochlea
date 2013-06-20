#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"

import numpy as np
from numpy.testing import assert_array_almost_equal

from inverse_cochlea import (
    isgram_reconstr,
    ANF,
    Spectrogram
)

from inverse_cochlea.isgram_reconstr import Net


def test_make_mlp_sets():

    anf_data = np.arange(10.).reshape((5,2))
    anf = ANF(
        data=anf_data,
        cfs=np.array([1,2]),
        fs=1,
        type='hsr'
    )

    sgram_data = np.arange(10.).reshape((5,2))
    sgram = Spectrogram(
        data=sgram_data,
        fs=1,
        freqs=np.array([1,2]),
        sgram_shift=1
    )


    net = Net(
        net=None,
        freq=1,
        fs=1,
        sgram_shift=1,
        cfs=np.array([1,2]),
        win_len=2
    )

    input_set, target_set = isgram_reconstr._make_mlp_sets(
        net=net,
        anf=anf,
        sgram=sgram
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
        [[ 0.], [ 2.], [ 4.], [ 6.]]
    )
