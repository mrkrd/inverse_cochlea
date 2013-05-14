#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
import scipy.signal as dsp
import multiprocessing
import cPickle as pickle
from collections import namedtuple

import cochlea
import elmar.thorns as th

import joblib
mem = joblib.Memory("work", verbose=2)


ANF = namedtuple("ANF", "data, fs, cfs, type")
Signal = namedtuple("Signal", "data, fs")


class Reconstructor(object):

    def dump(self, fname):
        pickle.dump(self, open(fname, 'w'))






#@mem.cache
def run_ear(
        sound,
        fs,
        cf,
        anf_type='msr',
        cohc=1,
        cihc=1
):

    assert sound.ndim == 1

    fs_model = 100e3
    sound_model = dsp.resample(sound, len(sound) * fs_model / fs)


    if isinstance(anf_type, tuple):
        anf_trains = cochlea.run_zilany2013(
            sound=sound_model,
            fs=fs_model,
            anf_num=anf_type,
            seed=0,
            cf=cf,
            cohc=cohc,
            cihc=cihc,
            species='human',
        )
        acc = th.accumulate_spike_trains(
            anf_trains,
            keep=['duration', 'cf']
        )
        arr = th.trains_to_array(acc, fs_model)
        cfs = np.array(acc['cf'])

    elif anf_type in ('hsr','msr','lsr'):
        rates = cochlea.run_zilany2013_rate(
            sound=sound_model,
            fs=fs_model,
            anf_types=[anf_type],
            cf=cf,
            cohc=cohc,
            cihc=cihc,
            species='human'
        )

        arr = np.array(
            rates
        )
        cfs = np.array(
            rates.columns.get_level_values('cf')
        )


    anf = ANF(data=arr, fs=fs_model, cfs=cfs, type=anf_type)

    return anf





def main():
    fs = 16e3
    t = np.arange(0, 0.1, 1/fs)
    s = np.sin(2 * np.pi * t * 1000)
    s = cochlea.set_dbspl(s, 50)

    anf = run_ear(
        sound=s,
        fs=fs,
        cfs=(80, 2000, 10),
        anf_type='msr'
    )


    plt.imshow(
        anf.data.T,
        aspect='auto'
    )
    plt.show()



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
