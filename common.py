#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
import scipy.signal as dsp
import multiprocessing
import cPickle as pickle

import cochlea
import marlib.thorns as th

import joblib
mem = joblib.Memory("work", verbose=2)



class Reconstructor(object):

    def dump(self, fname):
        pickle.dump(self, open(fname, 'w'))



def band_pass_filter(signal, fs, band):
    lo, hi = band

    freqs = np.abs( np.fft.fftfreq(signal.size, 1/fs) )

    signal_fft = np.fft.fft(signal)
    signal_fft[ (freqs < lo) | (freqs > hi) ] = 0

    filtered = np.fft.ifft(signal_fft)
    filtered = np.array(filtered.real)

    return filtered




@mem.cache
def run_ear(sound,
            fs,
            cfs,
            anf=(0,1000,0),
            cohc=1,
            cihc=1):

    assert sound.ndim == 1

    fs_model = 100e3
    sound_model = dsp.resample(sound, len(sound) * fs_model / fs)


    if isinstance(anf, tuple):
        anf = cochlea.run_zilany2009_human(
            sound=sound_model,
            fs=fs_model,
            anf_num=anf,
            seed=0,
            cf=cfs,
            cohc=cohc,
            cihc=cihc,
            parallel=True
        )
        acc = th.accumulate_spike_trains(
            anf,
            keep=['duration', 'cf']
        )
        arr = th.trains_to_array(acc, fs_model)
        cfs = acc['cf']

    elif anf in ('hsr','msr','lsr'):
        arr,cfs = cochlea.run_zilany2009_human_psp(
            sound=sound_model,
            fs=fs_model,
            anf_type=anf,
            seed=0,
            cf=cfs,
            cohc=cohc,
            cihc=cihc,
            parallel=True
        )



    arr = dsp.resample(arr, len(arr) * fs / fs_model)


    return arr, cfs


def _run_ear_helper( (anf_num, cf, sound, fs, cohc, cihc) ):

    print "CF:", cf

    fs_model = 100e3
    sound_model = dsp.resample(sound, len(sound) * fs_model / fs)

    ear = cochlea.Zilany2009_Human(
        anf_num=anf_num,
        cf=cf,
        cohc=cohc,
        cihc=cihc
    )

    anf = ear.run(
        sound_model,
        fs_model,
        seed=0
    )

    anf_acc = th.accumulate_spike_trains(
        anf,
        ignore=['index', 'type']
    )
    train = th.trains_to_array(anf_acc, fs_model).squeeze()
    train = dsp.resample(train, len(train) * fs / fs_model)

    return train


def main():
    pass

if __name__ == "__main__":
    main()
