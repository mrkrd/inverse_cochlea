#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np

mem = joblib.Memory("tmp", verbose=1)

def band_pass_filter(signal, fs, band):
    lo, hi = band

    freqs = np.abs( np.fft.fftfreq(signal.size, 1/fs) )

    signal_fft = np.fft.fft(signal)
    signal_fft[ (freqs < lo) | (freqs > hi) ] = 0

    filtered = np.fft.ifft(signal_fft)
    filtered = np.array(filtered.real)

    return filtered




@mem.cache
def run_ear(signal,
            fs,
            cfs,
            anf_num=(0,1000,0)):

    fs_s = 100e3
    s = dsp.resample(signal, len(signal) * fs_s / fs)

    ear = cochlea.Zilany2009_Human(
        anf_num=anf_num,
        cf=cfs
    )
    anf_trains = ear.run(
        s,
        fs_s,
        seed=0
    )

    anf_acc = th.accumulate_spike_trains(
        anf_trains,
        ignore=['index', 'type']
    )
    anf_matrix = th.trains_to_signal(anf_acc, fs).squeeze()

    # anf_matrix -= 5
    # anf_matrix[ anf_matrix < 0 ] = 0

    anfs = []
    for cf,train in zip(cfs, anf_matrix.T):
        anfs.append( (train, fs, cf, anf_num) )

    anfs = np.rec.array(
        anfs,
        dtype=[
            ('trains', float, train.shape),
            ('fs', float),
            ('cfs', float),
            ('anf_num', int, (3,))
        ])

    return anfs




def main():
    pass

if __name__ == "__main__":
    main()
