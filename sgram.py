#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
from collections import namedtuple

import pytave
import joblib

pytave.addpath("~/src/ltfat")
pytave.addpath("/nfs/system/opt/ltfat")
pytave.eval(0, 'ltfatstart')

mem = joblib.Memory(cachedir="tmp", verbose=2)

SGram = namedtuple("SGram", "data, fs, freqs, time_shift")

def calc_sgram(signal, fs, channel_num, time_shift):

    assert channel_num % 2, "channel_num must be odd"
    dgtreal_channel_num = (channel_num - 1) * 2

    ss = pytave.feval(
        1,
        'dgtreal',
        signal,
        'gauss',
        time_shift,
        dgtreal_channel_num
    )
    ss = ss[0].squeeze()

    sg = np.abs(ss.T)**2

    sgram = SGram(
        data=sg,
        fs=fs,
        freqs=np.linspace(0, fs/2, sg.shape[1]),
        time_shift=time_shift
    )

    return sgram



@mem.cache
def calc_isgram(sgram, iter_num=1000):

    channel_num = 2 * (sgram.data.shape[1] - 1)

    ### Spectrogram len must be multiple of channel_num
    m = len(sgram.data) % channel_num
    if m > 0:
        pad_len = channel_num - m
        pad = np.zeros(( pad_len, sgram.data.shape[1] ))
        padded = np.append(sgram.data, pad, axis=0)
    else:
        padded = sgram.data


    r = pytave.feval(
        1,
        'isgramreal',
        padded.T,
        'gauss',
        sgram.time_shift,
        channel_num,
        'maxit',
        iter_num)
    r = r[0].squeeze()

    return r






def main():
    time_shift = 1

    fs = 16e3
    t = np.arange(0, 0.1, 1/fs)
    s = dsp.chirp(t, 1000, t[-1], 6000)
    s[100:300] = 0

    sgram = calc_sgram(
        s,
        fs,
        channel_num=51,
        time_shift=time_shift
    )
    print sgram.freqs

    r = calc_isgram(
        sgram,
        iter_num=200
    )


    fig,ax = plt.subplots(
        nrows=3,
        ncols=1
    )
    ax[0].plot(s)
    ax[1].imshow(sgram.data.T, aspect='auto')
    ax[2].plot(r)
    plt.show()




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.signal as dsp

    main()
