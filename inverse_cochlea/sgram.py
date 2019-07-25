#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
from collections import namedtuple

import joblib
mem = joblib.Memory(cachedir="work", verbose=2)


_backend = None


if _backend is None:
    try:
        import oct2py
        octave = oct2py.Oct2Py()
        _backend = 'oct2py'
    except ImportError:
        _backend = None

if _backend is None:
    try:
        import pytave
        _backend = 'pytave'
    except ImportError:
        _backend = None




if _backend == 'pytave':
    pytave.addpath("~/.local/opt/ltfat")
    pytave.addpath("/nfs/system/opt/ltfat")
    pytave.eval(0, 'ltfatstart')
elif _backend == 'oct2py':
    octave.addpath("~/.local/opt/ltfat")
    octave.addpath("/nfs/system/opt/ltfat")
    octave.call("ltfatstart", nout=0)



Spectrogram = namedtuple("Spectrogram", "data, fs, freqs, sgram_shift")


def calc_sgram_pytave(signal, fs, channel_num, sgram_shift):

    assert channel_num % 2, "channel_num must be odd"
    dgtreal_channel_num = (channel_num - 1) * 2

    ss = pytave.feval(
        1,
        'dgtreal',
        signal,
        'gauss',
        sgram_shift,
        dgtreal_channel_num
    )
    ss = ss[0].squeeze()

    sg = np.abs(ss.T)**2

    sgram = Spectrogram(
        data=sg,
        fs=fs,
        freqs=np.linspace(0, fs/2, sg.shape[1]),
        sgram_shift=sgram_shift
    )

    return sgram


def calc_sgram_oct2py(signal, fs, channel_num, sgram_shift):

    assert channel_num % 2, "channel_num must be odd"
    dgtreal_channel_num = (channel_num - 1) * 2

    octave.put('signal', signal)
    octave.run(
        'ss = dgtreal(signal, "gauss", {sgram_shift}, {dgtreal_channel_num})'.format(
            sgram_shift=sgram_shift,
            dgtreal_channel_num=dgtreal_channel_num
        )
    )
    ss = octave.get('ss')

    sg = np.abs(ss.T)**2

    sgram = Spectrogram(
        data=sg,
        fs=fs,
        freqs=np.linspace(0, fs/2, sg.shape[1]),
        sgram_shift=sgram_shift
    )

    return sgram




@mem.cache
def calc_isgram_pytave(sgram, iter_num=1000):

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
        sgram.sgram_shift,
        channel_num,
        'maxit',
        iter_num)
    r = r[0].squeeze()

    return r






@mem.cache
def calc_isgram_oct2py(sgram, iter_num=1000):

    channel_num = 2 * (sgram.data.shape[1] - 1)

    ### Spectrogram len must be multiple of channel_num
    m = len(sgram.data) % channel_num
    if m > 0:
        pad_len = channel_num - m
        pad = np.zeros(( pad_len, sgram.data.shape[1] ))
        padded = np.append(sgram.data, pad, axis=0)
    else:
        padded = sgram.data

    octave.put('padded', padded.T)
    octave.run(
        'r = isgramreal(padded, "gauss", {sgram_shift}, {channel_num}, "maxit", {iter_num});'.format(
            sgram_shift=sgram.sgram_shift,
            channel_num=channel_num,
            iter_num=iter_num
        ),
        verbose=True
    )
    r = octave.get('r')
    r = np.squeeze(r)

    return r






def main():
    sgram_shift = 1

    fs = 16e3
    t = np.arange(0, 0.1, 1/fs)
    s = dsp.chirp(t, 1000, t[-1], 6000)
    s[100:300] = 0

    sgram = calc_sgram(
        s,
        fs,
        channel_num=51,
        sgram_shift=sgram_shift
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




if _backend == 'oct2py':
    calc_isgram = calc_isgram_oct2py
    calc_sgram = calc_sgram_oct2py
elif _backend == 'pytave':
    calc_isgram = calc_isgram_pytave
    calc_sgram = calc_sgram_pytave
else:
    raise ImportError



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.signal as dsp

    main()
