#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"

import numpy as np
import scipy.signal as dsp

import elmar.waves as wv
import inverse_cochlea
import matplotlib.pyplot as plt


def main():
    fs = 16e3
    t = np.arange(0, 0.1, 1/fs)
    s0 = np.zeros_like(t)

    s1 = dsp.chirp(t, 150, t[-1], 2000)
    s1 = inverse_cochlea.set_dbspl(s1, 60)
    s1[300:400] = 0

    s2 = dsp.chirp(t, 2000, t[-1], 150)
    s2 = inverse_cochlea.set_dbspl(s2, 50)
    s2[300:400] = 0

    s = np.concatenate( (s0, s1, s2) )


    mlp_reconstructor = inverse_cochlea.MlpReconstructor(
        band=(125,2000),
        fs_mlp=8e3,
        hidden_layer=5,
        anf_type=(0,1000,0),
        channel_num=50,
    )


    ### Training
    mlp_reconstructor.train(
        s,
        fs,
        maxfun=1000
    )


    ### Testing
    anf = inverse_cochlea.run_ear(
        sound=s,
        fs=fs,
        cfs=mlp_reconstructor.cfs,
        anf_type=mlp_reconstructor.anf_type
    )
    r, fs = mlp_reconstructor.run(
        anf
    )


    # print("SNR", wv.calc_snr_db(s, s-r))


    fig, ax = plt.subplots(3, 1)
    ax[0].plot(s)
    ax[1].imshow(anf.data.T, aspect='auto')
    ax[2].plot(r)

    plt.show()




if __name__ == "__main__":

    main()