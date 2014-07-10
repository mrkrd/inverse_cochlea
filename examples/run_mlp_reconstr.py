#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

__author__ = "Marek Rudnicki"

import numpy as np
import scipy.signal as dsp

import thorns.waves as wv
import inverse_cochlea
import matplotlib.pyplot as plt


def main():

    ### Make sound
    fs = 16e3
    t = np.arange(0, 0.1, 1/fs)
    s0 = np.zeros_like(t)

    s1 = dsp.chirp(t, 150, t[-1], 2000)
    s1 = inverse_cochlea.set_dbspl(s1, 60)
    s1[300:400] = 0

    s2 = dsp.chirp(t, 2000, t[-1], 150)
    s2 = inverse_cochlea.set_dbspl(s2, 50)
    s2[300:400] = 0

    sound = np.concatenate( (s0, s1, s2) )



    ### Setup the neural network
    mlp_reconstructor = inverse_cochlea.MlpReconstructor(
        band=(125,2000),
        fs_net=8e3,
        hidden_layer_size=5,
        anf_type='msr',
        channel_num=50,
    )



    ### Train
    mlp_reconstructor.train(
        sound,
        fs,
        maxfun=200
    )



    ### Test
    anf = inverse_cochlea.run_ear(
        sound=sound,
        fs=fs,
        cf=mlp_reconstructor.cfs,
        anf_type=mlp_reconstructor.anf_type
    )
    reconstruction, fs_reconstruction = mlp_reconstructor.run(
        anf
    )



    ### Plot results
    fig, ax = plt.subplots(3, 1, sharex=True)

    wv.plot_signal(sound, fs=fs, ax=ax[0])

    ax[1].imshow(
        np.flipud(anf.data.T),
        aspect='auto',
        extent=(0, anf.data.shape[0]/anf.fs, 0, anf.data.shape[1])
    )

    wv.plot_signal(reconstruction, fs=fs_reconstruction, ax=ax[2])

    plt.show()




if __name__ == "__main__":

    main()
