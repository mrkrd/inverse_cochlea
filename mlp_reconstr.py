#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"

import numpy as np
import scipy.signal as dsp

import ffnet
import joblib

import mrlib.waves as wv

from common import (
    run_ear,
    Reconstructor,
    Signal
)


mem = joblib.Memory("work", verbose=2)






class MlpReconstructor(Reconstructor):
    def __init__(
            self,
            band=(125, 2000),
            fs_net=8e3,
            hidden_layer=4,
            channel_num=10,
            win_len=10e-3,
            anf_type='msr'
    ):


        assert fs_net/2 >= band[1], "Nyquist"


        self.fs_net = fs_net
        self.band = band
        self.channel_num = channel_num
        self.anf_type = anf_type

        self.cfs = None



        self.win_len = win_len
        win_samp = int(np.round(
            self.win_len * fs_net
        ))


        if hidden_layer > 1:
            conec = ffnet.mlgraph(
                (win_samp*channel_num,
                 int(hidden_layer),
                 1)
            )

        elif hidden_layer > 0:
            conec = ffnet.mlgraph(
                (win_samp*channel_num,
                 int(win_samp*channel_num*hidden_layer),
                 1)
            )

        elif hidden_layer == 0:
            conec = ffnet.mlgraph(
                (win_samp*channel_num,
                 1)
            )

        else:
            raise RuntimeError("hidden_layer should not be negative")


        np.random.seed(0)
        self.net = ffnet.ffnet(conec)





    def train(self, sound, fs, filter=True, **kwargs):


        if filter:
            print("Filtering the signal:", self.band)
            sound = wv.fft_filter(sound, fs, self.band)


        if isinstance(self.band, tuple):
            cf = (self.band[0], self.band[1], self.channel_num)
        else:
            cf = self.band

        anf = run_ear(
            sound=sound,
            fs=fs,
            cf=cf,
            anf_type=self.anf_type
        )

        if self.cfs is None:
            self.cfs = anf.cfs
        else:
            assert np.all(self.cfs == anf.cfs)


        signal = Signal(sound, fs)

        self.net = _train_tnc(
            net=self.net,
            fs_net=self.fs_net,
            win_len=self.win_len,
            anf=anf,
            signal=signal,
            **kwargs
        )



    def run(self, anf, filter=True):

        ### Check anf_type
        assert self.anf_type == anf.type, "{} {}".format(self.anf_type, anf.type)
        assert np.all(self.cfs == anf.cfs)


        ### Run MLP
        input_set, _ = _make_mlp_sets(
            win_len=self.win_len,
            fs_net=self.fs_net,
            anf=anf
        )

        output_set = self.net(input_set)
        sound = output_set.squeeze()
        # sound = dsp.resample(sound, len(sound) * self.fs / self._net.fs)

        if filter:
            sound = wv.fft_filter(sound, self.fs_net, self.band)

        return Signal(sound, self.fs_net)





@mem.cache(ignore=['nproc'])
def _train_tnc(net, fs_net, win_len, anf, signal, nproc=None, **kwargs):

    input_set, target_set = _make_mlp_sets(
        win_len=win_len,
        fs_net=fs_net,
        anf=anf,
        signal=signal
    )

    net.train_tnc(
        input_set,
        target_set,
        messages=1,
        nproc=nproc,
        **kwargs
    )

    return net





def _make_mlp_sets(win_len, fs_net, anf, signal=None):

    ### Window length in samples
    win_samp = int(np.round(win_len*fs_net))


    ### Resample data to the desired output fs
    anf_data = dsp.resample(
        anf.data,
        len(anf.data) * fs_net / anf.fs
    )

    if signal is not None:
        signal_data = dsp.resample(
            signal.data,
            len(signal.data) * fs_net / signal.fs
        )


    ### Find the shorter data
    if signal is None:
        data_len = len(anf_data)
    else:
        data_len = min([len(anf_data), len(signal_data)])


    input_set = []
    target_set = []
    for i in np.arange(data_len - win_samp + 1):
        lo = i
        hi = i + win_samp

        input_set.append( anf_data[lo:hi].flatten() )

        if signal is not None:
            target_set.append( [signal_data[lo]] )

    input_set = np.array(input_set, dtype=float)
    target_set = np.array(target_set, dtype=float)

    return input_set, target_set
