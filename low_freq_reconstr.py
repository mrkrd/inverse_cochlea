#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
import scipy.signal as dsp
from collections import namedtuple

import ffnet
import joblib

import cochlea
import thorns as th

from common import run_ear, band_pass_filter

Net = namedtuple("Net", "net, fs, cfs, win_len")
Signal = namedtuple("Signal", "data, fs")
mem = joblib.Memory("tmp", verbose=1)

class LowFreqReconstructor(object):
    def __init__(self,
                 band=(80, 2000),
                 fs_mlp=8e3,
                 hidden_layer=0.25,
                 channel_num=10,
                 anf_num=(0,1000,0)
             ):

        self.fs = None
        self.band = band
        self.fs_mlp = fs_mlp
        self.channel_num = channel_num
        self.anf_num = anf_num
        self._hidden_layer = hidden_layer

        self._net = None



    def train(self, sound, fs, iter_num=1000):
        if self.fs is None:
            self.fs = fs
        else:
            assert self.fs == fs

        s = Signal(sound, fs)


        anfs = run_ear(
            s.data,
            s.fs,
            (self.band[0], self.band[1], self.channel_num),
            self.anf_num
        )

        if self.cfs is None:
            self.cfs = np.unique(anfs.cfs)




        if self._net is None:
            win_len = np.round(10e-3 * self.fs_mlp)

            if self._hidden_layer > 0:
                conec = ffnet.mlgraph(
                    (win_len*self.channel_num,
                     int(win_len*self.channel_num*self._hidden_layer),
                     1)
                )
            else:
                conec = ffnet.mlgraph(
                    (win_len*self.channel_num, 1)
                )

            net = ffnet.ffnet(conec)

            self._net = Net(
                net=net,
                fs=fs_mlp,
                cfs=self.cfs,
                win_len=win_len
            )



        self._net = _train(
            net=self._net,
            anfs=anfs,
            signal=s,
            iter_num=iter_num
        )



@mem.cache
def _train(net, anfs, signal, iter_num):
    pass



def _make_mlp_data(net, anfs, signal=None):
    fs_anf = anfs.fs[0]
    assert np.all(anfs.fs == fs_anf)


    ### Select ANF channels
    trains = []
    for cf in net.cfs:
        trains.append( anfs[anfs.cfs==cf].trains )
    trains = np.array(trains).T.squeeze()


    ### Resample ANF to net.fs
    if signal is not None:
        s = dsp.resample(signal.data, len(signal.data) * net.fs / signal.fs)
        assert net.fs == sgram.fs
        assert net.time_shift == sgram.time_shift
        assert net.freq in sgram.freqs
    anf_mat = dsp.resample(
        trains,
        len(trains) * net.fs / net.time_shift / fs_anf
    )



def main():
    pass

if __name__ == "__main__":
    main()
