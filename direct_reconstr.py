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

from common import run_ear, band_pass_filter, Reconstructor

Net = namedtuple("Net", "net, fs, cfs, win_len")
Signal = namedtuple("Signal", "data, fs")

mem = joblib.Memory("tmp", verbose=0)



@mem.cache
def _train_tnc(net, anfs, signal, **kwargs):
    input_data, target_data = _make_mlp_data(
        net,
        anfs,
        signal
    )

    net.net.train_tnc(
        input_data,
        target_data,
        # maxfun=iter_num,
        messages=1,
        nproc=None,
        **kwargs
    )

    return net




class DirectReconstructor(Reconstructor):
    def __init__(self,
                 band=(80, 2000),
                 fs_mlp=8e3,
                 hidden_layer=4,
                 channel_num=10,
                 anf_num=(0,1000,0)
             ):

        self.fs = None
        self.band = band
        self.fs_mlp = fs_mlp
        self.channel_num = channel_num
        self.anf_num = anf_num
        self._hidden_layer = hidden_layer

        self.cfs = None
        self._net = None



    def train(self, sound, fs, filter=True, func=_train_tnc, **kwargs):
        if self.fs is None:
            self.fs = fs
        else:
            assert self.fs == fs

        if filter and isinstance(self.band, tuple):
            print "Filtering the siganl:", self.band
            sound = band_pass_filter(sound, fs, self.band)


        s = Signal(sound, fs)

        if isinstance(self.band, tuple):
            cfs = (self.band[0], self.band[1], self.channel_num)
        else:
            cfs = self.band

        anfs = run_ear(
            sound=s.data,
            fs=s.fs,
            cfs=cfs,
            anf_num=self.anf_num
        )


        if self.cfs is None:
            self.cfs = np.unique(anfs['cfs'])




        if self._net is None:
            win_len = int(np.round(10e-3 * self.fs_mlp))

            if self._hidden_layer > 1:
                conec = ffnet.mlgraph(
                    (win_len*self.channel_num,
                     int(self._hidden_layer),
                     1)
                )

            elif self._hidden_layer > 0:
                conec = ffnet.mlgraph(
                    (win_len*self.channel_num,
                     int(win_len*self.channel_num*self._hidden_layer),
                     1)
                )

            elif self._hidden_layer == 0:
                conec = ffnet.mlgraph(
                    (win_len*self.channel_num,
                     1)
                )

            else:
                assert False, "hidden_layer should not be negative"


            np.random.seed(0)
            net = ffnet.ffnet(conec)

            self._net = Net(
                net=net,
                fs=self.fs_mlp,
                cfs=self.cfs,
                win_len=win_len
            )



        self._net = func(
            net=self._net,
            anfs=anfs,
            signal=s,
            **kwargs
        )





    def run(self, anfs, filter=True):
        ### Check anf_num
        for anf_num in anfs['anf_num']:
            assert np.all( anf_num == np.array(self.anf_num) )


        ### Run MLP
        input_data = _make_mlp_data(
            self._net,
            anfs
        )

        output_data = self._net.net(input_data)
        sound = output_data.squeeze()
        sound = dsp.resample(sound, len(sound) * self.fs / self._net.fs)

        if filter and isinstance(self.band, tuple):
            sound = band_pass_filter(sound, self.fs, self.band)

        return sound, self.fs



def _make_mlp_data(net, anfs, signal=None):
    fs_anf = anfs['fs'][0]
    assert np.all(anfs['fs'] == fs_anf)


    ### Select ANF channels
    trains = []
    for cf in net.cfs:
        trains.append( anfs[anfs['cfs']==cf]['trains'] )
    trains = np.array(trains).T.squeeze()

    ### Resample ANF to net.fs
    anf_mat = dsp.resample(
        trains,
        len(trains) * net.fs / fs_anf
    )
    if signal is not None:
        sig = dsp.resample(signal.data, len(anf_mat))



    input_data = []
    target_data = []
    for i in np.arange(len(anf_mat) - net.win_len):
        lo = i
        hi = i + net.win_len

        input_data.append( anf_mat[lo:hi].flatten() )
        if signal is not None:
            target_data.append( [sig[lo]] )

    input_data = np.array(input_data, dtype=float)
    target_data = np.array(target_data, dtype=float)


    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2,1)
    # ax[0].imshow(input_data.T, aspect='auto')
    # if signal is not None:
    #     ax[1].plot(target_data)
    # plt.show()

    if signal is None:
        return input_data
    else:
        return input_data, target_data



def main():
    fs = 16e3
    t = np.arange(0, 0.1, 1/fs)

    s1 = dsp.chirp(t, 50, t[-1], 2000)
    s1 = cochlea.set_dbspl(s1, 60)
    s1[300:400] = 0

    s2 = dsp.chirp(t, 2000, t[-1], 50)
    s2 = cochlea.set_dbspl(s2, 50)
    s2[300:400] = 0

    s = np.concatenate( (s1, s2) )


    direct_reconstructor = DirectReconstructor(
        band=(80,2000),
        fs_mlp=4e3,
        hidden_layer=4
    )


    ### Training
    direct_reconstructor.train(
        s,
        fs,
        maxfun=300
    )


    ### Testing
    anfs = run_ear(
        sound=s,
        fs=fs,
        cfs=direct_reconstructor.cfs,
        anf_num=direct_reconstructor.anf_num
    )
    r, fs = direct_reconstructor.run(
        anfs
    )



    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(s)
    ax[2].plot(r)
    ax[1].imshow(anfs['trains'], aspect='auto')

    plt.show()




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
