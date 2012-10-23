#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"

import numpy as np
import scipy.signal as dsp
from collections import namedtuple

import ffnet
import joblib

from common import (
    run_ear,
    band_pass_filter,
    Reconstructor,
    Signal
)


mem = joblib.Memory("work", verbose=2)



@mem.cache
def _train_tnc(net, fs, win_len, anf, signal, **kwargs):

    input_set, target_set = _make_mlp_sets(
        win_len=win_len,
        fs=fs,
        anf=anf,
        signal=signal
    )

    net.train_tnc(
        input_set,
        target_set,
        messages=1,
        nproc=None,
        **kwargs
    )

    return net





class MlpReconstructor(Reconstructor):
    def __init__(self,
                 band=(80, 2000),
                 fs_mlp=8e3,
                 hidden_layer=4,
                 channel_num=10,
                 anf_type='msr'
             ):

        assert fs_mlp/2 >= band[1], "Nyquist"


        self.fs = fs_mlp
        self.band = band
        self.channel_num = channel_num
        self.anf_type = anf_type

        self.cfs = None



        self.win_len = 10e-3
        win_samp = int(np.round(
            self.win_len*self.fs
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




    def train(self, sound, fs, filter=True, func=None, **kwargs):

        if filter:
            print("Filtering the siganl:", self.band)
            sound = band_pass_filter(sound, fs, self.band)


        s = Signal(sound, fs)


        if isinstance(self.band, tuple):
            cfs = (self.band[0], self.band[1], self.channel_num)
        else:
            cfs = self.band

        anf = run_ear(
            sound=s.data,
            fs=s.fs,
            cfs=cfs,
            anf_type=self.anf_type
        )

        if self.cfs is None:
            self.cfs = anf.cfs
        else:
            assert np.all(self.cfs == anf.cfs)



        assert func is None, "Not implemented"


        self.net = _train_tnc(
            net=self.net,
            fs=self.fs,
            win_len=self.win_len,
            anf=anf,
            signal=s,
            **kwargs
        )



    def run(self, anf, filter=True):

        ### Check anf_type
        assert self.anf_type == anf.type
        assert np.all(self.cfs = anf.cfs)


        ### Run MLP
        input_set, _ = _make_mlp_sets(
            win_len=self.win_len,
            fs=self.fs,
            anf=anf
        )

        output_set = self.net(input_set)
        sound = output_set.squeeze()
        # sound = dsp.resample(sound, len(sound) * self.fs / self._net.fs)

        if filter:
            sound = band_pass_filter(sound, self.fs, self.band)

        return sound, self.fs



def _make_mlp_sets(win_len, fs, anf, signal=None):

    ### Window length in samples
    win_samp = int(np.round(win_len*fs))


    ### Resample data to the desired output fs
    anf_data = dsp.resample(
        anf.data,
        len(anf.data) * fs / anf.fs
    )
    if signal is not None:
        signal_data = dsp.resample(
            signal.data,
            len(signal.data) * fs / signal.fs
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


    mlp_reconstructor = MlpReconstructor(
        band=(80,2000),
        fs_mlp=4e3,
        hidden_layer=0,
        anf_type='msr' #(0,1000,0)
    )


    ### Training
    mlp_reconstructor.train(
        s,
        fs,
        maxfun=300
    )


    ### Testing
    anf = run_ear(
        sound=s,
        fs=fs,
        cfs=mlp_reconstructor.cfs,
        anf_type=mlp_reconstructor.anf_type
    )
    r, fs = mlp_reconstructor.run(
        anf
    )



    fig, ax = plt.subplots(3, 1)
    ax[0].plot(s)
    ax[1].imshow(anf.data.T, aspect='auto')
    ax[2].plot(r)

    plt.show()




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cochlea

    main()
