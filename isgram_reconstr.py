#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
import scipy.signal as dsp
from collections import namedtuple
import multiprocessing

import ffnet
import joblib

import cochlea
import thorns as th

from sgram import calc_sgram, calc_isgram, SGram
from common import band_pass_filter, run_ear

mem = joblib.Memory("tmp", verbose=2)
Net = namedtuple("Net", "net, freq, fs, time_shift, cfs, win_len")


class ISgramReconstructor(object):
    def __init__(self,
                 time_shift=2,
                 hidden_layer=0.25,
                 band=(2000,8000),
                 channel_num=51,
                 cfs_per_channel=[0.8, 1],
                 anf_num=(0,1000,0)
             ):

        self.fs = None
        self.time_shift = time_shift
        self.band = band
        self.channel_num = channel_num
        self.cfs_per_channel = cfs_per_channel
        self.anf_num = anf_num
        self.cfs = None
        self._hidden_layer = hidden_layer

        self._nets = None




    def train(self, sound, fs, iter_num=1000):
        if self.fs is None:
            self.fs = fs
        else:
            assert self.fs == fs

        sound = band_pass_filter(sound, fs, self.band)

        if self._nets is None:
            self._nets = _generate_nets(
                fs=self.fs,
                time_shift=self.time_shift,
                channel_num=self.channel_num,
                band=self.band,
                cfs_per_channel=self.cfs_per_channel,
                hidden_layer=self._hidden_layer,
                win_len_sec=5e-3
            )


        if self.cfs is None:
            cfs = []
            for net in self._nets:
                cfs.extend(net.cfs)
            self.cfs = np.unique(cfs)


        anfs = run_ear(
            sound,
            fs,
            self.cfs,
            self.anf_num
        )

        sgram = calc_sgram(
            sound,
            fs,
            channel_num=self.channel_num,
            time_shift=self.time_shift
        )

        self._nets = _train(
            self._nets,
            anfs,
            sgram,
            iter_num
        )



    def run(self, anfs, iter_num=1000, filter=True):

        ### Check anf_num
        for anf_num in anfs.anf_num:
            assert np.all( anf_num == np.array(self.anf_num) )

        ### Check fs
        fs_net, = np.unique([net.fs for net in self._nets])
        fs_anf, = np.unique(anfs.fs)
        assert fs_net == fs_anf
        fs = fs_net



        freqs = np.linspace(0, fs/2, self.channel_num)
        freqs_net = [net.freq for net in self._nets]



        output_data = []
        for net in self._nets:
            input_data = _make_mlp_data(net, anfs)
            output_data.append( net.net(input_data).squeeze() )

        empty_channel = np.zeros_like( output_data[0] )



        sg = []
        for freq in freqs:
            if freq in freqs_net:
                assert freq == freqs_net[0]
                freqs_net.remove(freq)
                sg.append( output_data.pop(0) )
            else:
                sg.append( empty_channel )

        sg = np.array(sg).T
        sg[ sg<0 ] = 0


        sgram = SGram(
            data=sg,
            fs=fs,
            freqs=freqs,
            time_shift=self.time_shift
        )


        signal = calc_isgram(sgram, iter_num)

        signal = band_pass_filter(signal, fs, self.band)

        return signal, fs








def _training_helper( (net, input_data, target_data, iter_num) ):

    net.net.train_tnc(
        input_data,
        target_data,
        maxfun=iter_num,
        messages=1,
        nproc=1
    )
    return net



@mem.cache
def _generate_nets(fs,
                   time_shift,
                   channel_num,
                   band,
                   cfs_per_channel,
                   hidden_layer,
                   win_len_sec=5e-3):


    win_len = np.round(win_len_sec * fs / time_shift)

    freqs = np.linspace(0, fs/2, channel_num)
    lo, hi = band

    nets = []
    for freq in freqs:
        np.random.seed(int(freq))

        if (freq < lo) or (freq > hi):
            continue

        print "Generating MLP for", freq, "Hz"
        cfs = freq * np.array(cfs_per_channel)


        ### Make MLP
        if hidden_layer > 0:
            conec = ffnet.mlgraph(
                (int(len(cfs)*win_len),
                 int(len(cfs)*win_len*hidden_layer),
                 1)
            )
        else:
            conec = ffnet.mlgraph(
                (int(len(cfs)*win_len), 1)
            )
        net = ffnet.ffnet(conec)


        nets.append(
            Net(
                net=net,
                fs=fs,
                time_shift=time_shift,
                freq=freq,
                cfs=cfs,
                win_len=win_len
            )
        )

    return nets



@mem.cache
def _train(nets, anfs, sgram, iter_num):
    training_data = []
    for net in nets:
        input_data, target_data = _make_mlp_data(net, anfs, sgram)
        training_data.append( (net,input_data,target_data,iter_num) )

    # nets = map(_training_helper, training_data)
    pool = multiprocessing.Pool()
    nets = pool.map(_training_helper, training_data)

    return nets



def _make_mlp_data(net, anfs, sgram=None):

    fs_anf = anfs.fs[0]
    assert np.all(anfs.fs == fs_anf)


    ### Select ANF channels
    trains = []
    for cf in net.cfs:
        trains.append( anfs[anfs.cfs==cf].trains )
    trains = np.array(trains).T.squeeze()


    ### Resample ANF to net.fs
    if sgram is not None:
        assert net.fs == sgram.fs
        assert net.time_shift == sgram.time_shift
        assert net.freq in sgram.freqs
    anf_mat = dsp.resample(
        trains,
        len(trains) * net.fs / net.time_shift / fs_anf
    )


    ### Select spectrogram frquency
    if sgram is not None:
        sgram_target = sgram.data[:, sgram.freqs==net.freq]
        sgram_target = sgram_target.squeeze()


    ### Make MLP data
    input_data = []
    target_data = []

    for i in np.arange(len(anf_mat) - net.win_len):
        lo = i
        hi = i + net.win_len

        input_data.append( anf_mat[lo:hi].flatten() )
        if sgram is not None:
            target_data.append( sgram_target[i] )


    input_data = np.array(input_data, dtype=float)
    target_data = np.array(target_data, dtype=float)


    if sgram is None:
        return input_data
    else:
        return input_data, target_data







def main():

    fs = 16e3
    t = np.arange(0, 0.1, 1/fs)

    s1 = dsp.chirp(t, 1000, t[-1], 6000)
    s1 = cochlea.set_dbspl(s1, 60)
    s1[300:400] = 0

    s2 = dsp.chirp(t, 6000, t[-1], 1000)
    s2 = cochlea.set_dbspl(s2, 50)
    s2[300:400] = 0

    s = np.concatenate( (s1, s2) )



    isgram_reconstructor = ISgramReconstructor(
        time_shift=2,
        hidden_layer=0,
        band=(1000,8000),
        channel_num=51,
        cfs_per_channel=[1, 1.2],
        anf_num=(0,1000,0)
    )


    ### Training
    isgram_reconstructor.train(
        s,
        fs,
        iter_num=500
    )



    ### Testing
    anfs = run_ear(
        sound=s,
        fs=fs,
        cfs=isgram_reconstructor.cfs,
        anf_num=isgram_reconstructor.anf_num
    )


    r, fs = isgram_reconstructor.run(
        anfs,
        iter_num=100
    )


    fig, ax = plt.subplots(nrows=3, ncols=1)
    # ax[0].specgram(s, Fs=fs)
    # ax[2].specgram(r, Fs=fs)
    # ax[0].imshow(calc_sgram(s, fs, 51, 1).data.T, aspect='auto')
    # ax[2].imshow(calc_sgram(r, fs, 51, 1).data.T, aspect='auto')
    ax[0].plot(s)
    ax[2].plot(r)
    ax[1].imshow(anfs.trains, aspect='auto')


    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
