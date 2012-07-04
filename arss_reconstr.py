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
import arss

from common import band_pass_filter, run_ear

mem = joblib.Memory("tmp", verbose=2)
Net = namedtuple("Net", "net, freq, fs, fs_sgram, cfs, win_len")
SGram = namedtuple("SGram", "data, fs, freqs, fs_sgram")

class ArssReconstructor(object):
    def __init__(self,
                 fs_sgram=8e3,
                 hidden_layer=0.25,
                 band=(2000,8000),
                 channel_num=50,
                 cfs_per_channel=[1, 1.2],
                 anf_num=(0,1000,0)
             ):

        self.fs = None
        self.fs_sgram = fs_sgram
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


        sg, freqs = arss.analyze(
            signal=sound,
            fs=fs,
            fs_sgram=self.fs_sgram,
            band=self.band,
            channel_num=self.channel_num
        )
        sgram = SGram(
            data=sg,
            fs=fs,
            freqs=freqs,
            fs_sgram=self.fs_sgram
        )

        plt.imshow(sg, aspect='auto')
        plt.plot()


        if self._nets is None:
            self._nets = _generate_nets(
                fs=self.fs,
                fs_sgram=self.fs_sgram,
                freqs=freqs,
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

        self._nets = _train(
            self._nets,
            anfs,
            sgram,
            iter_num
        )



    def run(self, anfs, iter_num=1000, filter=True, store_sgram=False):

        ### Check anf_num
        for anf_num in anfs['anf_num']:
            assert np.all( anf_num == np.array(self.anf_num) )

        ### Check fs
        fs_net, = np.unique([net.fs for net in self._nets])
        fs_anf, = np.unique(anfs['fs'])
        assert fs_net == fs_anf
        fs = fs_net



        freqs = [net.freq for net in self._nets]



        output_data = []
        for net in self._nets:
            input_data = _make_mlp_data(net, anfs)
            output_data.append( net.net(input_data).squeeze() )



        sg = np.array(output_data).T
        sg[ sg<0 ] = 0


        if store_sgram:
            sgram = SGram(
                data=sg,
                fs=fs,
                freqs=freqs,
                fs_sgram=self.fs_sgram
            )
            self.sgram = sgram


        signal = arss.synthesize_sine(
            sg,
            self.fs_sgram,
            self.fs,
            self.band
        )


        if filter:
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
                   fs_sgram,
                   freqs,
                   cfs_per_channel,
                   hidden_layer,
                   win_len_sec=5e-3):


    win_len = np.round(win_len_sec * fs_sgram)


    nets = []
    for freq in freqs:
        np.random.seed(int(freq))


        print "Generating MLP for", freq, "Hz"
        cfs = freq * np.array(cfs_per_channel)


        ### Make MLP
        if hidden_layer > 1:
            conec = ffnet.mlgraph(
                (int(len(cfs)*win_len),
                 int(hidden_layer),
                 1)
            )

        elif hidden_layer > 0:
            conec = ffnet.mlgraph(
                (int(len(cfs)*win_len),
                 int(len(cfs)*win_len*hidden_layer),
                 1)
            )
        elif hidden_layer == 0:
            conec = ffnet.mlgraph(
                (int(len(cfs)*win_len), 1)
            )
        else:
            assert False

        net = ffnet.ffnet(conec)


        nets.append(
            Net(
                net=net,
                fs=fs,
                fs_sgram=fs_sgram,
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

        # print input_data[0], len(input_data[0]), input_data.shape
        # print target_data[0]

        # fig, ax = plt.subplots(2,1)
        # ax[0].imshow(input_data, aspect='auto')
        # ax[1].plot(target_data)
        # plt.show()

        training_data.append( (net,input_data,target_data,iter_num) )

    # nets = map(_training_helper, training_data)
    pool = multiprocessing.Pool()
    nets = pool.map(_training_helper, training_data)

    return nets



def _make_mlp_data(net, anfs, sgram=None):

    fs_anf = anfs['fs'][0]
    assert np.all(anfs['fs'] == fs_anf)


    ### Select ANF channels
    trains = []
    for cf in np.unique(net.cfs):
        trains.append( anfs[anfs['cfs']==cf]['trains'] )
    trains = np.array(trains).T.squeeze()


    ### Resample ANF to net.fs
    if sgram is not None:
        assert net.fs == sgram.fs
        assert net.fs_sgram == sgram.fs_sgram
        assert net.freq in sgram.freqs
    anf_mat = dsp.resample(
        trains,
        len(trains) * net.fs_sgram / fs_anf
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
            target_data.append( [sgram_target[i]] )



    # fig, ax = plt.subplots(3,1)
    # print trains
    # print net.cfs
    # ax[0].imshow(anf_mat.T, aspect='auto')
    # ax[1].imshow(np.array(input_data).T, aspect='auto')
    # ax[2].plot(target_data)
    # plt.show()

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


    s = np.sin(2 * np.pi * t * 1000)
    s = dsp.chirp(t, 6000, t[-1], 6500)
    s = cochlea.set_dbspl(s, 60)


    arss_reconstructor = ArssReconstructor(
        fs_sgram=8000,
        hidden_layer=4,
        band=(3000,7000),
        channel_num=70,
        cfs_per_channel=[0.5, 1, 1.5],
        anf_num=(0,1000,0)
    )


    ### Training
    arss_reconstructor.train(
        s,
        fs,
        iter_num=500
    )



    ### Testing
    anfs = run_ear(
        sound=s,
        fs=fs,
        cfs=arss_reconstructor.cfs,
        anf_num=arss_reconstructor.anf_num
    )


    r, fs = arss_reconstructor.run(
        anfs,
        iter_num=100,
        store_sgram=True
    )

    plt.imshow(arss_reconstructor.sgram.data, aspect='auto')
    plt.show()


    fig, ax = plt.subplots(nrows=3, ncols=1)
    # ax[0].specgram(s, Fs=fs)
    # ax[2].specgram(r, Fs=fs)
    # ax[0].imshow(calc_sgram(s, fs, 51, 1).data.T, aspect='auto')
    # ax[2].imshow(calc_sgram(r, fs, 51, 1).data.T, aspect='auto')
    ax[0].plot(s)
    ax[2].plot(r)
    ax[1].imshow(anfs['trains'], aspect='auto')


    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
