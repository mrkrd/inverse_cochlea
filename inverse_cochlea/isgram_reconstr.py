#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
import scipy.signal as dsp
from collections import namedtuple
import multiprocessing
import logging

import ffnet

from sgram import calc_sgram, calc_isgram, Spectrogram
from common import run_ear

import thorns.waves as wv

import joblib
mem = joblib.Memory("work", verbose=2)

Net = namedtuple("Net", "net, freq, fs, sgram_shift, cfs, win_len")


class ISgramReconstructor(object):
    def __init__(
            self,
            sgram_shift=2,
            hidden_layer=0.25,
            band=(2000,8000),
            channel_num=51,
            relative_cfs_per_channel=[1, 1.2],
            anf_type='msr'
    ):

        self.fs = None          # fs of the training and output signal
        self.sgram_shift = sgram_shift
        self.band = band
        self.channel_num = channel_num
        self.relative_cfs_per_channel = relative_cfs_per_channel
        self.anf_type = anf_type
        self.cfs = None
        self._hidden_layer = hidden_layer

        self._nets = None




    def train(self, sound, fs, filter=True, **kwargs):
        if self.fs is None:
            self.fs = fs
        else:
            assert self.fs == fs

        if filter:
            logging.info("Filtering the siganl: {}".format(str(self.band)))
            sound = wv.fft_filter(sound, fs, self.band)


        if self._nets is None:
            self._nets = _generate_nets(
                fs=self.fs,
                sgram_shift=self.sgram_shift,
                channel_num=self.channel_num,
                band=self.band,
                relative_cfs_per_channel=self.relative_cfs_per_channel,
                hidden_layer=self._hidden_layer,
                win_len=5e-3
            )


        if self.cfs is None:
            cfs = []
            for net in self._nets:
                cfs.extend(net.cfs)
            self.cfs = np.unique(cfs)


        anfs = run_ear(
            sound=sound,
            fs=fs,
            cf=self.cfs,
            anf_type=self.anf_type
        )

        sgram = calc_sgram(
            sound,
            fs,
            channel_num=self.channel_num,
            sgram_shift=self.sgram_shift
        )

        self._nets = _train(
            self._nets,
            anfs,
            sgram,
            **kwargs
        )



    def run(self, anfs, iter_num=1000, filter=True, store_sgram=False):

        assert anfs.type == self.anf_type

        fs, = np.unique([net.fs for net in self._nets])


        freqs = np.linspace(0, fs/2, self.channel_num)
        freqs_net = [net.freq for net in self._nets]



        output_data = []
        for net in self._nets:
            input_data = _make_mlp_sets(net, anfs)
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


        sgram = Spectrogram(
            data=sg,
            fs=fs,
            freqs=freqs,
            sgram_shift=self.sgram_shift
        )

        if store_sgram:
            self.sgram = sgram


        signal = calc_isgram(sgram, iter_num)

        if filter:
            logging.info("Filtering the siganl: {}".format(str(self.band)))
            sound = wv.fft_filter(signal, fs, self.band)


        return signal, fs





def _train_signle_channel( (net, input_data, target_data, kwargs) ):

    net.net.train_tnc(
        input_data,
        target_data,
        messages=1,
        nproc=1,
        **kwargs
    )
    return net





def _generate_nets(
        fs,
        sgram_shift,
        channel_num,
        band,
        relative_cfs_per_channel,
        hidden_layer,
        win_len=5e-3
):

    win_samp = np.round(win_len * fs / sgram_shift)

    freqs = np.linspace(0, fs/2, channel_num)
    lo, hi = band

    nets = []
    for freq in freqs:
        np.random.seed(int(freq)) # we want deterministic initial MLP weights

        if (freq < lo) or (freq > hi):
            continue

        logging.info("Generating MLP for {} Hz".format(freq))
        cfs = freq * np.array(relative_cfs_per_channel)


        ### Make MLP
        if hidden_layer > 1:
            conec = ffnet.mlgraph(
                (int(len(cfs)*win_samp),
                 int(hidden_layer),
                 1)
            )

        elif hidden_layer > 0:
            conec = ffnet.mlgraph(
                (int(len(cfs)*win_samp),
                 int(len(cfs)*win_samp*hidden_layer),
                 1)
            )
        elif hidden_layer == 0:
            conec = ffnet.mlgraph(
                (int(len(cfs)*win_samp), 1)
            )
        else:
            raise RuntimeError

        net = ffnet.ffnet(conec)


        nets.append(
            Net(
                net=net,
                fs=fs,
                sgram_shift=sgram_shift,
                freq=freq,
                cfs=cfs,
                win_len=win_len
            )
        )

    return nets



@mem.cache
def _train(nets, anfs, sgram, **kwargs):
    training_data = []
    for net in nets:
        input_data, target_data = _make_mlp_sets(net, anfs, sgram)
        training_data.append( (net,input_data,target_data,kwargs) )

    # nets = map(_train_signle_channel, training_data)
    pool = multiprocessing.Pool()
    nets = pool.map(_train_signle_channel, training_data)

    return nets



def _make_mlp_sets(net, anf, sgram=None):

    win_samp = np.round(net.win_len * net.fs / net.sgram_shift)


    ### Select ANF channels
    anf_sel = []
    for cf in net.cfs:
        anf_sel.append( anf.data[:,anf.cfs==cf] )
    anf_sel = np.array(anf_sel).T.squeeze()


    ### Resample ANF to fs/sgram_shift
    if sgram is not None:
        assert net.fs == sgram.fs
        assert net.sgram_shift == sgram.sgram_shift
        assert net.freq in sgram.freqs
    anf_mat = dsp.resample(
        anf_sel,
        len(anf_sel) * net.fs / net.sgram_shift / anf.fs
    )


    ### Select spectrogram frquency
    if sgram is not None:
        sgram_target = sgram.data[:, sgram.freqs==net.freq]
        # sgram_target = sgram_target.squeeze()


    ### Make MLP data
    input_set = []
    target_set = []

    for i in np.arange(len(anf_mat) - win_samp + 1):
        lo = i
        hi = i + win_samp

        input_set.append( anf_mat[lo:hi].flatten() )
        if sgram is not None:
            target_set.append( sgram_target[i] )


    input_set = np.array(input_set, dtype=float)
    target_set = np.array(target_set, dtype=float)


    if sgram is None:
        return input_set
    else:
        return input_set, target_set







def main():

    fs = 16e3
    t = np.arange(0, 0.1, 1/fs)

    s0 = np.zeros(0.1*fs)

    s1 = dsp.chirp(t, 1000, t[-1], 6000)
    s1 = cochlea.set_dbspl(s1, 60)
    s1[300:400] = 0

    s2 = dsp.chirp(t, 6000, t[-1], 1000)
    s2 = cochlea.set_dbspl(s2, 50)
    s2[300:400] = 0

    s = np.concatenate( (s1, s0, s2) )



    isgram_reconstructor = ISgramReconstructor(
        sgram_shift=2,
        hidden_layer=0,
        band=(1000,8000),
        channel_num=51,
        relative_cfs_per_channel=[0.9, 1, 1.2],
        anf_type='msr'
    )


    ### Training
    isgram_reconstructor.train(
        s,
        fs,
        maxfun=500
    )



    ### Testing
    anf = run_ear(
        sound=s,
        fs=fs,
        cf=isgram_reconstructor.cfs,
        anf_type=isgram_reconstructor.anf_type
    )


    r, fs = isgram_reconstructor.run(
        anf,
        iter_num=100
    )


    fig, ax = plt.subplots(nrows=3, ncols=1)
    # ax[0].specgram(s, Fs=fs)
    # ax[2].specgram(r, Fs=fs)
    # ax[0].imshow(calc_sgram(s, fs, 51, 1).data.T, aspect='auto')
    # ax[2].imshow(calc_sgram(r, fs, 51, 1).data.T, aspect='auto')
    ax[0].plot(s)
    ax[2].plot(r)
    ax[1].imshow(anf.data.T, aspect='auto')

    plt.show()




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cochlea

    main()
