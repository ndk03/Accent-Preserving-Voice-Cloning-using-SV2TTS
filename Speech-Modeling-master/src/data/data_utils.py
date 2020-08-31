import sys
import os
#nopep8
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

from yaml import safe_load
import pandas as pd
from collections import Counter
import librosa
from librosa.display import specshow
import numpy as np
import webrtcvad
import soundfile
from scipy.ndimage import binary_dilation
import pathlib
from colorama import Fore
from tqdm import tqdm
import warnings
import h5py
import matplotlib.pyplot as plt
from torch.utils import data
import torch
from random import randint, sample
from yaml import safe_load
import matplotlib

warnings.filterwarnings("ignore")

with open('src/config.yaml', 'r') as f:
    config = safe_load(f.read())

RAW_DATA_DIR = config['RAW_DATA_DIR']
PROC_DATA_DIR = config['PROC_DATA_DIR']
INTERIM_DATA_DIR = config['INTERIM_DATA_DIR']
MODEL_SAVE_DIR = config['MODEL_SAVE_DIR']
GMU_DATA_PACKED = config['GMU_DATA_PACKED']
GMU_DATA_INFO = config['GMU_DATA_INFO']
GMU_DATA = config['GMU_DATA']
GMU_ACCENT_COUNT = config['GMU_ACCENT_COUNT']
SAMPLING_RATE = config['SAMPLING_RATE']
WINDOW_SIZE = config['WINDOW_SIZE']
WINDOW_STEP = config['WINDOW_STEP']
N_FFT = int(WINDOW_SIZE * SAMPLING_RATE / 1000)
H_L = int(WINDOW_STEP * SAMPLING_RATE / 1000)
MEL_CHANNELS = config['MEL_CHANNELS']
SMOOTHING_LENGTH = config['SMOOTHING_LENGTH']
SMOOTHING_WSIZE = config['SMOOTHING_WSIZE']
DBFS = config['DBFS']
SMOOTHING_WSIZE = int(SMOOTHING_WSIZE * SAMPLING_RATE / 1000)

dirs_ = set([globals()[d] for d in globals() if d.__contains__('DIR')] +
            [config[d] for d in config if d.__contains__('DIR')])

VAD = webrtcvad.Vad(mode=config['VAD_MODE'])


def structure(dirs=[]):
    """
    Summary:
    
    Args:
    
    Returns:
    
    """
    dirs_reqd = set(list(dirs_) + list(dirs))
    for data_dir in dirs_reqd:
        if not pathlib.Path.exists(pathlib.Path(data_dir)):
            os.makedirs(data_dir)


def normalization(aud, norm_type='peak'):
    """
    Summary:
    
    Args:
    
    Returns:
    
    """
    try:
        assert len(aud) > 0
        if norm_type is 'peak':
            aud = aud / np.max(aud)

        elif norm_type is 'rms':
            dbfs_diff = DBFS - (20 *
                                np.log10(np.sqrt(np.mean(np.square(aud)))))
            if DBFS > 0:
                aud = aud * np.power(10, dbfs_diff / 20)

        return aud
    except AssertionError as e:
        raise AssertionError("Empty audio sig")


def preprocess(fname):
    """
    Summary:
    
    Args:
    
    Returns:
    
    """
    aud, sr = librosa.load(fname, sr=None)
    aud = librosa.resample(aud, sr, SAMPLING_RATE)
    trim_len = len(aud) % SMOOTHING_WSIZE
    aud = np.append(aud, np.zeros(SMOOTHING_WSIZE - trim_len))

    assert len(aud) % SMOOTHING_WSIZE == 0, print(len(aud) % trim_len, aud)

    pcm_16 = np.round(
        (np.iinfo(np.int16).max * aud)).astype(np.int16).tobytes()
    voices = [
        VAD.is_speech(pcm_16[2 * ix:2 * (ix + SMOOTHING_WSIZE)],
                      sample_rate=SAMPLING_RATE)
        for ix in range(0, len(aud), SMOOTHING_WSIZE)
    ]
    smoothing_mask = np.repeat(
        binary_dilation(voices, np.ones(SMOOTHING_LENGTH)), SMOOTHING_WSIZE)
    aud = aud[smoothing_mask]
    try:
        aud = normalization(aud, norm_type='peak')
        return aud

    except AssertionError as e:
        raise AssertionError("Empty audio sig")
        # exit()


def mel_spectogram(aud):
    """
    Summary:
    
    Args:
    
    Returns:
    
    """
    mel = librosa.feature.melspectrogram(aud,
                                         sr=SAMPLING_RATE,
                                         n_fft=N_FFT,
                                         hop_length=H_L,
                                         n_mels=MEL_CHANNELS)
    mel = np.log(mel + 1e-5)
    return mel


class HDF5TorchDataset(data.Dataset):
    def __init__(self, accent_data, device=torch.device('cpu')):
        hdf5_file = os.path.join(PROC_DATA_DIR, '{}.hdf5'.format(accent_data))
        self.hdf5_file = h5py.File(hdf5_file, 'r')
        self.accents = self.hdf5_file.keys()
        with open('src/config.yaml', 'r') as f:
            self.config = safe_load(f.read())
        self.device = device

    def __len__(self):
        return len(self.accents) + int(1e4)

    def _get_acc_uttrs(self):
        while True:
            rand_accent = sample(list(self.accents), 1)[0]
            if self.hdf5_file[rand_accent].__len__() > 0:
                break
        wavs = list(self.hdf5_file[rand_accent])
        while len(wavs) < self.config['UTTR_COUNT']:
            wavs.extend(sample(wavs, 1))

        rand_wavs = sample(wavs, self.config['UTTR_COUNT'])
        rand_accent_ix = list(self.accents).index(rand_accent)
        rand_uttrs = []
        labels = []
        for wav in rand_wavs:
            wav_ = self.hdf5_file[rand_accent][wav]
            
            rix = randint(0, wav_.shape[1] - self.config['SLIDING_WIN_SIZE'])

            ruttr = wav_[:, rix:rix + self.config['SLIDING_WIN_SIZE']]
            
            ruttr = torch.Tensor(ruttr)
            rand_uttrs.append(ruttr)
            labels.append(rand_accent_ix)
        return rand_uttrs, labels

    def __getitem__(self, ix=0):
        rand_uttrs, labels = self._get_acc_uttrs()
        rand_uttrs = torch.stack(rand_uttrs).to(device=self.device)
        labels = torch.LongTensor(labels).to(device=self.device)
        return rand_uttrs, labels

    def collate(self, data):
        pass


def write_hdf5(out_file, data):
    """
    Summary:
    
    Args:
    
    Returns:
    
    """
    gmu_proc_file = h5py.File(out_file, 'w')
    for g in data:
        group = gmu_proc_file.create_group(g)
        for datum in data[g]:
            group.create_dataset("mel_spects_{}".format(datum[0]),
                                 data=datum[1])
    gmu_proc_file.close()


def heatmap(data,
            row_labels,
            col_labels,
            ax=None,
            cbar_kw={},
            cbarlabel="",
            **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=-30,
             ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im,
                     data=None,
                     valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None,
                     **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


if __name__ == "__main__":
    structure()
    hdf5d = HDF5TorchDataset('gmu_4_train')
    loader = data.DataLoader(hdf5d, 4)
    for y in loader:
        print(y.shape)
        break