import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_load
import os
import sys

# from src.data.data_utils import ImbalancedDatasetSampler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

from src.data.data_utils import HDF5TorchDataset
import matplotlib.pyplot as plt
import librosa
from src.data.data_utils import mel_spectogram, preprocess
np.random.seed(42)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import argparse
import torchvision
import soundfile

from src.data.data_utils import heatmap, annotate_heatmap
# from librosa.feature.inverse import mel_to_audio

class ACCENT_ENCODER_AE(nn.Module):
    def __init__(self,
                 input_shape=(160, 40),
                 load_model=False,
                 epoch=0,
                 dataset_train='gmu_4_train',
                 dataset_val='gmu_4_val',
                 device=torch.device('cpu'),
                 loss_=None):
        super(ACCENT_ENCODER_AE, self).__init__()
        self.input_shape = input_shape
        with open('src/config.yaml', 'r') as f:
            self.config_yml = safe_load(f.read())

        model_save_dir = os.path.join(self.config_yml['MODEL_SAVE_DIR'],
                                      dataset_train)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        self.model_save_string = os.path.join(
            model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')

        log_dir = os.path.join(
            self.config_yml['RUNS_DIR'],
            '{}_{}'.format(dataset_train, self.__class__.__name__))
        self.writer = SummaryWriter(log_dir=os.path.join(
            log_dir, "run_{}".format(
                len(os.listdir(log_dir)) if os.path.exists(log_dir) else 0)))

        self.device = device
        self.dataset_train = HDF5TorchDataset(dataset_train, device=device)
        self.dataset_val = HDF5TorchDataset(dataset_val, device=device)

        self.encoder = nn.Sequential(
            nn.Conv1d(self.config_yml['MEL_CHANNELS'], 32, 4, 2, 1),
            nn.ReLU(True), nn.Conv1d(32, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv1d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv1d(64, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv1d(64, 5, 4, 1), nn.ReLU(True)#, nn.Linear(7, 1)
            )

        self.decoder = nn.Sequential(
            # nn.Linear(1, 7),
            nn.ConvTranspose1d(5, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, self.config_yml['MEL_CHANNELS'], 4, 2, 1),
        )

        self.linear1 = nn.Linear(self.config_yml['HIDDEN_SIZE'],
                                 self.config_yml['EMBEDDING_SIZE'])
        # self.linear2 = nn.Linear(1, 1)
        # torch.nn.init.constant_(self.linear2.weight, 10.0)
        # torch.nn.init.constant_(self.linear2.bias, -5.0)
        self.W = torch.nn.Parameter(torch.Tensor([10.0]), requires_grad=True)
        self.b = torch.nn.Parameter(torch.Tensor([-5.0]), requires_grad=True)

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.load_model = load_model
        self.epoch = epoch
        self.loss_ = loss_
        self.opt = None
        self.mce_loss = nn.MSELoss(reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, frames):

        embs = self.encoder(frames)
        x = self.decoder(embs)

        return embs, x

    def loss_fn(self, embeds, data, recon, labels):
        rcl = self.reconstruction_loss(data, recon)
        dcl = 10*self.direct_classification_loss(labels, embeds)

        return rcl, dcl

    def reconstruction_loss(self, x, recon_x):
        return self.mce_loss(recon_x, x)
        # return self.bce_loss(recon_x,x)

    def direct_classification_loss(self, labels, embeds):
        labels = labels.reshape(-1, 1)
        return self.ce_loss(embeds, labels)

    def train_loop(self,
                   opt,
                   lr_scheduler,
                   loss_,
                   batch_size=1,
                   gaze_pred=None,
                   cpt=0):

        train_iterator = torch.utils.data.DataLoader(self.dataset_train,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     drop_last=True)

        self.val_iterator = torch.utils.data.DataLoader(self.dataset_val,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        drop_last=True)

        if self.load_model:
            self.load_model_cpt(cpt=cpt)

        for epoch in range(self.epoch, 20000):
            for i, (data, labels) in enumerate(train_iterator):

                data = data.view(
                    -1,
                    self.config_yml['MEL_CHANNELS'],
                    self.config_yml['SLIDING_WIN_SIZE'],
                )
                opt.zero_grad()
                embeds, recon = self.forward(data)

                rcl, dcl = self.loss_fn(embeds, data, recon, labels)
                self.loss = rcl + dcl
                self.loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                opt.step()
                self.writer.add_scalar('Loss', self.loss.data.item(), epoch)
                self.writer.add_scalar('rcl', rcl.data.item(), epoch)
                self.writer.add_scalar('dcl', dcl.data.item(), epoch)
                self.writer.add_scalar('ValLoss', self.val_loss(), epoch)
                # self.writer.add_scalar('EER', self.eer(sim_matrix), epoch)
                # if i % 1000 == 0:
                #     self.save_recon(recon, data, epoch)

            if epoch % 1 == 0:
                # self.writer.add_scalar('Loss', loss.data.item(), epoch)
                # self.writer.add_scalar('Val Loss', self.val_loss(), epoch)
                # self.writer.add_scalar('EER', self.eer(sim_matrix), epoch)
                # self.writer.add_histogram('sim', sim_matrix, epoch)

                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': self.loss,
                    }, self.model_save_string.format(epoch))

    # def save_recon(self, recon, data, epoch):
    #     self.griffin_lim_aud(recon[-1].cpu().data.numpy())
        

    # def griffin_lim_aud(self, spec):
    #     y = mel_to_audio(
    #         spec,
    #         sr=self.config_yml['SAMPLING_RATE'],
    #         n_fft=1024,
    #         hop_length=256,
    #         win_length=1024)

    #     soundfile.write(os.path.join(self.config_yml['VIS_PREDS_DIR'],
    #                                  '{}.wav'.format(self.epoch)),
    #                     y,
    #                     samplerate=self.config_yml['SAMPLING_RATE'])
    #     return y

    def embed(self, aud):

        mel = mel_spectogram(aud)  #preprocessed audio
        part_frames = []
        for ix in range(
                0, mel.shape[1] - self.config_yml['SLIDING_WIN_SIZE'],
                int(self.config_yml['SLIDING_WIN_SIZE'] *
                    self.config_yml['SLIDING_WIN_OVERLAP'])):
            part_frame = mel[:, ix:ix + self.config_yml['SLIDING_WIN_SIZE']]
            part_frames.append(part_frame)

        frames = np.stack(part_frames)
        frames = torch.Tensor(frames).view(
            -1,
            self.config_yml['MEL_CHANNELS'],
            self.config_yml['SLIDING_WIN_SIZE'],

        ).to(device=self.device)

        model_pickle = torch.load(self.model_save_string.format(self.epoch),
                                  map_location=self.device)
        self.load_state_dict(model_pickle['model_state_dict'])
        with torch.no_grad():
            self.eval()
            embeds,_ = self.forward(frames)  #.cpu().data.numpy()
            embeds = embeds * torch.reciprocal(
                torch.norm(embeds, dim=1, keepdim=True))
            embeds = torch.mean(embeds, dim=0)
            embeds = embeds.cpu().data.numpy()
            # plt.imshow(embeds)
            # plt.show()
        return embeds

    def accuracy(self):
        acc = 0
        ix = 0
        for i, data in enumerate(self.val_data):
            uttrs = data[0]
            embeds = self.forward(uttrs)
            #TODO
        return (acc / ix)

    def eer(self, sim_matrix=None):
        with torch.no_grad():

            targets = F.one_hot(
                torch.arange(0, self.config_yml['ACC_COUNT']),
                num_classes=self.config_yml['ACC_COUNT']).repeat_interleave(
                    self.config_yml['UTTR_COUNT'], 1).long().T

            fpr, tpr, thresholds = roc_curve(
                targets.flatten(),
                sim_matrix.detach().flatten().cpu().numpy())

            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            # thresh = interp1d(fpr, thresholds)(eer)

        return eer

    def val_loss(self):
        with torch.no_grad():
            val_loss = []

            for ix, (datum, labels) in enumerate(self.val_iterator):
                datum = datum.view(
                                -1,
                                self.config_yml['MEL_CHANNELS'],
                                self.config_yml['SLIDING_WIN_SIZE'],
                            )
                embeds, recon = self.forward(datum)

                rcl, dcl = self.loss_fn(embeds, datum, recon, labels)
                loss = rcl + dcl

                val_loss.append(loss)

                if ix == self.config_yml['VAL_LOSS_COUNT']:
                    break

        return torch.mean(torch.stack(val_loss)).data.item()

    def load_model_cpt(self, cpt=0, opt=None, device=torch.device('cuda')):
        self.epoch = cpt

        model_pickle = torch.load(self.model_save_string.format(self.epoch),
                                  map_location=device)
        self.load_state_dict(model_pickle['model_state_dict'])
        self.opt.load_state_dict(model_pickle['optimizer_state_dict'])
        self.epoch = model_pickle['epoch']
        self.loss = model_pickle['loss']
        print("Loaded Model at epoch {},with loss {}".format(
            self.epoch, self.loss))

    def infer(self, fname, cpt):
        aud = preprocess(fname)
        embeds = encoder.embed(aud)
        return embeds

    def similarity(self, embed1, embed2):
        sim = torch.cosine_similarity(
            torch.tensor(embed1).unsqueeze(0),
            torch.tensor(embed2).unsqueeze(0)).data.item()
        # sim = embed1 @ embed2
        return sim

    def sim_matrix_infer(self, fnames, cpt):
        sim_matrix = np.zeros((len(fnames), len(fnames)))
        for i, f1 in enumerate(fnames):
            for j, f2 in enumerate(fnames):
                # if i==j:
                #     sim_matrix[i, j] = 1
                sim_matrix[i, j] = self.similarity(self.infer(f1, cpt),
                                                   self.infer(f2, cpt))
            # for i, f1 in enumerate(fnames):
            #     for j, f2 in enumerate(fnames):
            #         sim_matrix[i, j] = max(sim_matrix[i, j], sim_matrix[j, i])

        return sim_matrix

    def vis_sim_matrix(self, sim_matrix, labels):
        fig, ax = plt.subplots()
        labels = [l.split('/')[-1].split('.')[0].split('_')[0] for l in labels]

        im, cbar = heatmap(sim_matrix,
                           labels,
                           labels,
                           ax=ax,
                           cmap="inferno",
                           cbarlabel="Similarity")
        # texts = annotate_heatmap(im, valfmt="{x:.1f} t")

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",
                        help="cpu or cuda",
                        default='cuda',
                        choices=['cpu', 'cuda'])
    parser.add_argument("--dataset_train",
                        help="path to train_dataset",
                        required=True)
    parser.add_argument("--dataset_val",
                        help="path to val_dataset",
                        required=True)
    parser.add_argument("--mode",
                        help="train or eval",
                        required=True,
                        choices=['train', 'eval'])
    parser.add_argument(
        "--filedir",
        help=
        "dir with fnames to run similiarity eval,atleast 2, separted by a comma",
        type=str)
    parser.add_argument("--load_model",
                        help="to load previously saved model checkpoint",
                        default=False)
    parser.add_argument(
        "--cpt",
        help="# of the save model cpt to load, only valid if valid_cpt is true"
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    loss_ = torch.nn.CrossEntropyLoss(reduction='sum')
    encoder = ACCENT_ENCODER_AE(dataset_train=args.dataset_train,
                                dataset_val=args.dataset_val,
                                device=device,
                                loss_=loss_,
                                load_model=args.load_model).to(device=device)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=1e-2)
    encoder.opt = optimizer
    cpt = args.cpt
    if args.load_model:
        encoder.load_model_cpt(cpt=cpt, device=device)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lr_lambda=lambda x: x*0.95)
    lr_scheduler = None
    if args.mode == 'train':
        encoder.train_loop(optimizer,
                           lr_scheduler,
                           loss_,
                           batch_size=encoder.config_yml['ACC_COUNT'],
                           cpt=cpt)
    elif args.mode == 'eval':
        fnames = [
            os.path.join(args.filedir, x)
            for x in sorted(os.listdir(args.filedir))
        ]

        sim_matrix = encoder.sim_matrix_infer(fnames, cpt)
        encoder.vis_sim_matrix(sim_matrix, fnames)
