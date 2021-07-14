import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

from decaynet_toolbox import DecaynetToolbox
import core

EVAL_TYPE = 'roomtransition'
# EVAL_TYPE = 'motus'
# DATA_PATH = '/Volumes/scratch/elec/t40527-hybridacoustics/datasets/decayfit_toolbox'
DATA_PATH = '/Volumes/ARTSRAM/AkuLab_Datasets'

AUDIO_EXTENSIONS = ['.wav']


def is_audio_file(filename):
    filename_lower = filename.lower()
    return any((filename_lower.endswith(ext) and not filename_lower.startswith('.')) for ext in AUDIO_EXTENSIONS)


def make_dataset(dir):
    audio_files = []

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_audio_file(fname):
                path = os.path.join(root, fname)
                audio_files.append(path)

    return audio_files


class RirDataset(Dataset):
    """Dataset that loads RIRs from a directory."""

    def __init__(self, data_path: str, transform=None):
        print(f"Reading fnames in path {data_path}.")
        assert os.path.exists(data_path), 'ERROR: Data path does not exist.'
        audio_files = make_dataset(data_path)
        if len(audio_files) == 0:
            raise (RuntimeError("Found 0 audio files in subfolders of: " + data_path + "\n"
                                                                                       "Supported audio extensions are: " + ",".join(
                AUDIO_EXTENSIONS)))

        self.data_path = data_path
        self.audio_files = audio_files
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        path = self.audio_files[index]
        audio, _ = torchaudio.load(path)
        if audio.shape[0] > 1:  # load the omni channel
            audio = audio[0, :]
        if self.transform is not None:
            audio = self.transform(audio)

        return audio


def test_fit_precomputedEDCs():
    """ Test the fit of the pre trained DecayFitNet to one of the test datasets."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the model
    decaynet = DecaynetToolbox(sample_rate=48000, normalization=True, device=device)

    # Process
    if EVAL_TYPE == 'motus':
        f_edcs = h5py.File(os.path.join(DATA_PATH, 'summer830', 'edcs.mat'), 'r')
        edcs = torch.from_numpy(np.array(f_edcs.get('summer830edcs/edcs'))).float().view(-1, 2400)

        n_batches = 240
        batch_size = 83
    elif EVAL_TYPE == 'roomtransition':
        f_edcs = h5py.File(os.path.join(DATA_PATH, 'roomtransition', 'edcs.mat'), 'r')
        edcs = torch.from_numpy(np.array(f_edcs.get('roomTransitionEdcs/edcs'))).float().view(-1, 2400)  # range [0,1]

        n_batches = 101
        batch_size = 72
    else:
        raise NotImplementedError()

    with torch.no_grad():
        # Normalize precomputed EDCs: normalized result will be [-inf, 1], and unless the RIR has very small values
        # (<-120dB), it will be somewhere between [-1, 1]. For noisy RIRs, min(edcs_normalized) can still be closer
        # to 0, i.e., it doesn't need to be exactly -1
        edcs_db = 10 * torch.log10(edcs)
        edcs_normalized = 2 * edcs_db / decaynet.input_transform["edcs_db_normfactor"]
        edcs_normalized += 1

        assert torch.all(torch.min(edcs) >= 0 and torch.max(edcs) <= 2), \
            "Range for raw linear EDCs should be [0,1]"  # range [0,1]
        assert torch.all(torch.max(edcs_db) <= 0), \
            "Range for EDCs in dB should be [-inf, 0]"
        assert torch.all(torch.max(edcs_normalized) <= 1), \
            "Range for EDCs in dB should be [-inf, 0]"  # usually somewhere between [-1, 1], but cannot be enforced

        all_mse = torch.Tensor([])
        edcMSELoss = 0
        for idx in range(n_batches):
            these_edcs = edcs[idx * batch_size:(idx + 1) * batch_size, :]
            these_edcs_normalized = edcs_normalized[idx * batch_size:(idx + 1) * batch_size, :]

            prediction = decaynet.estimate_parameters(these_edcs_normalized, do_preprocess=False,
                                                      do_scale_adjustment=False)
            t_prediction, a_prediction, n_prediction = prediction[0], prediction[1], prediction[2]

            # Write arbitary number (1) into T values that are equal to zero (inactive slope), because their amplitude
            # will be 0 as well (i.e. they don't contribute to the EDC)
            t_prediction[t_prediction == 0] = 1

            thisLoss = core.edc_loss(t_prediction, a_prediction, n_prediction, these_edcs, device, training_flag=False)

            print('Batch {}/{} [{:.2f} %] -- \t EDC Loss: {:.2f} dB'.format(idx, n_batches, 100 * idx / n_batches,
                                                                            thisLoss))

            edcMSELoss += (1 / n_batches) * thisLoss

            this_loss_edcwise = torch.mean(core.edc_loss(t_prediction, a_prediction, n_prediction, these_edcs, device,
                                                         training_flag=False, apply_mean=False), 1)

            all_mse = torch.cat((all_mse, this_loss_edcwise), 0)

        print('MSE loss on {}: {} dB'.format(EVAL_TYPE, edcMSELoss))

        assert torch.mean(all_mse) < 1, "The mean error should be < 1 dB"

        import seaborn as sns
        p = sns.boxplot(data=all_mse.numpy())
        p.set_yscale("log")
        plt.show()

        print('Test finished succesfully.')


def test_fit_preprocessEDCs():
    """ Test the fit of the pre trained DecayFitNet to one of the test datasets, doing the preprocessing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare the model
    decaynet = DecaynetToolbox(sample_rate=48000, normalization=True, device=device)

    if EVAL_TYPE == 'motus':
        data_path = os.path.join(DATA_PATH, 'summer830', 'sh_rirs')
    elif EVAL_TYPE == 'roomtransition':
        data_path = os.path.join(DATA_PATH, 'roomtransition', 'Wav Files')
    else:
        raise NotImplementedError()

    with torch.no_grad():
        dataset = RirDataset(data_path=data_path)
        num_workers = 4 if torch.cuda.is_available() else 0
        loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=num_workers)
        n_batches = len(loader)

        all_mse = torch.Tensor([])
        edcMSELoss = 0
        for idx, rir in enumerate(loader):
            # Room transition dataset has a fade-out window of 0.1s, which must be removed before the fitting
            if EVAL_TYPE == 'roomtransition':
                rir = rir[:, 0:round(-0.1 * 48000)]

            # This is the fully preprocessed (Octave-band filtering, normalization, downsampling) EDC, which is used as
            # the input of the network
            edcs_preprocessed = decaynet.preprocess(rir.to(device))
            assert not torch.any(torch.isnan(edcs_preprocessed)), "Nan after pre processing"
            assert not torch.any(torch.isinf(edcs_preprocessed)), "Inf after pre processing"

            # Additionally, we need the unnormalized EDCs to compare the fitting result against
            # 1) Schroeder integration
            edcs_unnormalized = decaynet._preprocess.schroeder(rir.to(device))
            # 2) Discard last 5 percent of EDC
            edcs_unnormalized = decaynet._preprocess.discard_last5(edcs_unnormalized)
            # 3) Downsample
            edcs_unnormalized = torch.nn.functional.interpolate(edcs_unnormalized, size=2400, scale_factor=None,
                                                                mode='linear', align_corners=False,
                                                                recompute_scale_factor=None)
            # 4) reshape like the prediction
            edcs_unnormalized = edcs_unnormalized.view(-1, 2400)

            # do prediction with network
            prediction = decaynet.estimate_parameters(edcs_preprocessed, do_preprocess=False, do_scale_adjustment=False)
            t_prediction, a_prediction, n_prediction = prediction[0].to(device), \
                                                       prediction[1].to(device), \
                                                       prediction[2].to(device)

            # Write arbitary number (1) into T values that are equal to zero (inactive slope), because their amplitude
            # will be 0 as well (i.e. they don't contribute to the EDC)
            t_prediction[t_prediction == 0] = 1

            # Compare the fitting result with the unnormalized EDCs
            thisLoss = core.edc_loss(t_prediction, a_prediction, n_prediction, edcs_unnormalized, device,
                                     training_flag=False)

            print('Batch {}/{} [{:.2f} %] -- \t EDC Loss: {:.2f} dB'.format(idx, n_batches, 100 * idx / n_batches,
                                                                            thisLoss))

            edcMSELoss += (1 / n_batches) * thisLoss

            this_loss_edcwise = torch.mean(core.edc_loss(t_prediction, a_prediction, n_prediction,
                                                         edcs_unnormalized, device, training_flag=False,
                                                         apply_mean=False), 1)

            all_mse = torch.cat((all_mse.detach().cpu(), this_loss_edcwise.detach().cpu()), 0)

        print('MSE loss on {}: {} dB'.format(EVAL_TYPE, edcMSELoss))

        assert torch.mean(all_mse) < 1, "The mean error should be > 1 dB"

        import seaborn as sns
        p = sns.boxplot(data=all_mse.numpy())
        p.set_yscale("log")
        plt.show()

        print('Test finished succesfully.')


def helper():
    """ This is just a helper function to debug. Nothing important to see here. """
    from utils import plot_waveform
    fs = 48000  # 24000

    # plot_waveform(x, fs)
    plot_waveform(edcs[0:6, :].unsqueeze(0), fs)
    plot_waveform(edcs_normalized.unsqueeze(0), fs)
    plot_waveform(edcs_normalized.permute([1, 0, 2]), fs)


if __name__ == '__main__':
    test_fit_precomputedEDCs()
    test_fit_preprocessEDCs()
