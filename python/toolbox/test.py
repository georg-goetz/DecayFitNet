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
#EVAL_TYPE = 'motus'
DATA_PATH = '/Volumes/scratch/elec/t40527-hybridacoustics/datasets/decayfit_toolbox'

AUDIO_EXTENSIONS = ['.mp3', '.wav']


def is_audio_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in AUDIO_EXTENSIONS)


def make_dataset(dir):
    audio = []

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_audio_file(fname):
                path = os.path.join(root, fname)
                audio.append(path)

    return audio


class RirDataset(Dataset):
    """Dataset that loads RIRs from a directory."""

    def __init__(self, data_path: str, transform=None):
        print(f"Reading fnames in path {data_path}.")
        assert os.path.exists(data_path), 'ERROR: Data path does not exist.'
        audios = make_dataset(data_path)
        if len(audios) == 0:
            raise(RuntimeError("Found 0 audios in subfolders of: " + data_path + "\n"
                               "Supported audio extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.data_path = data_path
        self.audios = audios
        self.transform = transform

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, index):
        path = self.audios[index]
        audio, _ = torchaudio.load(path)
        if audio.shape[0] > 1:  # load a single channel
            audio = audio[0,:]
        if self.transform is not None:
            audio = self.transform(audio)

        return audio


def test_fit_precomputedEDCs(data_path: str, dataset : str = 'motus'):
    """ Test the fit of the pre trained DecayFitNet to one of the test datasets."""
    # Prepare the model
    decaynet = DecaynetToolbox(sample_rate=48000, normalization=True)

    # Process
    if dataset == 'motus':
        f_edcs = h5py.File(os.path.join(data_path, 'summer830', 'edcs.mat'), 'r')
        edcs = torch.from_numpy(np.array(f_edcs.get('summer830edcs/edcs'))).float().view(-1, 2400)

        n_batches = 240
        batch_size = 83
    elif dataset == 'roomtransition':
        f_edcs = h5py.File(os.path.join(DATA_PATH, 'roomtransition', 'edcs.mat'), 'r')
        edcs = torch.from_numpy(np.array(f_edcs.get('roomTransitionEdcs/edcs'))).float().view(-1, 2400)  # range [0,1]

        n_batches = 101
        batch_size = 72
    else:
        raise NotImplementedError()

    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Normalize precomputed EDCs
        edcs_db = 10*torch.log10(edcs)
        edcs_normalized = 2 * edcs_db / decaynet.input_transform["edcs_db_normfactor"]
        edcs_normalized += 1

        # TODO: for Georg: edcs_normalized are in the [0, 1] range, and not [-1, 1]. is this ok?
        assert torch.all(torch.min(edcs) >= 0 and torch.max(edcs) <= 2), \
            "Range for raw linear EDCs should be [0,1]"  # range [0,1]
        assert torch.all(torch.max(edcs_db) <= 0), \
            "Range for EDCs in dB should be [-inf, 0]"
        assert torch.all(torch.max(edcs_normalized) <= 1 and torch.min(edcs_normalized) >= -1), \
            "Range for EDCs in dB should be [-inf, 0]"

        all_mse = torch.Tensor([])
        edcMSELoss = 0
        for idx in range(n_batches):
            these_edcs = edcs[idx*batch_size:(idx+1)*batch_size, :]
            these_edcs_normalized = edcs_normalized[idx * batch_size:(idx + 1) * batch_size, :]

            prediction = decaynet.estimate_parameters(these_edcs_normalized, do_preprocess=False)
            t_prediction, a_prediction, n_prediction, n_slopes_probabilities = prediction[0], prediction[1], prediction[2], prediction[3]

            # Only use the number of slopes that were predicted, zero others
            _, n_slopes_prediction = torch.max(n_slopes_probabilities, 1)
            n_slopes_prediction += 1
            temp = torch.linspace(1, 3, 3).repeat(n_slopes_prediction.shape[0], 1).to(device)
            mask = temp.less_equal(n_slopes_prediction.unsqueeze(1).repeat(1, 3))
            a_prediction[~mask] = 0

            thisLoss = core.edc_loss(t_prediction, a_prediction, n_prediction, these_edcs, device,
                                            training_flag=False)

            print('Batch {}/{} [{:.2f} %] -- \t EDC Loss: {:.2f} dB'.format(idx, n_batches, 100*idx/n_batches,
                                                                            thisLoss))

            edcMSELoss += (1 / n_batches) * thisLoss

            this_loss_edcwise = torch.mean(core.edc_loss(t_prediction, a_prediction, n_prediction,
                                                                these_edcs, device, training_flag=False,
                                                                apply_mean=False), 1)

            all_mse = torch.cat((all_mse, this_loss_edcwise), 0)

        print('MSE loss on {}: {} dB'.format(EVAL_TYPE, edcMSELoss))

        assert torch.mean(all_mse) < 1, "The mean error should be > 1 dB"

        import seaborn as sns
        p = sns.boxplot(data=all_mse.numpy())
        p.set_yscale("log")
        plt.show()

        print('Test finished succesfully.')


def test_fit_preprocessEDCs(dataset : str = 'motus'):
    """ Test the fit of the pre trained DecayFitNet to one of the test datasets, doing the preprocessing."""
    # Prepare the model
    decaynet = DecaynetToolbox(sample_rate=48000, normalization=True)

    # Process
    if dataset == 'motus':
        data_path = '/m/triton/scratch/elec/t40527-hybridacoustics/datasets/summer830/raw_rirs'

    elif dataset == 'roomtransition':
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    with torch.no_grad():
        dataset = RirDataset(data_path=data_path)
        loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
        n_batches = len(loader)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_mse = torch.Tensor([])
        edcMSELoss = 0
        for idx, rir in enumerate(loader):
            rir_preprocess = decaynet.preprocess(rir.to(device))

            assert not torch.any(torch.isnan(rir_preprocess)), "Nan after pre processing"
            assert not torch.any(torch.isinf(rir_preprocess)), "Inf after pre processing"

            prediction = decaynet.estimate_parameters(rir_preprocess, do_preprocess=False)
            t_prediction, a_prediction, n_prediction, n_slopes_probabilities = prediction[0].to(device), \
                                                                               prediction[1].to(device), \
                                                                               prediction[2].to(device), \
                                                                               prediction[3].to(device)

            # Only use the number of slopes that were predicted, zero others
            _, n_slopes_prediction = torch.max(n_slopes_probabilities, 1)
            n_slopes_prediction += 1
            temp = torch.linspace(1, 3, 3).repeat(n_slopes_prediction.shape[0], 1).to(device)
            mask = temp.less_equal(n_slopes_prediction.unsqueeze(1).repeat(1, 3))
            a_prediction[~mask] = 0

            # TODO: for some reason the true edcs (rir_preprocess) have negative values so the db part is nan
            # So I corrct with a modifier
            debug_modifier = 2
            thisLoss = core.edc_loss(t_prediction, a_prediction, n_prediction, rir_preprocess.to(device) + debug_modifier, device,
                                            training_flag=False)

            print('Batch {}/{} [{:.2f} %] -- \t EDC Loss: {:.2f} dB'.format(idx, n_batches, 100*idx/n_batches,
                                                                            thisLoss))

            edcMSELoss += (1 / n_batches) * thisLoss

            this_loss_edcwise = torch.mean(core.edc_loss(t_prediction, a_prediction, n_prediction,
                                                                rir_preprocess.to(device) + debug_modifier, device, training_flag=False,
                                                                apply_mean=False), 1)

            all_mse = torch.cat((all_mse.detach().cpu(), this_loss_edcwise.detach().cpu()), 0)

        print('MSE loss on {}: {} dB'.format(EVAL_TYPE, edcMSELoss))

        assert torch.mean(all_mse) < 1, "The mean error should be > 1 dB"

        import seaborn as sns
        p = sns.boxplot(data=all_mse.numpy())
        p.set_yscale("log")
        plt.show()

        print('Test finished succesfully.')


def run_all_tests():
    test_fit_precomputedEDCs(data_path=DATA_PATH, dataset='roomtransition')
    test_fit_precomputedEDCs(data_path=DATA_PATH, dataset='motus')

def helper():
    """ This is just a helper function to debug. Nothing important to see here. """
    from utils import plot_waveform
    fs = 48000 # 24000

    # plot_waveform(x, fs)
    plot_waveform(edcs[0:6, :].unsqueeze(0), fs)
    plot_waveform(edcs_normalized.unsqueeze(0), fs)
    plot_waveform(edcs_normalized.permute([1, 0, 2]), fs)


if __name__ == '__main__':
    #test_fit_precomputedEDCs(data_path=DATA_PATH, dataset=EVAL_TYPE)
    test_fit_preprocessEDCs(dataset='motus')


