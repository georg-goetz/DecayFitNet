import h5py
import torch
import numpy as np
import scipy.io

import toolbox.core as core
import toolbox.utils as utils

import pickle
import os
import pathlib

UNITS_PER_LAYER = 1500
DROPOUT = 0.0
N_LAYERS = 3
N_FILTER = 128

EVAL_TYPE = 'roomtransition'

MODEL_PATH = pathlib.Path.joinpath(pathlib.Path(__file__).parent.parent, 'model')
NETWORK_NAME = 'DecayFitNet.pth'

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_transform = pickle.load(open(os.path.join(MODEL_PATH, 'input_transform.pkl'), 'rb'))

    net = core.DecayFitNetLinear(3, UNITS_PER_LAYER, N_FILTER, N_LAYERS, 0, DROPOUT, 1, device)

    utils.load_model(net, os.path.join(MODEL_PATH, NETWORK_NAME), device)
    net.eval()

    fs = 240
    l_edc = 10

    # Generate the t values that would be discarded as well, otherwise the models do not match.
    t = (torch.linspace(0, l_edc * fs - 1, round((1 / 0.95) * l_edc * fs)) / fs)

    if EVAL_TYPE == 'synth':
        f_edcs = h5py.File('/Volumes/ARTSRAM/AkuLab_Datasets/decayfitnet_training/edcs.mat', 'r')
        edcs = torch.from_numpy(np.array(f_edcs.get('edcs'))).float().T

        # Generate vector that specifies how many slopes are in every EDC
        n_slopes_groundtruth = torch.zeros((edcs.shape[0]))
        for slope_idx in range(1, 3 + 1):
            n_slopes_groundtruth[(slope_idx - 1) * 50000:slope_idx * 50000] = slope_idx
        n_slopes_groundtruth = n_slopes_groundtruth.long()

        n_batches = 1000
        batch_size = 150
        all_n_correct = 0
        all_n_over = 0
        all_n_under = 0
    elif EVAL_TYPE == 'MEAS':
        raise NotImplementedError()
    elif EVAL_TYPE == 'motus':
        f_edcs = h5py.File('/Volumes/ARTSRAM/AkuLab_Datasets/summer830/edcs.mat', 'r')
        edcs = torch.from_numpy(np.array(f_edcs.get('summer830edcs/edcs'))).float().view(-1, 2400)

        n_batches = 240
        batch_size = 83
    elif EVAL_TYPE == 'roomtransition':
        f_edcs = h5py.File('/Volumes/ARTSRAM/AkuLab_Datasets/roomtransition/edcs.mat', 'r')
        edcs = torch.from_numpy(np.array(f_edcs.get('roomTransitionEdcs/edcs'))).float().view(-1, 2400)

        n_batches = 101
        batch_size = 72
    else:
        raise NotImplementedError()

    with torch.no_grad():
        edcs_db = 10 * torch.log10(edcs)
        edcs_normalized = 2 * edcs_db / input_transform["edcs_db_normfactor"]
        edcs_normalized += 1

        all_mse = torch.Tensor([])
        all_fits = torch.Tensor([])
        all_a_vals = torch.Tensor([])
        all_t_vals = torch.Tensor([])
        all_n_vals = torch.Tensor([])
        edcMSELoss = 0
        for idx in range(n_batches):
            these_edcs = edcs[idx * batch_size:(idx + 1) * batch_size, :]
            these_edcs_normalized = edcs_normalized[idx * batch_size:(idx + 1) * batch_size, :]

            t_prediction, a_prediction, n_prediction, n_slopes_probabilities = net(these_edcs_normalized)

            t_prediction, a_prediction, n_prediction, n_slopes_prediction = core.postprocess_parameters(t_prediction,
                                                                                                        a_prediction,
                                                                                                        n_prediction,
                                                                                                        n_slopes_probabilities,
                                                                                                        device)

            thisLoss = core.edc_loss(t_prediction, a_prediction, n_prediction, these_edcs, device,
                                     training_flag=False)

            if EVAL_TYPE == 'synth':
                these_n_slopes_groundtruth = n_slopes_groundtruth[idx * batch_size:(idx + 1) * batch_size]
                n_correct = torch.sum(these_n_slopes_groundtruth == n_slopes_prediction)
                n_over = torch.sum(these_n_slopes_groundtruth < n_slopes_prediction)
                n_under = torch.sum(these_n_slopes_groundtruth > n_slopes_prediction)
                all_n_correct += n_correct
                all_n_over += n_over
                all_n_under += n_under
                print('Batch {}/{} [{:.2f} %] -- \t EDC Loss: {:.2f} dB -- '
                      '\t NSlope prediction accuracy: {:.2f}% (over-estimated: {:.2f}%, '
                      'under-estimated: {:.2f}%)'.format(idx, n_batches, 100 * idx / n_batches, thisLoss,
                                                         100 * n_correct / batch_size, 100 * n_over / batch_size,
                                                         100 * n_under / batch_size))
            else:
                print('Batch {}/{} [{:.2f} %] -- \t EDC Loss: {:.2f} dB'.format(idx, n_batches, 100 * idx / n_batches,
                                                                                thisLoss))

            edcMSELoss += (1 / n_batches) * thisLoss

            this_loss_edcwise = torch.mean(core.edc_loss(t_prediction, a_prediction, n_prediction,
                                                         these_edcs, device, training_flag=False,
                                                         apply_mean=False), 1)

            all_mse = torch.cat((all_mse, this_loss_edcwise), 0)

            these_fits = core.generate_synthetic_edc(t_prediction, a_prediction, torch.pow(10, n_prediction),
                                                     t, device)

            all_fits = torch.cat((all_fits, these_fits), 0)
            all_a_vals = torch.cat((all_a_vals, a_prediction), 0)
            all_t_vals = torch.cat((all_t_vals, t_prediction), 0)
            all_n_vals = torch.cat((all_n_vals, 10 ** n_prediction), 0)

        print('MSE loss on {}: {} dB'.format(EVAL_TYPE, edcMSELoss))
        mse_mat = {'mse': all_mse.numpy().astype(np.double)}
        scipy.io.savemat('mse_decayfitnet_{}.mat'.format(EVAL_TYPE), mse_mat)

        fits_mat = {'fits_DecayFitNet': all_fits.numpy().astype(np.double),
                    't_vals': all_t_vals.numpy().astype(np.double),
                    'a_vals': all_a_vals.numpy().astype(np.double),
                    'n_vals': all_n_vals.numpy().astype(np.double)}
        scipy.io.savemat('fits_decayfitnet_{}.mat'.format(EVAL_TYPE), fits_mat)

        if EVAL_TYPE == 'synth':
            print('NSlope prediction accuracy: {:.2f}% (over-estimated: {:.2f}%, '
                  'under-estimated: {:.2f}%)'.format(all_n_correct / 1500, all_n_over / 1500, all_n_under / 1500))

            n_slope_accuracy = {'n_correct': all_n_correct, 'n_over': all_n_over, 'n_under': all_n_under}
            scipy.io.savemat('n_slope_accuracy_{}.mat'.format(EVAL_TYPE), n_slope_accuracy)
