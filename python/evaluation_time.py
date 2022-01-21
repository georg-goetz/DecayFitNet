import h5py
import torch
import time
import numpy as np

import toolbox.core as core
import toolbox.utils as utils
import pickle

UNITS_PER_LAYER = 400
N_LAYERS = 3
N_FILTER = 64

EVAL_TYPE = 'motus'

NETWORK_NAME = 'DecayFitNet.pth'

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_transform = pickle.load(open('../model/input_transform.pkl', 'rb'))

    net = core.DecayFitNet(n_slopes=3, n_max_units=UNITS_PER_LAYER, n_filters=N_FILTER, n_layers=N_LAYERS, relu_slope=0,
                           dropout=0, reduction_per_layer=1, device=device)

    utils.load_model(net, '../model/' + NETWORK_NAME, device)
    net.eval()

    n_params = 0
    for p in net.parameters():
        params_this_layer = 1
        for s in p.size():
            params_this_layer = params_this_layer * s
        n_params += params_this_layer
    print('Number of parameters: {}'.format(n_params))

    fs = 10
    l_edc = 10

    # Generate the t values that would be discarded as well, otherwise the models do not match.
    t = (torch.linspace(0, l_edc * fs - 1, round((1/0.95)*l_edc * fs)) / fs)

    if EVAL_TYPE == 'synth':
        f_edcs = h5py.File('../data/synthEDCs/edcs_100.mat', 'r')
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
        f_edcs = h5py.File('../data/motus/edcs_100.mat', 'r')
        edcs = torch.from_numpy(np.array(f_edcs.get('summer830edcs/edcs'))).float().view(-1, 100)

        n_batches = 240
        batch_size = 83
    elif EVAL_TYPE == 'roomtransition':
        f_edcs = h5py.File('../data/roomtransition/edcs_100.mat', 'r')
        edcs = torch.from_numpy(np.array(f_edcs.get('roomTransitionEdcs/edcs'))).float().view(-1, 100)

        n_batches = 101
        batch_size = 72
    else:
        raise NotImplementedError()

    with torch.no_grad():
        edcs_db = 10*torch.log10(edcs)
        edcs_normalized = 2 * edcs_db / input_transform["edcs_db_normfactor"]
        edcs_normalized += 1

        all_times = []

        for run_idx in range(100):
            start_time = time.time()

            for idx in range(n_batches):
                these_edcs = edcs[idx*batch_size:(idx+1)*batch_size, :]
                these_edcs_normalized = edcs_normalized[idx * batch_size:(idx + 1) * batch_size, :]

                t_prediction, a_prediction, n_prediction, n_slopes_probabilities = net(these_edcs_normalized)

                # Only use the number of slopes that were predicted, zero others
                _, n_slopes_prediction = torch.max(n_slopes_probabilities, 1)
                n_slopes_prediction += 1  # because python starts at 0
                temp = torch.linspace(1, 3, 3).repeat(n_slopes_prediction.shape[0], 1).to(device)
                mask = temp.less_equal(n_slopes_prediction.unsqueeze(1).repeat(1, 3))
                a_prediction[~mask] = 0

            time_elapsed = time.time() - start_time
            all_times.append(time_elapsed)

            print('Elapsed time for run {}: {:.02f}s.'.format(run_idx, time_elapsed))

        avg_time = sum(all_times) / len(all_times)
        print('Average time for fitting entire Motus dataset: {:0.2f}'.format(avg_time))
        print('Std time for fitting entire Motus dataset: {:0.2f}'.format(torch.std(torch.Tensor([all_times]))))



