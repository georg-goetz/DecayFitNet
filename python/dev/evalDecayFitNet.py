import h5py
import torch
import numpy as np
import scipy.io

import DecayFitNet
import tools
import pickle
import os

UNITS_PER_LAYER = 1500
DROPOUT = 0.0
N_LAYERS = 3
N_FILTER = 128

EVAL_TYPE = 'roomtransition'
EVAL_TYPE = 'motus'
NETWORK_NAME = 'DecayFitNet_final_red100_sl3_3layers_128f_relu0_1500units_do0_b2048_lr5e3_sch5_e100_wd1e3.pth'
DATA_PATH = '/Volumes/scratch/elec/t40527-hybridacoustics/datasets/decayfit_toolbox'

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_transform = pickle.load(open('input_transform_final.pkl', 'rb'))

    net = DecayFitNet.DecayFitNetLinear(3, UNITS_PER_LAYER, N_FILTER, N_LAYERS, 0, DROPOUT, 1, device)

    tools.load_model(net, NETWORK_NAME, device)
    net.eval()

    if EVAL_TYPE == 'SYNTH':
        raise NotImplementedError()
    elif EVAL_TYPE == 'MEAS':
        raise NotImplementedError()
    elif EVAL_TYPE == 'motus':
        f_edcs = h5py.File(os.path.join(DATA_PATH, 'summer830', 'edcs.mat'), 'r')
        edcs = torch.from_numpy(np.array(f_edcs.get('summer830edcs/edcs'))).float().view(-1, 2400)

        n_batches = 240
        batch_size = 83
    elif EVAL_TYPE == 'roomtransition':
        f_edcs = h5py.File(os.path.join(DATA_PATH, 'roomtransition', 'edcs.mat'), 'r')
        edcs = torch.from_numpy(np.array(f_edcs.get('roomTransitionEdcs/edcs'))).float().view(-1, 2400)

        n_batches = 101
        batch_size = 72
    else:
        raise NotImplementedError()

    with torch.no_grad():
        edcs_db = 10*torch.log10(edcs)
        edcs_normalized = 2 * edcs_db / input_transform["edcs_db_normfactor"]
        edcs_normalized += 1

        all_mse = torch.Tensor([])
        edcMSELoss = 0
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

            thisLoss = DecayFitNet.edc_loss(t_prediction, a_prediction, n_prediction, these_edcs, device,
                                            training_flag=False)

            print('Batch {}/{} [{:.2f} %] -- \t EDC Loss: {:.2f} dB'.format(idx, n_batches, 100*idx/n_batches,
                                                                            thisLoss))

            edcMSELoss += (1 / n_batches) * thisLoss

            this_loss_edcwise = torch.mean(DecayFitNet.edc_loss(t_prediction, a_prediction, n_prediction,
                                                                these_edcs, device, training_flag=False,
                                                                apply_mean=False), 1)

            all_mse = torch.cat((all_mse, this_loss_edcwise), 0)

        print('MSE loss on {}: {} dB'.format(EVAL_TYPE, edcMSELoss))
        mse_mat = {'mse': all_mse.numpy()}
        scipy.io.savemat('mse_decayfitnet_{}.mat'.format(EVAL_TYPE), mse_mat)



