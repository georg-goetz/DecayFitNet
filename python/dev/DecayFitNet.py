import argparse

import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import tools

class DecayDataset(Dataset):
    """Decay dataset."""

    def __init__(self, n_slopes_min=1, n_slopes_max=5, edcs_per_slope=10000, triton_flag=False, testset_flag=False,
                 testset_name='summer830'):
        """
        Args:
        """
        self.testset_flag = testset_flag

        if triton_flag:
            datasets_dir = '/scratch/elec/t40527-hybridacoustics/datasets/decayfitting/'
        else:
            datasets_dir = '/Volumes/ARTSRAM/GeneralDecayEstimation/decayFitting/'

        if not testset_flag:
            # Load EDCs
            f_edcs = h5py.File(datasets_dir + 'edcs.mat', 'r')
            edcs = np.array(f_edcs.get('edcs'))

            # Load noise values
            f_noise_levels = h5py.File(datasets_dir + 'noiseLevels.mat', 'r')
            noise_levels = np.array(f_noise_levels.get('noiseLevels'))

            # Get EDCs into pytorch format
            edcs = torch.from_numpy(edcs).float()
            self.edcs = edcs[:, (n_slopes_min - 1) * edcs_per_slope:n_slopes_max * edcs_per_slope]

            # Put EDCs into dB
            edcs_db = 10*torch.log10(self.edcs)
            assert not torch.any(torch.isnan(edcs_db)), 'NaN values in db EDCs'

            # Normalize dB values to lie between -1 and 1 (input scaling)
            self.edcs_db_normfactor = torch.max(torch.abs(edcs_db))
            edcs_db_normalized = 2 * edcs_db / self.edcs_db_normfactor
            edcs_db_normalized += 1

            assert not torch.any(torch.isnan(edcs_db_normalized)), 'NaN values in normalized EDCs'
            assert not torch.any(torch.isinf(edcs_db_normalized)), 'Inf values in normalized EDCs'
            self.edcs_db_normalized = edcs_db_normalized

            # Generate vector that specifies how many slopes are in every EDC
            self.n_slopes = torch.zeros((1, self.edcs.shape[1]))
            for slope_idx in range(n_slopes_min, n_slopes_max+1):
                self.n_slopes[0, (slope_idx - 1) * edcs_per_slope:slope_idx * edcs_per_slope] = slope_idx - 1
            self.n_slopes = self.n_slopes.long()

            # Noise level values are used in training for the noise loss
            noise_levels = torch.from_numpy(noise_levels).float()
            self.noise_levels = noise_levels[:, (n_slopes_min-1)*edcs_per_slope:n_slopes_max*edcs_per_slope]

            assert self.edcs.shape[1] == self.noise_levels.shape[1], 'More EDCs than noise_levels'
        else:
            if testset_name == 'summer830':
                f_edcs = h5py.File(datasets_dir + 'summer830/edcs.mat', 'r')
                edcs = torch.from_numpy(np.array(f_edcs.get('summer830edcs/edcs'))).float().view(-1, 2400).T
            elif testset_name == 'roomtransition':
                f_edcs = h5py.File(datasets_dir + 'roomtransition/edcs.mat', 'r')
                edcs = torch.from_numpy(np.array(f_edcs.get('roomTransitionEdcs/edcs'))).float().view(-1, 2400).T
            else:
                raise NotImplementedError('Unknown testset.')

            self.edcs = edcs

    def __len__(self):
        return self.edcs.shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.testset_flag:
            edcs = self.edcs[:, idx]
            return edcs
        else:
            edcs = self.edcs[:, idx]
            edcs_db_normalized = self.edcs_db_normalized[:, idx]
            noise_levels = self.noise_levels[:, idx]
            n_slopes = self.n_slopes[:, idx]

            return edcs, noise_levels, edcs_db_normalized, n_slopes


class DecayFitNetLinear(nn.Module):
    def __init__(self, n_slopes, n_max_units, n_filters, n_layers, relu_slope, dropout, reduction_per_layer, device):
        super(DecayFitNetLinear, self).__init__()

        self.n_slopes = n_slopes
        self.device = device

        self.activation = nn.LeakyReLU(relu_slope)
        self.dropout = nn.Dropout(dropout)

        # Base Network
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size=13, padding=6)
        self.maxpool1 = nn.MaxPool1d(10)
        self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size=7, padding=3)
        self.maxpool2 = nn.MaxPool1d(8)
        self.conv3 = nn.Conv1d(n_filters*2, n_filters*4, kernel_size=7, padding=3)
        self.maxpool3 = nn.MaxPool1d(6)
        self.input = nn.Linear(5*n_filters*4, n_max_units)

        self.linears = nn.ModuleList([nn.Linear(round(n_max_units * (reduction_per_layer**i)),
                                                round(n_max_units * (reduction_per_layer**(i+1)))) for i in range(n_layers-1)])

        # T_vals
        self.final1_t = nn.Linear(round(n_max_units * (reduction_per_layer ** (n_layers - 1))), 50)
        self.final2_t = nn.Linear(50, n_slopes)

        # A_vals
        self.final1_a = nn.Linear(round(n_max_units * (reduction_per_layer ** (n_layers-1))), 50)
        self.final2_a = nn.Linear(50, n_slopes)

        # Noise
        self.final1_n = nn.Linear(round(n_max_units * (reduction_per_layer ** (n_layers-1))), 50)
        self.final2_n = nn.Linear(50, 1)

        # N Slopes
        self.final1_n_slopes = nn.Linear(round(n_max_units * (reduction_per_layer ** (n_layers-1))), 50)
        self.final2_n_slopes = nn.Linear(50, n_slopes)

    def forward(self, edcs):
        """
        Args:

        Returns:
        """

        # Base network
        x = self.maxpool1(self.activation(self.conv1(edcs.unsqueeze(1))))
        x = self.maxpool2(self.activation(self.conv2(x)))
        x = self.maxpool3(self.activation(self.conv3(x)))
        x = self.activation(self.input(self.dropout(x.view(edcs.shape[0], -1))))
        for layer in self.linears:
            x = layer(x)
            x = self.activation(x)

        # T_vals
        t = self.activation(self.final1_t(x))
        t = torch.pow(self.final2_t(t), 2.0) + 0.01

        # A_vals
        a = self.activation(self.final1_a(x))
        a = torch.pow(self.final2_a(a), 2.0) + 1e-16

        # Noise
        n_exponent = self.activation(self.final1_n(x))
        n_exponent = self.final2_n(n_exponent)

        # N Slopes
        n_slopes = self.activation(self.final1_n_slopes(x))
        n_slopes = self.final2_n_slopes(n_slopes)

        return t, a, n_exponent, n_slopes


def generate_synthetic_edc(T, A, noiseLevel, t, device):
    # Calculate decay rates, based on the requirement that after T60 seconds, the level must drop to -60dB
    tau_vals = -torch.log(torch.Tensor([1e-6])).to(device) / T

    # Repeat values such that end result will be (batch_size, n_slopes, sample_idx)
    t_rep = t.repeat(T.shape[0], T.shape[1], 1)
    tau_vals_rep = tau_vals.unsqueeze(2).repeat(1, 1, t.shape[0])

    # Calculate exponentials from decay rates
    time_vals = -t_rep*tau_vals_rep
    exponentials = torch.exp(time_vals)

    # Offset is required to make last value of EDC be correct
    exp_offset = exponentials[:, :, -1].unsqueeze(2).repeat(1, 1, t.shape[0])

    # Repeat values such that end result will be (batch_size, n_slopes, sample_idx)
    A_rep = A.unsqueeze(2).repeat(1, 1, t.shape[0])

    # Multiply exponentials with their amplitudes and sum all exponentials together
    edcs = A_rep * (exponentials - exp_offset)
    edc = torch.sum(edcs, 1)

    # Add noise
    noise = noiseLevel * torch.linspace(len(t), 1, len(t)).to(device)
    edc = edc + noise
    return edc


def edc_loss(t_vals_prediction, a_vals_prediction, n_exp_prediction, edcs_true, device, training_flag=True,
             plot_fit=False, apply_mean=True):
    fs = 240
    l_edc = 10

    # Generate the t values that would be discarded as well, otherwise the models do not match.
    t = (torch.linspace(0, l_edc * fs - 1, round((1/0.95)*l_edc * fs)) / fs).to(device)

    # Clamp noise to reasonable values to avoid numerical problems and go from exponent to actual noise value
    n_exp_prediction = torch.clamp(n_exp_prediction, -32, 32)
    n_vals_prediction = torch.pow(10, n_exp_prediction)

    if training_flag:
        # use L1Loss in training
        loss_fn = nn.L1Loss(reduction='none')
    else:
        loss_fn = nn.MSELoss(reduction='none')

    # Use predicted values to generate an EDC
    edc_prediction = generate_synthetic_edc(t_vals_prediction, a_vals_prediction, n_vals_prediction, t, device)

    # discard last 5 percent (i.e. the step which is already done for the true EDC and the test datasets prior to
    # saving them to the .mat files that are loaded in the beginning of this script
    edc_prediction = edc_prediction[:, 0:l_edc*fs]

    if plot_fit:
        for idx in range(0, edcs_true.shape[0]):
            plt.plot(10 * torch.log10(edcs_true[idx, :]))
            plt.plot(10 * torch.log10(edc_prediction[idx, :].detach()))
            plt.show()

    # Go to dB scale
    edc_true_db = 10 * torch.log10(edcs_true + 1e-16)
    edc_prediction_db = 10 * torch.log10(edc_prediction + 1e-16)

    # Calculate loss on dB scale
    if apply_mean:
        loss = torch.mean(loss_fn(edc_true_db, edc_prediction_db))
    else:
        loss = loss_fn(edc_true_db, edc_prediction_db)

    return loss


def train(args, net, device, trainloader, optimizer, epoch, tb_writer):
    net.train()

    mseloss = nn.L1Loss()
    classification_loss_fn = nn.CrossEntropyLoss()

    n_already_analyzed = 0
    for batch_idx, data_frame in enumerate(trainloader):
        edcs, n_vals, edcs_db_normalized, n_slopes = data_frame

        # To cuda if available
        n_vals = n_vals.to(device)
        edcs = edcs.to(device)
        edcs_db_normalized = edcs_db_normalized.to(device)
        n_slopes = n_slopes.to(device)

        # Prediction
        t_prediction, a_prediction, n_prediction, n_slopes_probabilities = net(edcs_db_normalized)

        '''
        Calculate Losses
        '''
        # First, loss on n slope prediction
        n_slope_loss = classification_loss_fn(n_slopes_probabilities, n_slopes.squeeze())

        # Only use the number of slopes that were predicted, zero others
        _, n_slopes_prediction = torch.max(n_slopes_probabilities, 1)
        n_slopes_prediction += 1  # because python starts at 0
        tmp = torch.linspace(1, args.n_slopes_max, args.n_slopes_max).repeat(n_slopes_prediction.shape[0], 1).to(device)
        mask = tmp.less_equal(n_slopes_prediction.unsqueeze(1).repeat(1, args.n_slopes_max))
        a_prediction[~mask] = 0

        # Calculate EDC Loss
        edc_loss_val = edc_loss(t_prediction, a_prediction, n_prediction, edcs, device)

        # Calculate noise loss
        if args.exclude_noiseloss:
            noise_loss = 0
        else:
            n_vals_true_db = torch.log10(n_vals)
            n_vals_prediction_db = n_prediction  # network already outputs values in dB
            noise_loss = mseloss(n_vals_true_db, n_vals_prediction_db)

        # Add up losses
        total_loss = n_slope_loss + edc_loss_val + noise_loss

        # Do optimizer step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        n_already_analyzed += edcs.shape[0]
        if batch_idx % args.log_interval == 0:
            edc_loss_val_db = edc_loss(t_prediction, a_prediction, n_prediction, edcs, device, training_flag=False)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.3f}, N Slope Loss: {:.3f}, Noise Loss: {:.3f}, '
                  'EDC Loss (scale): {:.3f}, EDC Loss (dB): {:.3f}'.format(epoch, n_already_analyzed,
                                                                           len(trainloader.dataset),
                                                                           100. * n_already_analyzed / len(trainloader.dataset),
                                                                           total_loss, n_slope_loss, noise_loss,
                                                                           edc_loss_val, edc_loss_val_db))
            tb_writer.add_scalar('Loss/Total_train_step', total_loss, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.add_scalar('Loss/N_slope_train_step', n_slope_loss, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.add_scalar('Loss/Noise_train_step', noise_loss, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.add_scalar('Loss/EDC_train_step', edc_loss_val, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.add_scalar('Loss/EDC_dB_train_step', edc_loss_val_db, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.flush()


def test(args, net, device, testloader, tb_writer, epoch, input_transform, testset_name='summer830'):
    net.eval()
    print('=== Test: ' + testset_name + ' ===')

    with torch.no_grad():
        n_already_analyzed = 0
        total_test_loss = 0
        for batch_idx, data_frame in enumerate(testloader):
            edcs = data_frame

            # Normalize according to input transform that was used to normalize training data to -1 .. 1
            edcs_normalized = 2 * 10 * torch.log10(edcs) / input_transform["edcs_db_normfactor"]
            edcs_normalized += 1

            # To cuda if available
            edcs = edcs.to(device)
            edcs_normalized = edcs_normalized.to(device)

            # Prediction
            t_prediction, a_prediction, n_prediction, n_slopes_probabilities = net(edcs_normalized)

            # Only use the number of slopes that were predicted, zero others
            _, n_slopes_prediction = torch.max(n_slopes_probabilities, 1)
            n_slopes_prediction += 1  # because python starts at 0
            tmp = torch.linspace(1, args.n_slopes_max, args.n_slopes_max).repeat(n_slopes_prediction.shape[0], 1).to(device)
            mask = tmp.less_equal(n_slopes_prediction.unsqueeze(1).repeat(1, args.n_slopes_max))
            a_prediction[~mask] = 0

            # Calculate EDC Loss
            edc_loss_val = edc_loss(t_prediction, a_prediction, n_prediction, edcs, device, training_flag=False)
            total_test_loss += (edcs.shape[0] / len(testloader.dataset)) * edc_loss_val

            n_already_analyzed += edcs.shape[0]
            print('Test Epoch: {} [{}/{} ({:.0f}%)]\t EDC Loss (dB): {:.3f}'.format(
                    epoch, n_already_analyzed, len(testloader.dataset),
                    100. * n_already_analyzed / len(testloader.dataset), edc_loss_val))
            tb_writer.add_scalar('Loss/EDC_test_step_' + testset_name, edc_loss_val, (epoch - 1) * len(testloader) + batch_idx)
            tb_writer.flush()

        print('Test Epoch: {} [{}/{} ({:.0f}%)]\t === Total EDC Loss (dB): {:.3f} ==='.format(
            epoch, n_already_analyzed, len(testloader.dataset),
            100. * n_already_analyzed / len(testloader.dataset), total_test_loss))
        tb_writer.add_scalar('Loss/EDC_test_epoch_' + testset_name, total_test_loss, epoch)
        tb_writer.flush()

    return total_test_loss


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Neural network to predict exponential parameters from EDCs")
    parser.add_argument('--triton-flag', action='store_true', default=False,
                        help='has to be true if script is run on triton cluster')
    parser.add_argument('--units-per-layer', type=int, default=1500, metavar='N',
                        help='units per layer in the neural network (default: 1500)')
    parser.add_argument('--n-layers', type=int, default=3, metavar='N_layer',
                        help='number of layers in the neural network (default: 3)')
    parser.add_argument('--n-filters', type=int, default=128, metavar='N_filt',
                        help='number of filters in the conv neural network (default: 128)')
    parser.add_argument('--relu-slope', type=float, default=0.0, metavar='relu',
                        help='negative relu slope (default: 0.0)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='DO',
                        help='probability of dropout (default: 0.0)')
    parser.add_argument('--reduction-per-layer', type=float, default=1, metavar='red',
                        help='fraction for reducting the number of units in consecutive layers '
                             '(default: 1 = no reduction)')
    parser.add_argument('--skip-training', action='store_true', default=False,
                        help='skips training and loads previously trained model')
    parser.add_argument('--exclude-noiseloss', action='store_true', default=False,
                        help='has to be true if the noise loss should be excluded')
    parser.add_argument('--model-filename', default='DecayFitNet',
                        help='filename for saving and loading net (default: DecayFitNet')
    parser.add_argument('--batch-size', type=int, default=2048, metavar='bs',
                        help='input batch size for training (default: 2048)')
    parser.add_argument('--test-batch-size', type=int, default=2048, metavar='bs_t',
                        help='input batch size for testing (default: 2048)')
    parser.add_argument('--n-slopes-min', type=int, default=1, metavar='smin',
                        help='minimum number of slopes to consider (default: 1)')
    parser.add_argument('--n-slopes-max', type=int, default=3, metavar='smax',
                        help='maximum number of slopes to consider (default: 3)')
    parser.add_argument('--edcs-per-slope', type=int, default=10000, metavar='S',
                        help='number of edcs per slope in the dataset (default: 10000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--lr-schedule', type=int, default=5, metavar='LRsch',
                        help='learning rate is reduced with every epoch, restart after lr-schedule epochs (default: 5)')
    parser.add_argument('--weight-decay', type=float, default=1e-3, metavar='WD',
                        help='weight decay of Adam Optimizer (default: 1e-3)')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='SEED',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='LOGINT',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    # set up torch and cuda
    use_cuda = args.use_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir='runs/' + args.model_filename)

    print('Reading dataset.')
    dataset_decays = DecayDataset(args.n_slopes_min, args.n_slopes_max, args.edcs_per_slope, args.triton_flag)

    input_transform = {'edcs_db_normfactor': dataset_decays.edcs_db_normfactor}

    dataset_summer830 = DecayDataset(triton_flag=args.triton_flag, testset_flag=True, testset_name='summer830')

    trainloader = DataLoader(dataset_decays, batch_size=args.batch_size, shuffle=True, **kwargs)
    testloader = DataLoader(dataset_summer830, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Create network
    net = DecayFitNetLinear(args.n_slopes_max, args.units_per_layer, args.n_filters, args.n_layers,
                            args.relu_slope, args.dropout, args.reduction_per_layer, device).to(device)
    net = net.float()

    if not args.skip_training:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.lr_schedule)

        # Training loop
        for epoch in range(1, args.epochs + 1):
            train(args, net, device, trainloader, optimizer, epoch, tb_writer)
            test(args, net, device, testloader, tb_writer, epoch, input_transform, testset_name='summer830')

            scheduler.step()

        tools.save_model(net, args.model_filename + '.pth')

    else:
        tools.load_model(net, args.model_filename + '.pth', device)

        test(args, net, device, testloader, tb_writer, 1111, input_transform, testset_name='summer830')


if __name__ == '__main__':
    main()
