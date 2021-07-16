import argparse
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import toolbox.core as core
import toolbox.utils as utils


def train(args, net, device, trainloader, optimizer, epoch, tb_writer):
    net.train()

    maeloss = nn.L1Loss()
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

        # Postprocess estimated parameters
        t_prediction, a_prediction, n_prediction, __ = core.postprocess_parameters(t_prediction, a_prediction,
                                                                                   n_prediction, n_slopes_probabilities,
                                                                                   device, sort_values=False)

        # Calculate EDC Loss
        edc_loss_val = core.edc_loss(t_prediction, a_prediction, n_prediction, edcs, device)

        # Calculate noise loss
        if args.exclude_noiseloss:
            noise_loss = 0
        else:
            n_vals_true_db = torch.log10(n_vals)
            n_vals_prediction_db = n_prediction  # network already outputs values in dB
            noise_loss = maeloss(n_vals_true_db, n_vals_prediction_db)

        # Add up losses
        total_loss = n_slope_loss + edc_loss_val + noise_loss

        # Do optimizer step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        n_already_analyzed += edcs.shape[0]
        if batch_idx % args.log_interval == 0:
            edc_loss_val_db = core.edc_loss(t_prediction, a_prediction, n_prediction, edcs, device, training_flag=False)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.3f}, N Slope Loss: {:.3f}, Noise Loss: {:.3f}, '
                  'EDC Loss (scale): {:.3f}, EDC Loss (dB): {:.3f}'.format(epoch, n_already_analyzed,
                                                                           len(trainloader.dataset),
                                                                           100. * n_already_analyzed / len(
                                                                               trainloader.dataset),
                                                                           total_loss, n_slope_loss, noise_loss,
                                                                           edc_loss_val, edc_loss_val_db))
            tb_writer.add_scalar('Loss/Total_train_step', total_loss, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.add_scalar('Loss/N_slope_train_step', n_slope_loss, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.add_scalar('Loss/Noise_train_step', noise_loss, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.add_scalar('Loss/EDC_train_step', edc_loss_val, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.add_scalar('Loss/EDC_dB_train_step', edc_loss_val_db, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.flush()


def test(net, device, testloader, tb_writer, epoch, input_transform, testset_name='summer830'):
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

            # Postprocess estimated parameters
            t_prediction, a_prediction, n_prediction, __ = core.postprocess_parameters(t_prediction, a_prediction,
                                                                                       n_prediction,
                                                                                       n_slopes_probabilities,
                                                                                       device, sort_values=False)

            # Calculate EDC Loss
            edc_loss_val = core.edc_loss(t_prediction, a_prediction, n_prediction, edcs, device, training_flag=False)
            total_test_loss += (edcs.shape[0] / len(testloader.dataset)) * edc_loss_val

            n_already_analyzed += edcs.shape[0]
            print('Test Epoch: {} [{}/{} ({:.0f}%)]\t EDC Loss (dB): {:.3f}'.format(
                epoch, n_already_analyzed, len(testloader.dataset),
                100. * n_already_analyzed / len(testloader.dataset), edc_loss_val))
            tb_writer.add_scalar('Loss/EDC_test_step_' + testset_name, edc_loss_val,
                                 (epoch - 1) * len(testloader) + batch_idx)
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

    # Reproducibility, also add 'env PYTHONHASHSEED=42' to shell
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True  # if set as true, dilated convs are really slow
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir='runs/' + args.model_filename)

    print('Reading dataset.')
    dataset_decays = core.DecayDataset(args.n_slopes_min, args.n_slopes_max, args.edcs_per_slope, args.triton_flag)

    input_transform = {'edcs_db_normfactor': dataset_decays.edcs_db_normfactor}

    dataset_summer830 = core.DecayDataset(triton_flag=args.triton_flag, testset_flag=True, testset_name='summer830')

    trainloader = DataLoader(dataset_decays, batch_size=args.batch_size, shuffle=True, **kwargs)
    testloader = DataLoader(dataset_summer830, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Create network
    net = core.DecayFitNetLinear(args.n_slopes_max, args.units_per_layer, args.n_filters, args.n_layers,
                                 args.relu_slope, args.dropout, args.reduction_per_layer, device).to(device)
    net = net.float()

    if not args.skip_training:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.lr_schedule)

        # Training loop
        for epoch in range(1, args.epochs + 1):
            train(args, net, device, trainloader, optimizer, epoch, tb_writer)
            test(net, device, testloader, tb_writer, epoch, input_transform, testset_name='summer830')

            scheduler.step()

        utils.save_model(net, args.model_filename + '.pth')

    else:
        utils.load_model(net, args.model_filename + '.pth', device)

        test(net, device, testloader, tb_writer, 1111, input_transform, testset_name='summer830')


if __name__ == '__main__':
    main()
