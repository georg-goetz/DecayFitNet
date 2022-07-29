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


def train(args, net, trainloader, optimizer, epoch, tb_writer):
    net.train()
    device = net.device

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
        # Only do these steps if n slopes should be predicted by network
        if not net.exactly_n_slopes_mode:
            # First, loss on n slope prediction
            n_slope_loss = classification_loss_fn(n_slopes_probabilities, n_slopes.squeeze())

            # Only use the number of slopes that were predicted, zero others
            _, n_slopes_prediction = torch.max(n_slopes_probabilities, 1)
            n_slopes_prediction += 1  # because python starts at 0
            tmp = torch.linspace(1, args.n_slopes_max, args.n_slopes_max).repeat(n_slopes_prediction.shape[0], 1).to(device)
            mask = tmp.less_equal(n_slopes_prediction.unsqueeze(1).repeat(1, args.n_slopes_max))
            a_prediction[~mask] = 0
        else:
            n_slope_loss = 0

        # Calculate EDC Loss
        edc_loss_mae = core.edc_loss(t_prediction, a_prediction, n_prediction, edcs, device)

        # Calculate noise loss
        if args.exclude_noiseloss:
            noise_loss = 0
        else:
            n_vals_true_db = torch.log10(n_vals)
            n_vals_prediction_db = n_prediction  # network already outputs values in dB
            noise_loss = maeloss(n_vals_true_db, n_vals_prediction_db)

        # Add up losses
        total_loss = n_slope_loss + edc_loss_mae + noise_loss

        # Do optimizer step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        n_already_analyzed += edcs.shape[0]
        if batch_idx % args.log_interval == 0:
            edc_loss_mse = core.edc_loss(t_prediction, a_prediction, n_prediction, edcs, device, training_flag=False)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.3f}, N Slope Loss: {:.3f}, Noise Loss: {:.3f}, '
                  'EDC Loss (MAE, dB): {:.3f}, MSE (dB): {:.3f}'.format(epoch, n_already_analyzed,
                                                                        len(trainloader.dataset),
                                                                        100. * n_already_analyzed / len(
                                                                               trainloader.dataset),
                                                                        total_loss, n_slope_loss, noise_loss,
                                                                        edc_loss_mae, edc_loss_mse))
            tb_writer.add_scalar('Loss/Total_train_step', total_loss, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.add_scalar('Loss/N_slope_train_step', n_slope_loss, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.add_scalar('Loss/Noise_train_step', noise_loss, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.add_scalar('Loss/MAE_step', edc_loss_mae, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.add_scalar('Loss/MSE_step', edc_loss_mse, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.flush()


def test(args, net, testloader, epoch, input_transform, tb_writer):
    net.eval()
    device = net.device
    print('=== Test: Motus ===')

    with torch.no_grad():
        n_already_analyzed = 0
        total_test_loss = 0
        quantile_loss = 0

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

            # If n slopes should be predicted by network: Only use the number of slopes that were predicted, zero others
            if not net.exactly_n_slopes_mode:
                _, n_slopes_prediction = torch.max(n_slopes_probabilities, 1)
                n_slopes_prediction += 1  # because python starts at 0
                tmp = torch.linspace(1, args.n_slopes_max, args.n_slopes_max).repeat(n_slopes_prediction.shape[0], 1).to(device)
                mask = tmp.less_equal(n_slopes_prediction.unsqueeze(1).repeat(1, args.n_slopes_max))
                a_prediction[~mask] = 0

            # Calculate EDC Loss
            edc_loss_val = core.edc_loss(t_prediction, a_prediction, n_prediction, edcs, device, training_flag=False,
                                         apply_mean=False)
            total_test_loss += (edcs.shape[0] / len(testloader.dataset)) * torch.mean(edc_loss_val)
            this_quantile_loss = torch.quantile(torch.mean(edc_loss_val, 1), 0.99)
            if this_quantile_loss > quantile_loss:
                quantile_loss = this_quantile_loss

            n_already_analyzed += edcs.shape[0]
            print('Test Epoch: {} [{}/{} ({:.0f}%)]\t EDC Loss (dB): {:.3f}'.format(
                epoch, n_already_analyzed, len(testloader.dataset),
                100. * n_already_analyzed / len(testloader.dataset), torch.mean(edc_loss_val)))
            tb_writer.add_scalar('Loss/EDC_test_step_Motus', torch.mean(edc_loss_val), (epoch - 1) * len(testloader) + batch_idx)
            tb_writer.flush()

        print('Test Epoch: {} [{}/{} ({:.0f}%)]\t === Total EDC Loss (dB): {:.3f} ==='.format(
            epoch, n_already_analyzed, len(testloader.dataset),
            100. * n_already_analyzed / len(testloader.dataset), total_test_loss))
        tb_writer.add_scalar('Loss/EDC_test_epoch_Motus', total_test_loss, epoch)
        tb_writer.flush()

    return total_test_loss, quantile_loss


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Neural network to predict decay parameters from EDCs")
    parser.add_argument('--units-per-layer', type=int, default=400, metavar='N',
                        help='units per layer in the neural network (default: 400)')
    parser.add_argument('--n-layers', type=int, default=3, metavar='N_layer',
                        help='number of layers in the neural network (default: 3)')
    parser.add_argument('--n-filters', type=int, default=64, metavar='N_filt',
                        help='number of filters in the conv neural network (default: 64)')
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
    parser.add_argument('--n-slopes-max', type=int, default=3, metavar='smax',
                        help='maximum number of slopes to consider (default: 3)')
    parser.add_argument('--batch-size', type=int, default=2048, metavar='bs',
                        help='input batch size for training (default: 2048)')
    parser.add_argument('--test-batch-size', type=int, default=2048, metavar='bs_t',
                        help='input batch size for testing (default: 2048)')
    parser.add_argument('--edcs-per-slope', type=int, default=100000, metavar='S',
                        help='number of edcs per slope in the dataset (default: 100000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='E',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-schedule', type=int, default=40, metavar='LRsch',
                        help='learning rate is reduced with every epoch, restart after lr-schedule epochs (default: 40)')
    parser.add_argument('--weight-decay', type=float, default=3e-4, metavar='WD',
                        help='weight decay of Adam Optimizer (default: 3e-4)')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--exactly-n-slopes-mode', action='store_true', default=False,
                        help='should be true when exactly n slopes should be predicted, i.e., no model order '
                             'prediction is desired')
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
    tb_writer = SummaryWriter(log_dir='training_runs/' + args.model_filename)

    print('Reading dataset.')
    dataset_synthdecays = core.DecayDataset(n_slopes_max=args.n_slopes_max, edcs_per_slope=args.edcs_per_slope,
                                            testset_flag=False, exactly_n_slopes_mode=args.exactly_n_slopes_mode)

    input_transform = {'edcs_db_normfactor': dataset_synthdecays.edcs_db_normfactor}

    dataset_motus = core.DecayDataset(testset_flag=True, exactly_n_slopes_mode=args.exactly_n_slopes_mode)

    trainloader = DataLoader(dataset_synthdecays, batch_size=args.batch_size, shuffle=True, **kwargs)
    testloader = DataLoader(dataset_motus, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Create network
    net = core.DecayFitNet(n_slopes=args.n_slopes_max, n_max_units=args.units_per_layer, n_filters=args.n_filters,
                           n_layers=args.n_layers, relu_slope=args.relu_slope, dropout=args.dropout,
                           reduction_per_layer=args.reduction_per_layer, device=device,
                           exactly_n_slopes_mode=args.exactly_n_slopes_mode).to(device)
    net = net.float()

    if not args.skip_training:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.lr_schedule)

        # Training loop
        for epoch in range(1, args.epochs + 1):
            train(args, net, trainloader, optimizer, epoch, tb_writer)
            test(args, net, testloader, epoch, input_transform, tb_writer)

            scheduler.step()

        utils.save_model(net, '../model/' + args.model_filename + '.pth')

    else:
        utils.load_model(net, '../model/' + args.model_filename + '.pth', device)

        test(args, net, testloader, 1111, input_transform, tb_writer)


if __name__ == '__main__':
    main()
