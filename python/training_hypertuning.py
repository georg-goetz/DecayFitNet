import argparse
import logging
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import optuna
import sys
import joblib

import toolbox.core as core
import training
import toolbox.utils as utils


def objective(trial, args):
    lr = trial.suggest_categorical("lr", [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5,
                                          1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,
                                          1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
                                          1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2])
    wd = trial.suggest_categorical("wd", [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5,
                                          1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,
                                          1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
                                          1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2])
    cos_schedule_factor = trial.suggest_int('cos_sch', 1, 10)
    n_units_factor = 4  # trial.suggest_int('n_units_factor', 1, 5)
    n_filters_exp = 6  # trial.suggest_int('n_filter_exp', 4, 6)
    n_layers = 3  # trial.suggest_int('n_layers', 1, 3)
    cos_schedule = cos_schedule_factor * 5
    n_units = 100 * n_units_factor
    n_filters = 2**n_filters_exp

    print('==== Trial {}:\t lr: {}\t wd: {}\t units: {}\t layers: {}\t '
          'filters: {}\t sch: {} ===='.format(trial.number, lr, wd, n_units, n_layers, n_filters, cos_schedule))

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
    net = core.DecayFitNet(n_slopes=args.n_slopes_max, n_max_units=n_units, n_filters=n_filters,
                           n_layers=n_layers, relu_slope=args.relu_slope, dropout=args.dropout,
                           reduction_per_layer=args.reduction_per_layer, device=device,
                           exactly_n_slopes_mode=args.exactly_n_slopes_mode).to(device)
    net = net.float()

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cos_schedule)

    min_epochs = int(np.max([np.ceil(args.epochs/cos_schedule)*cos_schedule, 5*cos_schedule]))
    total_test_loss = None
    for epoch in range(0, min_epochs):
        training.train(args, net, trainloader, optimizer, epoch, tb_writer)
        edc_loss, quantile_loss = training.test(args, net, testloader, epoch, input_transform, tb_writer)
        total_test_loss = edc_loss + quantile_loss

        if ((epoch+1) % cos_schedule) == 0:
            iteration = int(np.floor((epoch+1)/cos_schedule))
            trial.report(total_test_loss, iteration)
            if trial.should_prune():
                raise optuna.TrialPruned()

        scheduler.step()

    utils.save_model(net, 'hypertuning/' + f'T{trial.number}_' + args.model_filename + '_lr' + str(lr) + '_wd' +
                     str(wd) + '_u' + str(n_units) + '_l' + str(n_layers) + '_f' + str(n_filters)
                     + '_sch' + str(cos_schedule) + '.pth')

    return total_test_loss


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description="Neural network to predict exponential parameters from EDCs")
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
    parser.add_argument('--batch-size', type=int, default=2048, metavar='bs',
                        help='input batch size for training (default: 2048)')
    parser.add_argument('--test-batch-size', type=int, default=2048, metavar='bs_t',
                        help='input batch size for testing (default: 2048)')
    parser.add_argument('--n-slopes-max', type=int, default=3, metavar='smax',
                        help='maximum number of slopes to consider (default: 3)')
    parser.add_argument('--edcs-per-slope', type=int, default=100000, metavar='S',
                        help='number of edcs per slope in the dataset (default: 10000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='E',
                        help='number of epochs to train (default: 100)')
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

    if args.exactly_n_slopes_mode:
        net_str = f'_{args.n_slopes_max}slopes'
    else:
        net_str = ''

    sampler = optuna.samplers.TPESampler(seed=args.seed)

    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(sampler=sampler, pruner=optuna.pruners.MedianPruner(n_min_trials=10),
                                direction='minimize')

    study.optimize(lambda trial: objective(trial, args), timeout=165600)  # 86400 seconds = 1 day
    joblib.dump(study, f'hypertuning/study{net_str}.pkl')

