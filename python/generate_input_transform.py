import toolbox.core as core
import pickle

N_SLOPES = 3
EXACTLY_N_SLOPES_MODE = False

if __name__ == '__main__':
    print('Reading dataset.')
    dataset_decays = core.DecayDataset(n_slopes_max=N_SLOPES, testset_flag=False,
                                       exactly_n_slopes_mode=EXACTLY_N_SLOPES_MODE)

    input_transform = {"edcs_db_normfactor": dataset_decays.edcs_db_normfactor}

    if EXACTLY_N_SLOPES_MODE:
        n_slopes_str = '_{}slopes'.format(N_SLOPES)
    else:
        n_slopes_str = ''

    pickle.dump(input_transform, open('../model/input_transform{}.pkl'.format(n_slopes_str), 'wb'))
