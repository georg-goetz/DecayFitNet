import toolbox.core as core
import pickle

if __name__ == '__main__':
    print('Reading dataset.')
    dataset_decays = core.DecayDataset(n_slopes_max=3, edcs_per_slope=50000, testset_flag=False)

    input_transform = {"edcs_db_normfactor": dataset_decays.edcs_db_normfactor}
    pickle.dump(input_transform, open('../model/input_transform.pkl', 'wb'))
