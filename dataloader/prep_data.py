import argparse

import numpy as np
from dataloader_mouse import dataloader_mouse_wheel

fpath_lst = [
    "G:/My Drive/Research/data/shikano/aim282_nodeconv/sw24_20231121_ctx-str-cbl-rsc/train-byTrial-sw24_20231121_str.mat",
    # "G:/My Drive/Research/data/shikano/aim282_nodeconv/sw24_20231121_ctx-str-cbl-rsc/train-byTrial-sw24_20231121_ctx.mat",
    # "G:/My Drive/Research/data/shikano/aim282_nodeconv/sw24_20231121_ctx-str-cbl-rsc/train-byTrial-sw24_20231121_cbl.mat",
    # "G:/My Drive/Research/data/shikano/aim282_nodeconv/sw24_20231121_ctx-str-cbl-rsc/train-byTrial-sw24_20231121_rsc.mat",
]
data_params = {
    'seed': 0,
    'min_trial_len' : 10,
    'max_trial_len' : 150,
    'val_size': 0.1,
    'test_size': 0.3,
    'neural_normalizor' : 'zscore',
    'behavior_normalizor' : 'zscore',
    'behavior_keys' : ['speed'],
    'pad': True
}

def prep_data(args):
    train_dataset, valid_dataset, test_dataset, dataset_info = dataloader_mouse_wheel(fpath_lst, **data_params)
    sess_name_lst = list(dataset_info.keys())
    print(f'[INFO] dataset names are {dataset_info.keys()}')

    latent_dim = args.latent_dim

    sess_name = sess_name_lst[0]
    max_seq_len = max([len(trial) for trial in train_dataset[sess_name]['neural_data']])
    print(f'[INFO] number of training trials: {len(train_dataset[sess_name]["neural_data"])}')
    print(f'[INFO] number of validation trials: {len(valid_dataset[sess_name]["neural_data"])}')
    print(f'[INFO] number of test trials: {len(test_dataset[sess_name]["neural_data"])}')
    print(f'[INFO] max sequence length is {max_seq_len}')

    # pad sequences to the same length
    def pad_sequences(sequences, max_len, pad_value=0.0):
        padded_seqs = []
        masks = []
        for seq in sequences:
            seq_len = len(seq)
            if seq_len < max_len:
                pad_width = ((0, max_len - seq_len), (0, 0))
                padded_seq = np.pad(seq, pad_width, mode='constant', constant_values=pad_value)
                mask = np.concatenate([np.ones(seq_len), np.zeros(max_len - seq_len)])
            else:
                padded_seq = seq[:max_len]
                mask = np.ones(max_len)
            padded_seqs.append(padded_seq)
            masks.append(mask)
        return np.array(padded_seqs), np.array(masks)

    train_neural_padded, train_masks = pad_sequences(train_dataset[sess_name]['neural_data'], max_seq_len)
    valid_neural_padded, val_masks = pad_sequences(valid_dataset[sess_name]['neural_data'], max_seq_len)
    test_neural_padded, test_masks = pad_sequences(test_dataset[sess_name]['neural_data'], max_seq_len)
    train_behavior_padded, _ = pad_sequences(train_dataset[sess_name]['behavior_data'], max_seq_len)
    valid_behavior_padded, _ = pad_sequences(valid_dataset[sess_name]['behavior_data'], max_seq_len)
    test_behavior_padded, _ = pad_sequences(test_dataset[sess_name]['behavior_data'], max_seq_len)

    print('[RESULTS] ', train_neural_padded.shape, train_behavior_padded.shape, train_masks.shape)

    num_trials_train = len(train_dataset[sess_name]['neural_data'])
    num_trials_valid = len(valid_dataset[sess_name]['neural_data'])
    num_trials_test = len(test_dataset[sess_name]['neural_data'])
    for trial_idx in range(num_trials_train):
        assert np.sum(train_masks[trial_idx,:]) == len(train_dataset[sess_name]['neural_data'][trial_idx]), f'{trial_idx}th training trial mask incorrect'
    print(f'[INFO] all {num_trials_train} training trial masks correct')
    for trial_idx in range(num_trials_valid):
        assert np.sum(val_masks[trial_idx,:]) == len(valid_dataset[sess_name]['neural_data'][trial_idx]), f'{trial_idx}th validation trial mask incorrect'
    print(f'[INFO] all {num_trials_valid} validation trial masks correct')
    for trial_idx in range(num_trials_test):
        assert np.sum(test_masks[trial_idx,:]) == len(test_dataset[sess_name]['neural_data'][trial_idx]), f'{trial_idx}th test trial mask incorrect'
    print(f'[INFO] all {num_trials_test} test trial masks correct')

    # Option 1: Dummy states (most common for real neural data)
    states_train = np.zeros((num_trials_train, max_seq_len, latent_dim))
    states_valid = np.zeros((num_trials_valid, max_seq_len, latent_dim))
    states_test = np.zeros((num_trials_test, max_seq_len, latent_dim))

    np.save(f'{args.output_dir}/{sess_name}.npy', {
                                                    'train_obs': train_neural_padded,
                                                    'train_behaviors': train_behavior_padded,
                                                    'train_states': states_train,
                                                    'train_masks': train_masks,
                                                    'val_obs': valid_neural_padded,
                                                    'val_behaviors': valid_behavior_padded,
                                                    'val_states': states_valid,
                                                    'val_masks': val_masks,
                                                    'test_obs': test_neural_padded,
                                                    'test_behaviors': test_behavior_padded,
                                                    'test_states': states_test,
                                                    'test_masks': test_masks,
                                                }, allow_pickle=True)
    
    print(f'[INFO] data saved to {args.output_dir}/{sess_name}.npy')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='G:/My Drive/Research/data/shikano/aim282_nodeconv/sw24_20231121_ctx-str-cbl-rsc', help='directory to save the processed data')
    parser.add_argument('--latent_dim', type=int, default=10, help='dimensionality of latent states')
    args = parser.parse_args()
    prep_data(args)