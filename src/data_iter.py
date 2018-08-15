import h5py
import numpy as np

import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_dataset(filename):
    return h5py.File(filename, mode='r')

# only care idx, rather than data
def sequence(hdf5_file, seq_len):
    data_len = hdf5_file['img'].shape[0]
    data_idx = []
    print('dataset index sequence')
    for i in range(data_len-seq_len+1):
        idx = i + np.arange(seq_len)
        if any([hdf5_file['episode_start'][index, ...] for index in idx[1:]]):
            continue
        data_idx.append(idx)
    
    np.random.shuffle(data_idx)
    example_num = len(data_idx)

    return data_idx

def action_combination(label_sc):
    move_forward_set = [5]
    move_forward_left_set = [1, 7, 11, 17]
    move_forward_right_set = [9]
    left_attack_set = [2, 8, 13, 14, 18, 19, 20, 25]
    right_attack_set = [3, 4, 10, 15, 16, 21, 23, 24, 27, 28]
    attack_set = [0, 6, 12, 22, 26]
    new_label_sc = np.zeros_like(label_sc)
    for i in range(len(label_sc)):
        if label_sc[i, ...] in move_forward_set:
            new_label_sc[i, ...] = 0
        elif label_sc[i, ...] in move_forward_left_set:
            new_label_sc[i, ...] = 1
        elif label_sc[i, ...] in move_forward_right_set:
            new_label_sc[i, ...] = 2
        elif label_sc[i, ...] in left_attack_set:
            new_label_sc[i, ...] = 3
        elif label_sc[i, ...] in right_attack_set:
            new_label_sc[i, ...] = 4
        else:
            new_label_sc[i, ...] = 5
    return new_label_sc

def train_test_data(filename, seq_len):
    hdf5_file = load_dataset(filename)
    print('dataset name: ', filename)
    print('dataset keys: ', hdf5_file.keys())
    print('dataset example numbers: ', hdf5_file['img'].shape[0])

    train_data_idx = sequence(hdf5_file, seq_len)

    #reduced_label = action_combination(hdf5_file['action_id'])
    reduced_label = hdf5_file['action_id']

    train_img = np.array(hdf5_file['img'])[train_data_idx, ...]
    train_label = np.array(reduced_label)[train_data_idx, ...]
    train_label_gf = np.array(hdf5_file['label_gf'])[train_data_idx, ...]

    train_len = len(train_data_idx)
    assert train_img.shape == (train_len, seq_len, 60, 108, 3)
    assert train_label.shape == (train_len, seq_len, 1)
    assert train_label_gf.shape == (train_len, seq_len, 2)

    print('Dataset prepared!')
    return train_img, train_label, train_label_gf
    
def next_batch(train_img, train_label, train_label_gf, batch_size):
    num_examples = train_img.shape[0]
    idx = list(range(num_examples))
    np.random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = np.array(idx[i: min(i+batch_size, num_examples)])
        yield train_img[j, ...], train_label[j, ...], train_label_gf[j, ...] 

def save_label_distribution(filename, seq_len):
    start = time.time()
    hdf5_file = load_dataset(filename)

    train_data_idx, test_data_idx = sequence(hdf5_file, seq_len)

    train_label = np.array(hdf5_file['action_id'])[train_data_idx, ...]
    test_label = np.array(hdf5_file['action_id'])[test_data_idx, ...]
    
    print('load and sequence dataset use %.02f mins' % ((time.time() - start)/60.))
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].hist(train_label[:, -1, ...], bins=29, alpha=0.9)
    axes[0].set_title('train_label distribution')
    axes[1].hist(test_label[:, -1, ...], bins=29, alpha=0.9)
    axes[1].set_title('test_label distribution')
    axes[1].set_xlabel('action id')
    fig.savefig('../data/fig/train_test_labels_hist.png')
 
if __name__ == '__main__':
    save_label_distribution("../data/dataset.hdf5", 4)
