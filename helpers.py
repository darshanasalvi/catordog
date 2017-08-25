import numpy as np
import os
import cv2
from tqdm import tqdm
from random import shuffle
from matplotlib import pyplot as plt

IM_SIZE = 100
TRAIN_C = 12500
TRAIN_D = 12500
TRAIN_SIZE = TRAIN_C + TRAIN_D
BATCH_SIZE = 250

my_train = int(TRAIN_SIZE * .75)
my_test = TRAIN_SIZE - my_train

model_name = 'dogs-vs-cats'

TEST_SIZE = 12500

here = os.path.abspath('.')
train_dir = os.path.join(here, 'data/train')
test_dir = os.path.join(here, 'data/test')


def load_single(n=-10, filename=None):
    if filename is None:
        if n < 0:
            n = np.random.randint(0, TRAIN_C)
        if np.random.randint(0, 1) == 0:
            filename = os.path.join(train_dir, 'dog.{}.jpg'.format(n))
            label = [1, 0]
        else:
            filename = os.path.join(train_dir, 'cat.{}.jpg'.format(n))
            label = [0, 1]
    else:
        if 'dog.' in filename:
            label = [1, 0]
        else:
            label = [0, 1]

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, dsize=(IM_SIZE, IM_SIZE), dst=img)
    img = np.flipud(img / 255.0)

    #show_image(1, img)
    return img, label


def show_image(fig_num, im, autoscale=False):
    plt.figure(fig_num)
    if not autoscale:
        plt.imshow(im, interpolation='nearest', cmap='gray', vmin=0, vmax=1, origin='lower')
    else:
        plt.imshow(im, interpolation='nearest', cmap='gray', origin='lower')


def make_tst_train():
    doggo_names = [os.path.join(train_dir, 'dog.{}.jpg'.format(n)) for n in
                   np.random.choice(TRAIN_D, TRAIN_D, replace=False)]
    kitty_names = [os.path.join(train_dir, 'cat.{}.jpg'.format(n)) for n in
                   np.random.choice(TRAIN_C, TRAIN_C, replace=False)]
    train_names = doggo_names[0:int(TRAIN_D * .75)] + kitty_names[0:int(TRAIN_D * .75)]
    test_names = doggo_names[int(TRAIN_C * .75):] + kitty_names[int(TRAIN_C * .75):]
    shuffle(train_names)
    shuffle(test_names)
    return train_names, test_names


def create_batches():
    train_names, test_names = make_tst_train()
    num_train_batches = len(train_names) // BATCH_SIZE
    num_test_batches = len(test_names) // BATCH_SIZE

    train_batches = [train_names[b * BATCH_SIZE:(b + 1) * BATCH_SIZE] for b in range(num_train_batches)]
    test_batches = [test_names[b * BATCH_SIZE:(b + 1) * BATCH_SIZE] for b in range(num_test_batches)]
    return train_batches, test_batches

def fetch_batch(img_list):
    im_batch = np.zeros([BATCH_SIZE, IM_SIZE,IM_SIZE])
    lab_batch = np.zeros([BATCH_SIZE, 2])
    i = 0
    for filename in img_list:
        im_batch[i,:,:], lab_batch[i,:] = load_single(filename=filename)
        i+=1
    return im_batch, lab_batch

if __name__ == '__main__':
    train_batches, test_batches = create_batches()
    for train_batch in train_batches:
        im_batch, lab_batch = fetch_batch(train_batch)
