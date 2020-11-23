# Codes reference: https://www.kesci.com/mw-org/boyuai/project/5eb3cf11366f4d002d76cfd8

# import keras
import keras.backend as K
from keras.models import Model
# import different layers
from keras.layers import Conv2D, BatchNormalization, Input, Dropout, Add
from keras.layers import Conv2DTranspose, Reshape, Activation
from keras.layers import Concatenate
# import adam optimiser
from keras.optimizers import Adam
# import activation function
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu,tanh
# import image processing package
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
# import glob to process files
import glob
# import random
import random


def load_image(fn, image_size):
    """
    load an image
    fn: path of image
    image_size: size expected
    """
    im = load_img(fn)
    # Cut the image: get the maximal square at the centre and resize the image
    # crop: cut the image, input (x1, y1, x2, y2) positions for left-up and right-down image
    if (im.size[0] >= im.size[1]):
        im = im.crop(((im.size[0] - im.size[1]) // 2, 0, (im.size[0] + im.size[1]) // 2, im.size[1]))
    else:
        im = im.crop((0, (im.size[1] - im.size[0]) // 2, im.size[0], (im.size[0] + im.size[1]) // 2))
    # resize
    im = im.resize((image_size, image_size))
    # transfer int from 0 to 255 into [0, 1]
    arr = img_to_array(im) / 255 * 2 - 1

    return arr

class DataSet(object):
    """
    Class to manage dataset
    """
    def __init__(self, data_path, image_size=256):
        # path of dataset
        self.data_path = data_path
        self.epoch = 0
        # initiate dataset list
        self.__init_list()
        # image size
        self.image_size = image_size

    def __init_list(self):
        # glob.glob given pathname, return list of file names corresponding pathname
        # https://docs.python.org/3/library/glob.html
        self.data_list = glob.glob(self.data_path)
        "/home/su*.jpg"
        # random.shuffle 打乱列表
        # https://docs.python.org/3/library/random.html#random.shuffle
        random.shuffle(self.data_list)
        # initiate the pointer
        self.ptr = 0

    def get_batch(self, batchsize):
        """
        Get (batchsize) pictures
        """
        if (self.ptr + batchsize >= len(self.data_list)):
            # Case when all the pictures have been taken out
            # add pictures in the list
            batch = [load_image(x, self.image_size) for x in self.data_list[self.ptr:]]
            rest = self.ptr + batchsize - len(self.data_list)
            # re-initiate the list
            self.__init_list()
            # add the rest
            batch.extend([load_image(x, self.image_size) for x in self.data_list[:rest]])
            self.ptr = rest
            self.epoch += 1
        else:
            batch = [load_image(x, self.image_size) for x in self.data_list[self.ptr:self.ptr + batchsize]]
            self.ptr += batchsize

        return self.epoch, np.array(batch)

    def get_pics(self, num):
        """
        Take num pictures for snapshot
        This will not influence the queue
        """
        return np.array([load_image(x, self.image_size) for x in random.sample(self.data_list, num)])

def arr2image(X):
    """
    Transfer RGB values into int from [0, 255]
    """
    int_X = ((X + 1) / 2 * 255).clip(0, 255).astype('uint8')
    return array_to_img(int_X)

def generate(img, fn):
    """
    Put image(img) into a network(fn)
    """
    r = fn([np.array([img])])[0]
    return arr2image(np.array(r[0]))


def res_block(x, dim):
    """
    Resnet
    [x] --> [Conv] --> [Normalization] --> [Activation] --> [Conv] --> [Normalization] --> [+] --> [Activation]
     |                                                              ^
     |                                                              |
     +--------------------------------------------------------------+
    """
    # Convolution layer
    # using same padding, no need for bias with the following normalization layer
    x1 = Conv2D(dim, 3, padding="same", use_bias=False)(x)
    # normalization，training mode is 1
    x1 = BatchNormalization()(x1, training=1)
    # relu activation
    x1 = Activation('relu')(x1)
    # convolution layer
    x1 = Conv2D(dim, 3, padding="same", use_bias=False)(x1)
    x1 = BatchNormalization()(x1, training=1)
    # add input and activate
    x1 = Activation("relu")(Add()([x, x1]))

    return x1

def NET_G(ngf=64, block_n=6, downsampling_n=2, upsampling_n=2, image_size=256):
    """
    Generative network with Resnet structure

    block_n: number added for the Resnet
    Here the parameters are: if the size of picture is 128, use 6；if the size of picture is 256, use 9

    [First layer] Convolution layer of size 7, channel number: 3->ngf
    [Down pooling] Convolution layer of size 3, stride = 2, number for every layer doubles
    [Resnet] superposition of 9 block
    [Up pooling]
    [Last layer] turn the number of channels to 3
    """
    # input layer
    input_t = Input(shape=(image_size, image_size, 3))

    # first layer
    x = input_t
    dim = ngf
    x = Conv2D(dim, 7, padding="same")(x)
    x = Activation("relu")(x)

    # Down pooling
    for i in range(downsampling_n):
        dim *= 2
        x = Conv2D(dim, 3, strides=2, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x, training=1)
        x = Activation('relu')(x)

    # Resnet
    for i in range(block_n):
        x = res_block(x, dim)

    # Up pooling
    for i in range(upsampling_n):
        dim = dim // 2
        x = Conv2DTranspose(dim, 3, strides=2, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x, training=1)
        x = Activation('relu')(x)

    # Last layer
    dim = 3
    x = Conv2D(dim, 7, padding="same")(x)
    x = Activation("tanh")(x)

    return Model(inputs=input_t, outputs=x)

def NET_D(ndf=64, max_layers=3, image_size=256):
    """
    Discriminator network
    max_layers: layer number
    """
    # input layer
    input_t = Input(shape=(image_size, image_size, 3))
    x = input_t
    x = Conv2D(ndf, 4, padding="same", strides=2)(x)

    # Use LeakyReLU as activation function
    x = LeakyReLU(alpha=0.2)(x)
    dim = ndf

    for i in range(1, max_layers):
        dim *= 2
        x = Conv2D(dim, 4, use_bias=False, padding="same", strides=2)(x)
        x = BatchNormalization()(x, training=1)
        x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(dim, 4, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x, training=1)
    x = LeakyReLU(alpha=0.2)(x)

    # At last, calculate sigmoid for every pixel in feature map and calculate the mean value
    x = Conv2D(1, 4, padding="same", activation="sigmoid")(x)

    return Model(inputs=input_t, outputs=x)

def loss_func(output, target):
    """
    square error is recommended in the paper
    """
    return K.mean(K.square(output - target))


class CycleGAN(object):
    def __init__(self, image_size=256, lambda_cyc=10, lrD=2e-4, lrG=2e-4, ndf=64, ngf=64, resnet_blocks=9):
        """
        Construct the network
                      cyc loss
         +---------------------------------+
         |            (CycleA)             |
         v                                 |
        realA -> [GB] -> fakeB -> [GA] -> recA
         |                 |
         |                 +---------------+
         |                                 |
         v                                 v
        [DA]         <CycleGAN>           [DB]
         ^                                 ^
         |                                 |
         +----------------+                |
                          |                |
        recB <- [GB] <- fakeA <- [GA] <- realB
         |                                 ^
         |            (CycleB)             |
         +---------------------------------+
                        cyc loss
        """
        # create a generative network
        self.GA = NET_G(image_size=image_size, ngf=ngf, block_n=resnet_blocks)
        self.GB = NET_G(image_size=image_size, ngf=ngf, block_n=resnet_blocks)
        # create a discriminator network
        self.DA = NET_D(image_size=image_size, ndf=ndf)
        self.DB = NET_D(image_size=image_size, ndf=ndf)

        # get new, generated and recovered A and B tensor
        # the real image is the only input for generative network
        realA, realB = self.GB.inputs[0], self.GA.inputs[0]
        # the fake image is the only output for generative network
        fakeB, fakeA = self.GB.outputs[0], self.GA.outputs[0]
        recA, recB = self.GA([fakeB]), self.GB([fakeA])
        # get the function generating fake image from real image
        self.cycleA = K.function([realA], [fakeB, recA])
        self.cycleB = K.function([realB], [fakeA, recB])
        # get the result of discriminator network
        DrealA, DrealB = self.DA([realA]), self.DB([realB])
        DfakeA, DfakeB = self.DA([fakeA]), self.DB([fakeB])
        # calculate loss function
        lossDA, lossGA, lossCycA = self.get_loss(DrealA, DfakeA, realA, recA)
        lossDB, lossGB, lossCycB = self.get_loss(DrealB, DfakeB, realB, recB)
        lossG = lossGA + lossGB + lambda_cyc * (lossCycA + lossCycB)
        lossD = lossDA + lossDB
        # update parameters
        updaterG = Adam(lr=lrG, beta_1=0.5).get_updates(lossG, self.GA.trainable_weights + self.GB.trainable_weights)
        updaterD = Adam(lr=lrD, beta_1=0.5).get_updates(lossD, self.DA.trainable_weights + self.DB.trainable_weights)
        # build training function (those 2 can be used to train the network)
        self.trainG = K.function([realA, realB], [lossGA, lossGB, lossCycA, lossCycB], updaterG)
        self.trainD = K.function([realA, realB], [lossDA, lossDB], updaterD)

    def get_loss(self, Dreal, Dfake, real, rec):
        """
        get tensor for the loss function
        """
        lossD = loss_func(Dreal, K.ones_like(Dreal)) + loss_func(Dfake, K.zeros_like(Dfake))
        lossG = loss_func(Dfake, K.ones_like(Dfake))
        lossCyc = K.mean(K.abs(real - rec))
        return lossD, lossG, lossCyc

    def train(self, A, B):
        errDA, errDB = self.trainD([A, B])
        errGA, errGB, errCycA, errCycB = self.trainG([A, B])
        return errDA, errDB, errGA, errGB, errCycA, errCycB


def train_batch(batchsize, train_A, train_B):
    """
    Get a batch of data from dataset
    """
    epa, a = train_A.get_batch(batchsize)
    epb, b = train_B.get_batch(batchsize)
    return max(epa, epb), a, b

def gen(generator, X):
    # 把X中的每张图都送进generator里面
    r = np.array([generator([np.array([x])]) for x in X])
    g = r[:, 0, 0]
    rec = r[:, 1, 0]
    return g, rec


def snapshot(cycleA, cycleB, A, B):
    """
    Generate a snapshot
    A, B are 2 batches
    cycleA is a circle A->B->A
    cycleB is a circle B->A->B

    Return a picture
    +-----------+     +-----------+
    | X (in A)  | ... |  Y (in B) | ...
    +-----------+     +-----------+
    |   GB(X)   | ... |   GA(Y)   | ...
    +-----------+     +-----------+
    | GA(GB(X)) | ... | GB(GA(Y)) | ...
    +-----------+     +-----------+
    """
    gA, recA = gen(cycleA, A)
    gB, recB = gen(cycleB, B)

    lines = [
        np.concatenate(A.tolist() + B.tolist(), axis=1),
        np.concatenate(gA.tolist() + gB.tolist(), axis=1),
        np.concatenate(recA.tolist() + recB.tolist(), axis=1)
    ]
    arr = np.concatenate(lines)

    return arr2image(arr)

# image size in the network
IMG_SIZE = 128
# dataset name
DATASET = "vangogh2photo"
# dataset path
dataset_path = "../input/CycleGAN2791/{0}/{0}/".format(DATASET)
# training set path
trainA_path = dataset_path + "trainA/*.jpg"
trainB_path = dataset_path + "trainB/*.jpg"
train_A = DataSet(trainA_path, image_size = IMG_SIZE)
train_B = DataSet(trainB_path, image_size = IMG_SIZE)
# Create the model
model = CycleGAN(image_size = IMG_SIZE)
# Train the codes
from IPython.display import display

EPOCH_NUM = 20
epoch = 0

DISPLAY_INTERVAL = 200
SNAPSHOT_INTERVAL = 1000

BATCH_SIZE = 1

iter_cnt = 0
err_sum = np.zeros(6)

while epoch < EPOCH_NUM:
    # get a batch of data
    epoch, A, B = train_batch(BATCH_SIZE)
    # train in the network and get error
    err = model.train(A, B)
    # accumulate the error
    err_sum += np.array(err)
    iter_cnt += 1
    if (iter_cnt % DISPLAY_INTERVAL == 0):
        # calculate mean value for error
        err_avg = err_sum / DISPLAY_INTERVAL
        print('[iteration%d] discriminator loss: A %f B %f generative loss: A %f B %f cycle loss: A %f B %f'
              % (iter_cnt,
                 err_avg[0], err_avg[1], err_avg[2], err_avg[3], err_avg[4], err_avg[5]),
              )
        # return to 0
        err_sum = np.zeros_like(err_sum)

    if (iter_cnt % SNAPSHOT_INTERVAL == 0):
        # show the snapshot
        A = train_A.get_pics(4)
        B = train_B.get_pics(4)
        display(snapshot(model.cycleA, model.cycleB, A, B))
