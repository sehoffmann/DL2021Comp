import torch
from torch.utils.data import DataLoader , Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
# from imgaug.augmenters import Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, \
#         Noop, Lambda, AssertLambda, AssertShape, Scale, CropAndPad, \
#         Pad, Crop, Fliplr, Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, \
#         Grayscale, GaussianBlur, AverageBlur, MedianBlur, Convolve, \
#         Sharpen, Emboss, EdgeDetect, DirectedEdgeDetect, Add, AddElementwise, \
#         AdditiveGaussianNoise, Multiply, MultiplyElementwise, Dropout, \
#         CoarseDropout, Invert, ContrastNormalization, Affine, PiecewiseAffine, \
#         ElasticTransformation

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):

        x=self.data[index].astype(np.uint8)
        x= self.transform(x)
        # plt.imshow(x.numpy().transpose((1,2,0)))
        # plt.show()

        y = Image.fromarray(self.targets[index].astype(np.uint8))
        y = transforms.ToTensor()(y)
        return x, y

    def __len__(self):
        return len(self.data)


def get_dataloaders(batch_size,validation_set_size = 1000, train_data_path = 'data/train_noisy.npy',train_labels_path = 'data/train_clean.npy',test_data_path = 'data/test_noisy_100.npy'):
    """
    :param batch_size:
    :param validation_set_size: usually 10 percent of the training set. In the final run you want to submit you can set it to ß,since more date will increase the training performance.
    :param train_data_path: See Kaggle challenge for these files
    :param train_labels_path:
    :param test_data_path:
    :return: train_dataloader,val_dataloader,test_dataloader.  The retuned labels of the test dataloader are zero tensors.
    """
    test_data = np.load(test_data_path)
    train_data = np.load(train_data_path)
    train_labels = np.load(train_labels_path)


    # transform_train = transforms.Compose([transforms.ToTensor()])
    transform_train = transforms.Compose([
    iaa.Sequential([
    iaa.Sometimes(0.5, iaa.Affine(scale=(1.0, 1.2))),
    iaa.Sometimes(0.5, iaa.Affine(rotate=(-90, 90))),
    ], random_order=False).augment_image,transforms.ToTensor()
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    full_train_set = MyDataset(train_data, train_labels, transform=transform_train)
    train_size = train_data.shape[0] - validation_set_size
    val_size = validation_set_size
    lengths = [train_size, val_size]
    train_set, val_set = torch.utils.data.dataset.random_split(full_train_set, lengths,
                                                               torch.Generator().manual_seed(42))
    test_set = MyDataset(test_data, np.zeros_like(test_data), transform=transform_test)
    num_parallel_loading_threads=4
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=num_parallel_loading_threads)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True,num_workers=num_parallel_loading_threads)
    test_dataloader = DataLoader(test_set, batch_size=batch_size)

    return train_dataloader,val_dataloader,test_dataloader