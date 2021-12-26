from imgaug import augmenters as iaa
import torchvision.transforms as tvt

# from imgaug.augmenters import Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, \
#         Noop, Lambda, AssertLambda, AssertShape, Scale, CropAndPad, \
#         Pad, Crop, Fliplr, Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, \
#         Grayscale, GaussianBlur, AverageBlur, MedianBlur, Convolve, \
#         Sharpen, Emboss, EdgeDetect, DirectedEdgeDetect, Add, AddElementwise, \
#         AdditiveGaussianNoise, Multiply, MultiplyElementwise, Dropout, \
#         CoarseDropout, Invert, ContrastNormalization, Affine, PiecewiseAffine, \
#         ElasticTransformation

def _baseline():
    aug = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.Affine(scale=(1.0, 1.2))),
        iaa.Sometimes(0.5, iaa.Affine(rotate=(-90, 90))),
    ], random_order=False)

    return tvt.Compose([aug.augment_image,tvt.ToTensor()])


baseline = _baseline()
to_tensor = tvt.ToTensor()
