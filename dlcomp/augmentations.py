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

    return aug


def _weak1():
    """
    from imgaug guide
    """
    aug = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order

    return iaa.Sometimes(0.9, aug)


def _weak2(**cfg):
    """
    doesn't introduce artifacts or changes the "black box noise" by a lot
    thus stays more true to the original distribution
    """
    strength = cfg.pop('strength', 1.0)
    proportion = cfg.pop('proportion', 1.0)

    aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(
            scale={"x": (1 - 0.1*strength, 1 + 0.1*strength), "y": (1 - 0.1*strength, 1 + 0.1*strength)}, 
            translate_percent={"x": (-0.1*strength, 0.1*strength), "y": (-0.1*strength, 0.1*strength)}, 
            mode='symmetric',
            order=3  # bicubic
        ),
        iaa.MultiplySaturation((1.0 - 0.8*strength, 1.0 + 0.3*strength)),
        iaa.AddToHue((-255, 255))
    ])

    if proportion < 1.0:
        aug = iaa.Sometimes(proportion, aug)

    return aug


def _weak3(**cfg):
    """
    doesn't introduce artifacts or changes the "black box noise" by a lot
    thus stays more true to the original distribution
    """
    strength = cfg.pop('strength', 1.0)
    proportion = cfg.pop('proportion', 1.0)

    aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.CoarseDropout((0.00, 0.05 + strength*0.5), size_percent=(0.04, 0.08)),
        iaa.Affine(
            scale={"x": (1 - 0.1*strength, 1 + 0.1*strength), "y": (1 - 0.1*strength, 1 + 0.1*strength)}, 
            translate_percent={"x": (-0.1*strength, 0.1*strength), "y": (-0.1*strength, 0.1*strength)}, 
            mode='symmetric',
            order=3  # bicubic
        ),
        iaa.MultiplySaturation((1.0 - 0.8*strength, 1.0 + 0.3*strength)),
        iaa.AddToHue((-255, 255))
    ])

    if proportion < 1.0:
        aug = iaa.Sometimes(proportion, aug)

    return aug
    

def _medium1(**cfg):
    strength = cfg.pop('strength', 1.0)
    proportion = cfg.pop('proportion', 1.0)

    aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Sometimes(0.33, iaa.CoarseDropout((0.03, 0.03 + strength*0.1), size_percent=(0.05, 0.08))),
        
        iaa.Affine(
            scale={"x": (1 - 0.1*strength, 1 + 0.1*strength), "y": (1 - 0.1*strength, 1 + 0.1*strength)}, 
            rotate=(strength*-20, strength*20),
            translate_percent={"x": (-0.1*strength, 0.1*strength), "y": (-0.1*strength, 0.1*strength)}, 
            mode='symmetric',
            order=3  # bicubic
        ),
        iaa.Sometimes(0.3*strength, iaa.GaussianBlur(sigma=(0.5, 1.0+strength*1.0))),
        iaa.AddToHueAndSaturation((int(strength*-40), int(strength*40)), per_channel=True),
        iaa.LogContrast(gain=(1 - strength*0.2, 1 + strength*0.2), per_channel=True)
    ])

    if proportion < 1.0:
        aug = iaa.Sometimes(proportion, aug)

    return aug


augmentations = {
    'baseline': _baseline,
    'weak1': _weak1,
    'weak2': _weak2,
    'weak3': _weak3,
    'medium1': _medium1,
    'to_tensor': tvt.ToTensor
}

baseline = _baseline()
weak = _weak1()
weak2 = _weak2()
weak3 = _weak3()
medium1 = _medium1()
to_tensor = tvt.ToTensor()
