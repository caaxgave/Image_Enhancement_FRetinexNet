import numpy as np
from PIL import Image

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

def load_images(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 255.0

def save_images(filepath, *args, scale=0):
    print('saved ', filepath)
    assert len(args[0].shape) == 4, str([a.shape for a in args])

    if len(args) == 1:
        cat_image = args[0]
    elif len(args) == 2:
        cat_image = np.concatenate(args, axis = 2)
    elif len(args) == 4:
        img1 = np.concatenate(args[:2], axis = 2)
        img2 = np.concatenate(args[2:], axis = 2)
        cat_image = np.concatenate([img1, img2], axis = 1)

    cat_image = cat_image[0]

    if scale:
        print(cat_image.min(), cat_image.max())
        cat_image -= cat_image.min()
        cat_image /= (cat_image.max() - cat_image.min())

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath)
