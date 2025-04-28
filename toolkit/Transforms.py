import random
import torchvision.transforms.functional as F

class PadImg:
    def __init__(self, height, width):
        self.final_width = width
        self.final_height = height

    def __call__(self, img):
        w, h = img.size
        pad_w = max(0, self.final_width - w)
        pad_h = max(0, self.final_height - h)
        padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
        img = F.pad(img, padding, padding_mode='reflect')
        return img


def random_rotate(image):
    angle = random.choice([0, 90, 180, 270])
    return F.rotate(image, angle)


def rotate_if_wide(img):
    if img.height > img.width:
        return img.rotate(-90, expand=True)
    return img


class CropDivisibleBy:
    def __init__(self, divisor=4):
        self.divisor = divisor

    def __call__(self, img):
        w, h = img.size
        new_h = (h // self.divisor) * self.divisor
        new_w = (w // self.divisor) * self.divisor
        if new_h == 0:
            new_h = self.divisor
        if new_w == 0:
            new_w = self.divisor
        cropped_img = F.center_crop(img, (new_h, new_w))
        return cropped_img

    def __repr__(self):
        return self.__class__.__name__ + f'(divisor={self.divisor})'
