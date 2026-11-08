from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip, ToTensor, PILToTensor, Normalize, RandomApply, RandomGrayscale, RandomResizedCrop, ColorJitter, InterpolationMode, RandomSolarize
from PIL import ImageFilter
import random

class GaussianBlur:
    def __init__(self, p):
        self.p = p

    def __call__(self, image):
        if random.random() <= self.p:
            image = image.filter( ImageFilter.GaussianBlur(random.uniform(0.1, 2.0)))
        return image


class DINOCropAugmentation:
    def __init__(self, cfg):
        flip_and_color = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomApply([ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            RandomGrayscale(p=0.2),
        ])
        normalize = Compose([
            ToTensor(),
            Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.global1 = Compose([
            RandomResizedCrop(224, scale=cfg.global_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color,
            GaussianBlur(1.0),
            normalize
        ])

        self.global2 = Compose([
            RandomResizedCrop(224, cfg.global_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color,
            GaussianBlur(0.1),
            RandomSolarize(threshold=128, p=0.2),
            normalize
        ])
        self.local = Compose([
            RandomResizedCrop(96, cfg.local_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color,
            GaussianBlur(0.5),
            normalize
        ])
        self.local_views = cfg.local_views
    def __call__(self, image):
        crops = [self.global1(image), self.global2(image)]
        crops += [self.local(image) for _ in range(self.local_views)]
        return crops