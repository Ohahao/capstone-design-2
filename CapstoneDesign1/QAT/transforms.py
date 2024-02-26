import random
import torch
import numpy as np

class ToTensor(object):
    """Convert data sample to pytorch tensor"""
    def __call__(self, sample):
        image, noisy = sample.get('image'), sample.get('noisy') #입력받은 data sample에서 key를 활용해 image와 noise data 추출
        image = torch.from_numpy(image.transpose((2, 0, 1)).astype('float32') / 255.) #numpy array에서 tensor로 변환, channel first로 변경

        if noisy is not None:
            noisy = torch.from_numpy(noisy.transpose((2, 0, 1)).astype('float32') / 255.) #noise도 array에서 tensor로 변환, channel first로 변경

        return {'image': image, 'noisy': noisy} #dictionary로 return


class RandomVerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.uniform(0., 1.) < self.p: #위아래로 뒤집기
            image, noisy = sample.get('image'), sample.get('noisy')#입력받은 data sample에서 key를 활용해 image와 noise data 추출
            image = np.flipud(image) #image 뒤집기

            if noisy is not None:
                noisy = np.flipud(noisy) #noise 뒤집기

            return {'image': image, 'noisy': noisy} #dictionary 형태로 return

        return sample #뒤집지 않는 경우 그대로 return


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.uniform(0., 1.) < self.p:
            image, noisy = sample.get('image'), sample.get('noisy')
            image = np.fliplr(image)

            if noisy is not None:
                noisy = np.fliplr(noisy)

            return {'image': image, 'noisy': noisy}

        return sample


class RandomRot90(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.uniform(0., 1.) < self.p:
            image, noisy = sample.get('image'), sample.get('noisy')
            image = np.rot90(image)

            if noisy is not None:
                noisy = np.rot90(noisy)

            return {'image': image, 'noisy': noisy}

        return sample
