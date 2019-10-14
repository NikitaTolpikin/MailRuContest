import keras
import numpy as np
from skimage.transform import resize
import random


class Generator(keras.utils.Sequence):

    def __init__(self, itemIds, usr_coords, gtruth, delta, batch_size=32, mask_size=256, shuffle=True,
                 augment=False, predict=False):
        self.itemIds = itemIds
        self.usr_coords = usr_coords
        self.gtruth = gtruth
        self.delta = delta
        self.batch_size = batch_size
        self.mask_size = mask_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()

    def find_shape(self, itemId):
        xmin = 5000
        xmax = 0
        ymin = 5000
        ymax = 0

        for coord in self.usr_coords[itemId]:
            if coord['Xmin'] < xmin:
                xmin = coord['Xmin']
            if coord['Ymin'] < ymin:
                ymin = coord['Ymin']
            if coord['Xmax'] > xmax:
                xmax = coord['Xmax']
            if coord['Ymax'] > ymax:
                ymax = coord['Ymax']

        width = xmax - xmin
        height = ymax - ymin

        return [width, height], xmin, ymin

    def create_usr_mask(self, itemId, shape, xmin, ymin):
        msk = np.zeros(shape)
        i = 0
        constant = 0.5
        for coord in self.usr_coords[itemId]:
            x1 = coord['Xmin'] - xmin
            y1 = coord['Ymin'] - ymin
            x2 = coord['Xmax'] - xmin
            y2 = coord['Ymax'] - ymin
            msk[x1:x2, y1:y2] += 1
            i += 1
        msk = msk / i*2
        msk += constant

        (width, height) = shape
        dw = int(width * self.delta)
        dh = int(height * self.delta)

        msk = np.pad(msk, ((dw, dw), (dh, dh)), mode='constant', constant_values=constant)

        (width, height) = msk.shape

        if width > height:
            msk = np.pad(msk, ((0, 0), (0, width - height)), mode='constant', constant_values=constant)
        elif width < height:
            msk = np.pad(msk, ((0, height - width), (0, 0)), mode='constant', constant_values=constant)

        return msk, dw, dh

    def __load__(self, itemId):
        constant = 0.5
        usr_msk_shape, xmin, ymin = self.find_shape(itemId)
        usr_msk, dw, dh = self.create_usr_mask(itemId, usr_msk_shape, xmin, ymin)
        gt_msk = np.zeros(usr_msk.shape)
        if itemId in self.gtruth:
            x1 = self.gtruth[itemId]['Xmin'] - xmin + dw
            y1 = self.gtruth[itemId]['Ymin'] - ymin + dh
            x2 = self.gtruth[itemId]['Xmax'] - xmin + dw
            y2 = self.gtruth[itemId]['Ymax'] - ymin + dh
            gt_msk[x1:x2, y1:y2] = 1
        usr_msk = resize(usr_msk, (self.mask_size, self.mask_size), mode='constant', cval=constant)
        gt_msk = resize(gt_msk, (self.mask_size, self.mask_size), mode='constant')
        usr_msk = np.expand_dims(usr_msk, -1)
        gt_msk = np.expand_dims(gt_msk, -1)
        return usr_msk, gt_msk

    def __loadpredict__(self, itemId):
        constant = 0.5
        usr_msk_shape, xmin, ymin = self.find_shape(itemId)
        usr_msk, dw, dh = self.create_usr_mask(itemId, usr_msk_shape, xmin, ymin)
        usr_msk = resize(usr_msk, (self.mask_size, self.mask_size), mode='constant', cval=constant)
        usr_msk = np.expand_dims(usr_msk, -1)
        return usr_msk

    def __getitem__(self, index):
        itemIds = self.itemIds[index * self.batch_size:(index + 1) * self.batch_size]
        if self.predict:
            usr_msks = [self.__loadpredict__(itemId) for itemId in itemIds]
            usr_msks = np.array(usr_msks)
            return usr_msks, itemIds
        else:
            items = [self.__load__(itemId) for itemId in itemIds]
            usr_msks, gt_msks = zip(*items)
            usr_msks = np.array(usr_msks)
            gt_msks = np.array(gt_msks)
            return usr_msks, gt_msks

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.itemIds)

    def __len__(self):
        if self.predict:
            return int(np.ceil(len(self.itemIds) / self.batch_size))
        else:
            return int(len(self.itemIds) / self.batch_size)
