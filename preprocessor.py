import numpy as np
from skimage.transform import resize
from skimage import measure


class Preprocessor:

    def __init__(self, usr_coords, delta, mask_size=256):
        self.usr_coords = usr_coords
        self.delta = delta
        self.mask_size = mask_size

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

    def get_mask(self, itemId):
        constant = 0.5
        usr_msk_shape, xmin, ymin = self.find_shape(itemId)
        usr_msk, dw, dh = self.create_usr_mask(itemId, usr_msk_shape, xmin, ymin)
        width = usr_msk.shape[0]
        usr_msk = resize(usr_msk, (self.mask_size, self.mask_size), mode='constant', cval=constant)
        koef = float(width / self.mask_size)
        usr_msk = np.expand_dims(usr_msk, -1)
        return usr_msk, koef, xmin, ymin, dw, dh

    def get_coords_from_mask(self, msk, koef, xmin, ymin, dw, dh, trashhold=0.5):
        comp = msk[:, :, 0] > trashhold
        comp = measure.label(comp)
        max_sq = 0
        max_reg = {'x': 0, 'y': 0, 'x2': 0, 'y2': 0}

        for region in measure.regionprops(comp):
            x, y, x2, y2 = region.bbox
            if (x2 - x) * (y2 - y) > max_sq:
                max_reg['x'] = x
                max_reg['y'] = y
                max_reg['x2'] = x2
                max_reg['y2'] = y2

        for coord in max_reg:
            max_reg[coord] *= koef
            if coord == 'x' or coord == 'x2':
                max_reg[coord] = int(max_reg[coord] - dw + xmin)
            else:
                max_reg[coord] = int(max_reg[coord] - dh + ymin)

        return max_reg['x'], max_reg['y'], max_reg['x2'], max_reg['y2']
