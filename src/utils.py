# Copyright 2019 Usmann Khan.
# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
from os import listdir
import numpy as np
from skimage.draw import line
from matplotlib import pyplot as plt

def augment_strokes(strokes, prob=0.0):
    """Perform data augmentation by randomly dropping out strokes."""
    # drop each point within a line segments with a probability of prob
    # note that the logic in the loop prevents points at the ends to be dropped.
    result = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(strokes)):
        candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if candidate[2] == 1 or prev_stroke[2] == 1:
            count = 0
        else:
            count += 1
        urnd = np.random.rand()  # uniform random variable
        if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = candidate
            prev_stroke = stroke
            result.append(stroke)
    return np.array(result)

def to_big_strokes(stroke, max_len=250):
    """Converts from stroke-3 to stroke-5 format and pads to given length."""
    # (But does not insert special start token).

    result = np.zeros((max_len, 5), dtype=float)
    l = len(stroke)
    assert l <= max_len
    result[0:l, 0:2] = stroke[:, 0:2]
    result[0:l, 3] = stroke[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    return result

def get_max_len(strokes):
    """Return the maximum length of an array of strokes."""
    max_len = 0
    for stroke in strokes:
        ml = len(stroke)
        if ml > max_len:
            max_len = ml
    return max_len

def load_dataset(data_dir, max_files=10):
    """load preprepared datasets"""
    assert data_dir[-1] == '/', "data directory does not have trailing slash: {}".format(data_dir)
    data = {'train': [], 'valid': [], 'test': []}
    files = [data_dir + x for x in listdir(data_dir)]

    if len(files) > max_files:
        files = files[:10]
    assert len(files) <= max_files

    for file in files:
        images = np.load(file, allow_pickle=True, encoding='latin1')
        data['train'].extend(images['train'])
        data['test'].extend(images['test'])
        data['valid'].extend(images['valid'])
    return data

def get_bounds(data, factor=10):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)
        

def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result

class DataLoader(object):
    """Class for loading data."""

    def __init__(self,
                 strokes,
                 batch_size=100,
                 max_seq_length=250,
                 scale_factor=1.0,
                 random_scale_factor=0.0,
                 augment_stroke_prob=0.0,
                 limit=1000):
        self.batch_size = batch_size  # minibatch size
        self.max_seq_length = max_seq_length  # N_max in sketch-rnn paper
        self.scale_factor = scale_factor  # divide offsets by this factor
        self.random_scale_factor = random_scale_factor  # data augmentation method
        # Removes large gaps in the data. x and y offsets are clamped to have
        # absolute value no greater than this limit.
        self.limit = limit
        self.augment_stroke_prob = augment_stroke_prob  # data augmentation method
        self.start_stroke_token = [0, 0, 1, 0, 0]  # S_0 in sketch-rnn paper
        # sets self.strokes (list of ndarrays, one per sketch, in stroke-3 format,
        # sorted by size)
        self.preprocess(strokes)
        # self.render(strokes[0])
        # self.images = self.strokes_to_img(strokes)


    def strokes_to_img(self, strokes):
        images = []
        new_strokes = []
        for stroke in strokes:
            new_stroke = []
            x,y = 0,0
            for pt in stroke:
                x += pt[0]
                y += pt[1]
                new_stroke.append([x,y,pt[2]])

            min_x = min([pt[0] for pt in new_stroke])
            min_y = min([pt[1] for pt in new_stroke])

            if min_x < 0:
                for pt in new_stroke:
                    pt[0] += (0 - min_x)
            if min_y < 0:
                for pt in new_stroke:
                    pt[1] += (0 - min_y)

            max_x = max([pt[0] for pt in new_stroke])
            max_y = max([pt[1] for pt in new_stroke])

            max_max = max_y if max_y > max_x else max_x
            scale_factor = 255/max_max

            for pt in new_stroke:
                pt[0] = int(scale_factor * pt[0])
                pt[1] = int(scale_factor * pt[1])
            new_strokes.append(new_stroke)

            img = np.zeros((256,256))
            for i in range(1, len(new_stroke)):
                s_1 = new_stroke[i-1]
                s_2 = new_stroke[i]
                rr, cc = line(s_1[1], s_1[0], s_2[1], s_2[0])
                img[rr, cc] = 1
            images.append(img)
        return images

    def render(self, stroke):
        import cairo
        with cairo.SVGSurface("example1.svg", 1000, 1000) as surface:
            CONTEXT = cairo.Context(surface)
            CONTEXT.set_line_width(10)
            CONTEXT.set_line_cap(cairo.LINE_CAP_ROUND)
            CONTEXT.set_source_rgb(1,1,1)
            CONTEXT.paint()
            # x, y = 250, 250
            x, y = stroke[0][0], stroke[0][1]
            CONTEXT.set_source_rgb(0,0,0)
            CONTEXT.move_to(x,y)
            # CONTEXT.line_to(100,400)
            # CONTEXT.line_to(300,400)
            move = False
            for pt in stroke:
                # x += pt[0]
                # y += pt[1]
                x = pt[0]
                y = pt[1]
                print("{},{}".format(x,y))
                if move:
                    CONTEXT.move_to(x,y)
                else:
                    CONTEXT.line_to(x,y)
                if pt[2] == 1:
                    # CONTEXT.close_path()
                    CONTEXT.stroke()
                    move = True
            # CONTEXT.close_path()
            CONTEXT.stroke()

            CONTEXT.set_source_rgb(1, 0, 0)
            CONTEXT.move_to(250, 250)
            CONTEXT.close_path()
            CONTEXT.stroke()
        print('fin')

    def preprocess(self, strokes):
        """Remove entries from strokes having > max_seq_length points."""
        raw_data = []
        seq_len = []
        count_data = 0

        for i in range(len(strokes)):
            data = strokes[i]
            if len(data) <= (self.max_seq_length):
                count_data += 1
                # removes large gaps from the data
                data = np.minimum(data, self.limit)
                data = np.maximum(data, -self.limit)
                data = np.array(data, dtype=np.float32)
                data[:, 0:2] /= self.scale_factor
                raw_data.append(data)
                seq_len.append(len(data))
        seq_len = np.array(seq_len)  # nstrokes for each sketch
        idx = np.argsort(seq_len)
        self.strokes = []
        for i in range(len(seq_len)):
            self.strokes.append(raw_data[idx[i]])
        print("total images <= max_seq_len is %d" % count_data)
        self.num_batches = int(count_data / self.batch_size)

    def random_sample(self):
        """Return a random sample, in stroke-3 format as used by draw_strokes."""
        sample = np.copy(random.choice(self.strokes))
        return sample

    def random_scale(self, data):
        """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
        x_scale_factor = (
            np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (
            np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        result = np.copy(data)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result

    def calculate_normalizing_scale_factor(self):
        """Calculate the normalizing factor explained in appendix of sketch-rnn."""
        data = []
        for i in range(len(self.strokes)):
            if len(self.strokes[i]) > self.max_seq_length:
                continue
            for j in range(len(self.strokes[i])):
                data.append(self.strokes[i][j, 0])
                data.append(self.strokes[i][j, 1])
        data = np.array(data)
        return np.std(data)

    def normalize(self, scale_factor=None):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        if scale_factor is None:
            scale_factor = self.calculate_normalizing_scale_factor()
        self.scale_factor = scale_factor
        for i in range(len(self.strokes)):
            self.strokes[i][:, 0:2] /= self.scale_factor

    def _get_batch_from_indices(self, indices):
        """Given a list of indices, return the potentially augmented batch."""
        x_batch = []
        seq_len = []
        for idx in range(len(indices)):
            i = indices[idx]
            data = self.random_scale(self.strokes[i])
            data_copy = np.copy(data)
            if self.augment_stroke_prob > 0:
                data_copy = augment_strokes(data_copy, self.augment_stroke_prob)
            x_batch.append(data_copy)
            length = len(data_copy)
            seq_len.append(length)
        seq_len = np.array(seq_len, dtype=int)
        images = self.strokes_to_img(x_batch)
        # We return three things: stroke-3 format, stroke-5 format, list of seq_len.
        return x_batch, self.pad_batch(x_batch, self.max_seq_length), seq_len, images

    def random_batch(self):
        """Return a randomised portion of the training data."""
        idx = np.random.permutation(range(0, len(self.strokes)))[0:self.batch_size]
        return self._get_batch_from_indices(idx)

    def get_batch(self, idx):
        """Get the idx'th batch from the dataset."""
        assert idx >= 0, "idx must be non negative"
        assert idx < self.num_batches, "idx must be less than the number of batches"
        start_idx = idx * self.batch_size
        indices = range(start_idx, start_idx + self.batch_size)
        return self._get_batch_from_indices(indices)

    def pad_batch(self, batch, max_len):
        """Pad the batch to be stroke-5 bigger format as described in paper."""
        result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
        assert len(batch) == self.batch_size
        for i in range(self.batch_size):
            l = len(batch[i])
            assert l <= max_len
            result[i, 0:l, 0:2] = batch[i][:, 0:2]
            result[i, 0:l, 3] = batch[i][:, 2]
            result[i, 0:l, 2] = 1 - result[i, 0:l, 3]
            result[i, l:, 4] = 1
            # put in the first token, as described in sketch-rnn methodology
            result[i, 1:, :] = result[i, :-1, :]
            result[i, 0, :] = 0
            result[i, 0, 2] = self.start_stroke_token[2]  # setting S_0 from paper.
            result[i, 0, 3] = self.start_stroke_token[3]
            result[i, 0, 4] = self.start_stroke_token[4]
        return result
