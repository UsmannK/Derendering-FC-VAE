# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct
from struct import unpack


def unpack_drawing(file_handle):
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for _ in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x_coord = unpack(fmt, file_handle.read(n_points))
        y_coord = unpack(fmt, file_handle.read(n_points))
        image.append((x_coord, y_coord))

    return image


def unpack_drawings(filename):
    images = []
    with open(filename, 'rb') as f:
        while True:
            try:
                images.append(unpack_drawing(f))
            except struct.error:
                break