#!/usr/bin/env python
"""
    Copyright 2016 Denys Sobchyshak

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import SpanSelector, Slider

import matutil as mt

__author__ = "Denys Sobchyshak"
__email__ = "denys.sobchyshak@gmail.com"


def generate_series(series_length, word_length, sax_cardinality, sym_representation):
    """
    Generates normalized random walk, paa and sax
    :param series_length:
    :param word_length:
    :param sax_cardinality:
    :param sym_representation:
    :return:
    """
    series = mt.normalize(mt.random_walk(series_length))
    aggregate = mt.paa(series, word_length)
    symbolic = mt.sax(series, word_length, sax_cardinality, sym_representation)
    return series, aggregate, symbolic


def step_idx(series_length, word_length):
    """
    Calculates indexes for step plot
    :param series_length:
    :param word_length:
    :return:
    """
    return np.arange(word_length) * (series_length // word_length)


def calculate_sym_frequency(sax_series, sax_cardinality, sym_representation):
    """
    Calculates symbol frequency in provided sax series
    :param sax_series:
    :param sax_cardinality:
    :param sym_representation:
    :return:
    """
    sax_symbols = mt.generate_symbols(sax_cardinality, sym_representation)
    sym_frequency = [sax_series.count(i) for i in sax_symbols]
    sym_idx = np.arange(len(sax_symbols))
    return sym_idx, sax_symbols, sym_frequency


def plot_series(series, aggregate, axis, title, frame=None, enable_legend=False):
    """
    Plots a series with it's step aggregate on provided axis
    :param series:
    :param aggregate:
    :param axis:
    :param title:
    :param frame:
    :param enable_legend:
    :return:
    """
    series_length = len(series)
    word_length = len(aggregate)
    axis.cla()
    axis.plot(series, label='Original Data', color='green', linewidth=1.2)
    axis.step(step_idx(series_length, word_length), aggregate, where='post', label='PAA', color='blue', linewidth=1.6)
    if not frame:
        frame = (0, series_length)
    axis.set_xlim(*frame)
    axis.set_ylim(x[frame[0]:frame[1]].min(), x[frame[0]:frame[1]].max())
    axis.grid()
    if enable_legend:
        axis.legend(loc=2)
    axis.set_title(title)


def plot_frequency(sax_series, sax_cardinality, sym_representation, axis):
    """
    Calculates and plots sax symbol frequency histogram on provided axis
    :param sax_series:
    :param sax_cardinality:
    :param sym_representation:
    :param axis:
    :return:
    """
    idx, symbols, freq = calculate_sym_frequency(sax_series, sax_cardinality, sym_representation)
    bar_width = 0.9
    axis.cla()
    axis.barh(idx, freq, bar_width, alpha=0.5, label='Frequency')
    plt.sca(axis)
    plt.yticks(idx + bar_width / 2, symbols)
    axis.set_xlabel('Total count of occurrences')
    axis.set_ylabel('Symbol')
    axis.grid()


if __name__ == '__main__':
    # parameters
    n = 1000
    w = 100
    c = 30
    representation = 'binary'

    # transform
    x, paa, sax = generate_series(n, w, c, representation)
    # x1, paa1, sax1 = generate_series(n, w, c, representation)
    # print('Distance between two sax strings: {}'.format(mt.mindist(n, sax, sax1, c, representation)))
    # exit()

    # creating layout grid
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 3)
    ax1 = plt.subplot(gs[0:-1, 0:-1])
    ax2 = plt.subplot(gs[-1,   0:-1])
    ax3 = plt.subplot(gs[0:-1,   -1])

    # plotting complete time series
    plot_series(x, paa, ax1, 'Complete series (t={}, w={})'.format(n, w), enable_legend=True)

    # plotting windowed slice
    xmin = math.floor(n / 2)
    xmax = math.ceil(n / 2 + n / 10)
    plot_series(x, paa, ax2, 'Selected window x=[{},{}]'.format(xmin, xmax), frame=(xmin, xmax))

    # adding span selector
    def onselect(xmin, xmax):
        idx_min = math.floor(xmin)
        idx_max = math.ceil(xmax)
        plot_series(x, paa, ax2, 'Selected window x=[{},{}]'.format(idx_min, idx_max), frame=(idx_min, idx_max))
        fig.canvas.draw()
    span_selector = SpanSelector(ax1, onselect, 'horizontal', useblit=True,
                                 rectprops=dict(alpha=0.2, facecolor='magenta'))

    # plotting symbol frequency count
    plot_frequency(sax, c, representation, ax3)

    # plotting parameter sliders
    series_length_slider = Slider(plt.axes([0.75, 0.25, 0.2, 0.03]), 'Series Length',
                                  valmin=100, valmax=10000, valinit=n, valfmt='%d')
    word_length_slider = Slider(plt.axes([0.75, 0.2, 0.2, 0.03]), 'Word Length',
                                valmin=10, valmax=1000, valinit=w, valfmt='%d')
    cardinality_slider = Slider(plt.axes([0.75, 0.15, 0.2, 0.03]), 'Cardinality ',
                                valmin=2, valmax=70, valinit=c, valfmt='%d')

    def update(val):
        global n, w, c, x, paa, sax
        n = int(series_length_slider.val)
        w = int(word_length_slider.val)
        c = int(cardinality_slider.val)
        if n % w != 0:
            n = w * math.ceil(n / w)
        x, paa, sax = generate_series(n, w, c, representation)
        # update series plot
        plot_series(x, paa, ax1, 'Complete series (t={}, w={})'.format(n, w), enable_legend=True)
        # update frequency plot
        plot_frequency(sax, c, representation, ax3)
        fig.canvas.draw_idle()
    series_length_slider.on_changed(update)
    word_length_slider.on_changed(update)
    cardinality_slider.on_changed(update)

    # showing the figure
    fig.tight_layout()
    plt.show()
