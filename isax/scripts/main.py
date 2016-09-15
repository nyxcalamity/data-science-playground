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
import matutil as mt

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import SpanSelector, Slider, Button, RadioButtons

# parameters
t = 1000
w = 100
c = 30
representation = 'binary'

# transform
x = mt.random_walk(t)
X = mt.normalize(x)
paa = mt.paa(X, w)
sax = mt.sax(X, w, c, representation)

# visualization
# creating layout grid
fig = plt.figure()
gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[0:-1, 0:-1])
ax2 = plt.subplot(gs[-1,   0:-1])
ax3 = plt.subplot(gs[0:-1,   -1])
# ax4 = plt.subplot(gs[-1,     -1])

# plotting complete time series
gline, = ax1.plot(X, label='Original Data', color='green', linewidth=1.2)
gstep, = ax1.step(np.arange(w)*(t//w), paa, where='post', label='PAA', color='blue', linewidth=1.6)
ax1.set_title('Complete series (t={}, w={})'.format(t, w))
ax1.grid()
ax1.legend(loc=2)

# plotting windowed slice
lline, = ax2.plot(X, label='Synthetic Series', color='green', linewidth=1.2)
lstep, = ax2.step(np.arange(w)*(t//w), paa, where='post', label='PAA', color='blue', linewidth=1.6)
ax2.set_title('Selected window')
ax2.grid()
window = X[math.floor(t/2):math.ceil(t/2+t/10)]
ax2.set_xlim(math.floor(t/2), math.ceil(t/2+t/10))
ax2.set_ylim(window.min(), window.max())

# adding span selector
def onselect(xmin, xmax):
    idx_min = math.floor(xmin)
    idx_max = math.ceil(xmax)
    local_x = X[idx_min:idx_max]
    lline.set_data(np.arange(t), X.tolist())
    lstep.set_data(np.arange(w) * (t // w), paa)
    ax2.set_xlim(idx_min, idx_max)
    ax2.set_ylim(local_x.min(), local_x.max())
    fig.canvas.draw()
span = SpanSelector(ax1, onselect, 'horizontal', useblit=True, rectprops=dict(alpha=0.2, facecolor='magenta'))

# plotting symbol frequency count
symbols = mt.generate_symbols(c, representation)
frequency = [sax.count(i) for i in symbols]
idx = np.arange(len(symbols))
bar_width = 0.9
ax3.barh(idx, frequency, bar_width, alpha=0.5, label='Frequency')
plt.sca(ax3)
plt.yticks(idx+bar_width/2, symbols)
ax3.set_xlabel('Total count of occurrences')
ax3.set_ylabel('Symbol')
ax3.grid()

# plotting parametrised sliders
axfreq = plt.axes([0.75, 0.2, 0.2, 0.03])
axamp = plt.axes([0.75, 0.25, 0.2, 0.03])
ts_length = Slider(axfreq, 'Series Length', valmin=100, valmax=10000, valinit=t, valfmt='%d')
word_length = Slider(axamp, 'Word Length', valmin=10, valmax=1000, valinit=w, valfmt='%d')
def update(val):
    global t
    global w
    global x
    global X
    global paa
    global sax
    t = int(ts_length.val)
    w = int(word_length.val)
    if t % w != 0:
        t = w*math.ceil(t/w)
    x = mt.random_walk(t)
    X = mt.normalize(x)
    paa = mt.paa(X, w)
    gline.set_data(np.arange(t), X.tolist())
    gstep.set_data(np.arange(w)*(t//w), paa)
    ax1.set_xlim(0, t)
    ax1.set_ylim(X.min(), X.max())
    ax1.set_title('Complete series (t={}, w={})'.format(t, w))
    # updating frequency count
    sax = mt.sax(X, w, c, representation)
    symbols = mt.generate_symbols(c, representation)
    frequency = [sax.count(i) for i in symbols]
    idx = np.arange(len(symbols))
    ax3.cla()
    ax3.barh(idx, frequency, bar_width, alpha=0.5, label='Frequency')
    plt.sca(ax3)
    plt.yticks(idx + bar_width / 2, symbols)
    ax3.set_xlabel('Total count of occurrences')
    ax3.set_ylabel('Symbol')
    ax3.grid()
    fig.canvas.draw_idle()
ts_length.on_changed(update)
word_length.on_changed(update)

# showing the figure
fig.tight_layout()
plt.show()
