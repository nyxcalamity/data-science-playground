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
import string
import math
import numpy as np
import scipy.stats as stat

__author__ = "Denys Sobchyshak"
__email__ = "denys.sobchyshak@gmail.com"


def sax(series, w, c=256, representation='letter'):
    if representation == 'letter' and c > 26:
        representation = 'binary'
    # series = normalize(series)
    aggregate = paa(series, w)
    symbols = generate_symbols(c, representation)
    breakpoints = qnorm(c)
    symbolic = list()
    for i in range(w):
        symbolic.append(find_symbol(aggregate[i], breakpoints, symbols))
    return symbolic


def find_symbol(value, breakpoints, symbols):
    if value > breakpoints[-1]:
        return symbols[-1]
    else:
        for i in range(breakpoints.shape[0]):
            if value <= breakpoints[i]:
                return symbols[i+1]


def generate_symbols(n, representation='binary'):
    if representation == 'binary':
        return ['{:0>{width}b}'.format(i, width=math.ceil(math.log2(n))) for i in range(n)]
    elif representation == 'letter':
        return list(string.ascii_lowercase)[:n]
    elif representation == 'integer':
        return [str(i) for i in range(n)]


def paa(series, w):
    n = series.shape[0]
    if n % w != 0:
        return None
    avg_ratio = w/n
    step = n//w
    aggregate = np.zeros(w)
    for i in range(w):
        aggregate[i] = avg_ratio * series[step*i:step*(i+1)].sum()
    return aggregate


def qnorm(n=3):
    if n > 1:
        return stat.norm.ppf(np.linspace(0, 1, n+1)[1:-1])


def random_walk(t=1000):
    if t > 0:
        return np.cumsum(np.random.randn(t))


def normalize(features):
    return (features-features.mean())/features.std()
