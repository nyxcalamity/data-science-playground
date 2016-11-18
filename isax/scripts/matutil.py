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
import string

import numpy as np
import scipy.stats as stat

__author__ = "Denys Sobchyshak"
__email__ = "denys.sobchyshak@gmail.com"


def qnorm(n=3):
    """
    Generates quantile values of N(0, 1) for provided number of splits, e.g. for four splits it will generate an
    array [-0.67448975,  0,  0.67448975]
    :param n:
    :return:
    """
    if n > 1:
        return stat.norm.ppf(np.linspace(0, 1, n+1)[1:-1])


def random_walk(n=1000):
    """
    Generates a simple random walk sequence
    :param n:
    :return:
    """
    if n > 0:
        return np.cumsum(np.random.randn(n))


def normalize(series):
    """
    Normalizes features into a mean of 0 and standard deviation of 1.
    :param series:
    :return:
    """
    return (series - series.mean()) / series.std()


def distance(s1, s2, metric='euclidean'):
    """
    Computes distance measure for provided metric parameter
    :param s1:
    :param s2:
    :param metric: euclidean
    :return:
        distance between two time series according to specified metric
    """
    if metric == 'euclidean':
        return math.sqrt(((s1-s2)**2).sum())


def sax_dist(sym1, sym2, symbols, breakpoints):
    """
    Computes distance between two sax symbols
    :param sym1:
    :param sym2:
    :param symbols: list of symbols
    :param breakpoints:
    :return:
    """
    idx1 = symbols.index(sym1)
    idx2 = symbols.index(sym2)
    if abs(idx1-idx2) <= 1:
        return 0
    else:
        return breakpoints[max(idx1, idx2)-1] - breakpoints[min(idx1, idx2)]


def mindist(n, s1, s2, cardinality=256, representation='binary'):
    """
    Calculate distance between two sax series
    :param n:
    :param s1:
    :param s2:
    :param cardinality:
    :param representation: binary, letter, integer
    :return:
        None if series are of different length, distance measure otherwise
    """
    w1, w2 = len(s1), len(s2)
    if w1 == w2:
        symbols = generate_symbols(cardinality, representation)
        breakpoints = qnorm(cardinality)
        distances = [sax_dist(s1[i], s2[i], symbols, breakpoints) for i in range(w1)]
        return math.sqrt(n/w1)*math.sqrt((np.asarray(distances)**2).sum())


def find_symbol(value, breakpoints, symbols):
    """
    Find a symbol for provided value that corresponds to a specific interval of the discretized N(0, 1) distribution
    :param value:
    :param breakpoints:
    :param symbols:
    :return:
    """
    if value > breakpoints[-1]:
        return symbols[-1]
    else:
        for i in range(breakpoints.shape[0]):
            if value <= breakpoints[i]:
                return symbols[i+1]


def generate_symbols(n, representation='binary'):
    """
    Generates n symbols in provided representation. Will not return more than 26 symbols for 'letter' representation.
    Available representations: binary, letter, integer
    :param n:
    :param representation: binary, letter, integer
    :return:
        None, if unknown representation
    """
    if representation == 'binary':
        return ['{:0>{width}b}'.format(i, width=math.ceil(math.log2(n))) for i in range(n)]
    elif representation == 'letter':
        return list(string.ascii_lowercase)[:n]
    elif representation == 'integer':
        return [str(i) for i in range(n)]


def sax(series, w, c=256, representation='letter'):
    """
    Transforms provided series into a Symbolic Aggregate approXimation (SAX) representation. Requires series to be
    normalized around 0 with standard deviation of 1. Will switch to binary representation if cardinality is smaller
    than 26.
    :param series:
    :param w: word length
    :param c: cardinality
    :param representation: binary, letter, integer
    :return:
        None if series length is not divisible by word length without remainder
    """
    if representation == 'letter' and c > 26:
        representation = 'binary'
    aggregate = paa(series, w)
    symbols = generate_symbols(c, representation)
    breakpoints = qnorm(c)
    symbolic = list()
    for i in range(w):
        symbolic.append(find_symbol(aggregate[i], breakpoints, symbols))
    return symbolic


def paa(series, w):
    """
    Transforms provided series into a Piecewise Aggregate Approximation (PAA) representation
    :param series:
    :param w:
        word length or size of the transform with 0 < w <= len(series)
    :return:
        None if w < 1 or len(series) < w, PAA otherwise
    """
    n = len(series)
    series = np.array(series)
    if n == w:
        return series
    if w == 1:
        return [series.mean()]
    if w < 1 or n < w:
        return None
    aggregate = [0]*w
    idx = np.arange(n*w) // w
    for i in range(0, n*w, n):
        aggregate[i//n] = series[idx[i:i+n]].sum() / n
    return aggregate
