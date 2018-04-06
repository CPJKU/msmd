# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:40:17 2015

@author: matthias
"""

from __future__ import print_function
import sys

import numpy as np
from multiprocessing import Pool

# --- function definitions ---


def batch_compute1(X, compute, batch_size, verbose=False, prepare=None):
    """ Batch compute data """
    
    # init results
    R = None
    
    # get number of samples
    n_samples = X.shape[0]
    
    # get input shape
    in_shape = list(X.shape)[1:]
    
    # get number of batches
    n_batches = int(np.ceil(float(n_samples) / batch_size))
    
    # iterate batches
    for i_batch in xrange(n_batches):

        if verbose:
            print("Processing batch %d / %d" % (i_batch + 1, n_batches), end='\r')
            sys.stdout.flush()

        # extract batch
        start_idx = i_batch * batch_size
        excerpt = slice(start_idx, start_idx + batch_size)
        E = X[excerpt]
        
        # append zeros if batch is to small
        n_missing = batch_size - E.shape[0]
        if n_missing > 0:
            E = np.vstack((E, np.zeros([n_missing] + in_shape, dtype=X.dtype)))

        if prepare is not None:
            E = prepare(E)

        # compute results on batch
        r = compute(E)
        
        # init result array
        if R is None:
            R = np.zeros([n_samples] + list(r.shape[1:]), dtype=r.dtype)
        
        # store results
        R[start_idx:start_idx+r.shape[0]] = r[0:batch_size-n_missing]
        
    return R


def batch_compute2(X1, X2, compute, batch_size, prepare1=None, prepare2=None):
    """ Batch compute data """

    # init results
    R = None

    # get number of samples
    n_samples = X1.shape[0]

    # get input shape
    in_shape1 = list(X1.shape)[1:]
    in_shape2 = list(X2.shape)[1:]

    # get number of batches
    n_batches = int(np.ceil(float(n_samples) / batch_size))

    # iterate batches
    for i_batch in xrange(n_batches):

        # extract batch
        start_idx = i_batch * batch_size
        excerpt = slice(start_idx, start_idx + batch_size)
        E1, E2 = X1[excerpt], X2[excerpt]

        # append zeros if batch is to small
        n_missing = batch_size - E1.shape[0]
        if n_missing > 0:
            E1 = np.vstack((E1, np.zeros([n_missing] + in_shape1, dtype=X1.dtype)))
            E2 = np.vstack((E2, np.zeros([n_missing] + in_shape2, dtype=X2.dtype)))

        if prepare1 is not None:
            E1 = prepare1(E1)

        if prepare2 is not None:
            E2 = prepare1(E2)

        # compute results on batch
        r = compute(E1, E2)

        # init result array
        if R is None:
            R = np.zeros([n_samples] + list(r.shape[1:]), dtype=r.dtype)

        # store results
        R[start_idx:start_idx+r.shape[0]] = r[0:batch_size-n_missing]

    return R


def threaded_generator(generator, num_cached=10):
    """
    Threaded generator
    """
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    queue = Queue.Queue(maxsize=num_cached)
    end_marker = object()

    # define producer
    def producer():
        for item in generator:
            #item = np.array(item)  # if needed, create a copy here
            queue.put(item)
        queue.put(end_marker)

    # start producer
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer
    item = queue.get()
    while item is not end_marker:
        yield item
        queue.task_done()
        item = queue.get()


def generator_from_iterator(iterator):
    """
    Compile generator from iterator
    """
    for x in iterator:
        yield x


def threaded_generator_from_iterator(iterator, num_cached=10):
    """
    Compile threaded generator from iterator
    """
    generator = generator_from_iterator(iterator)
    return threaded_generator(generator, num_cached)


# --- class definitions ---


class MultiviewPoolIteratorUnsupervised(object):
    """
    Batch iterator for multiview data
    """

    def __init__(self, batch_size, prepare=None, k_samples=None, shuffle=True):
        self.batch_size = batch_size

        if prepare is None:
            def prepare(x, y):
                return x, y
        self.prepare = prepare
        self.shuffle = shuffle

        self.k_samples = k_samples
        self.epoch_counter = 0
        self.n_epochs = None

    def __call__(self, pool):
        self.pool = pool
        if self.k_samples is None:
            self.k_samples = self.pool.shape[0]
        self.n_batches = self.k_samples // self.batch_size
        self.n_epochs = max(1, self.pool.shape[0] // self.k_samples)

        return self

    def __iter__(self):
        n_samples = self.k_samples
        bs = self.batch_size

        # compute current epoch index
        idx_epoch = np.mod(self.epoch_counter, self.n_epochs)

        for i in range((n_samples + bs - 1) / bs):

            i_start = i * bs + idx_epoch * self.k_samples
            i_stop = (i + 1) * bs + idx_epoch * self.k_samples
            sl = slice(i_start, i_stop)
            xb, zb = self.pool[sl]

            if xb.shape[0] < self.batch_size:
                n_missing = self.batch_size - xb.shape[0]

                x_con, z_con = self.pool[0:n_missing]

                xb = np.concatenate((xb, x_con))
                zb = np.concatenate((zb, z_con))

            yield self.transform(xb, zb)

        self.epoch_counter += 1

        # shuffle train data after full set iteration
        if self.shuffle and (idx_epoch + 1) == self.n_epochs:
            self.pool.reset_batch_generator()

    def transform(self, xb, zb):
        return self.prepare(xb, zb)


class TripleviewPoolIteratorUnsupervised(object):
    """
    Batch iterator for multiview data
    """

    def __init__(self, batch_size, prepare=None, k_samples=None, shuffle=True):
        self.batch_size = batch_size

        if prepare is None:
            def prepare(x, y, z):
                return x, y, z
        self.prepare = prepare
        self.shuffle = shuffle

        self.k_samples = k_samples
        self.epoch_counter = 0
        self.n_epochs = None

    def __call__(self, pool):
        self.pool = pool
        if self.k_samples is None:
            self.k_samples = self.pool.shape[0]
        self.n_batches = self.k_samples // self.batch_size
        self.n_epochs = max(1, self.pool.shape[0] // self.k_samples)

        return self

    def __iter__(self):
        n_samples = self.k_samples
        bs = self.batch_size

        # compute current epoch index
        idx_epoch = np.mod(self.epoch_counter, self.n_epochs)

        for i in range((n_samples + bs - 1) / bs):

            i_start = i * bs + idx_epoch * self.k_samples
            i_stop = (i + 1) * bs + idx_epoch * self.k_samples
            sl = slice(i_start, i_stop)
            xb, zb, wb = self.pool[sl]

            if xb.shape[0] < self.batch_size:
                n_missing = self.batch_size - xb.shape[0]

                x_con, z_con, w_con = self.pool[0:n_missing]

                xb = np.concatenate((xb, x_con))
                zb = np.concatenate((zb, z_con))
                wb = np.concatenate((wb, w_con))

            yield self.transform(xb, zb, wb)

        self.epoch_counter += 1

        # shuffle train data after full set iteration
        if self.shuffle and (idx_epoch + 1) == self.n_epochs:
            self.pool.reset_batch_generator()

    def transform(self, xb, zb, wb):
        return self.prepare(xb, zb, wb)


# --- main ---

if __name__ == '__main__':
    pass