import ctypes
import logging
import multiprocessing as mp

from contextlib import closing

import numpy as np


googleDataPath="data/Google_test"
snopeDataPath="data/Snopes"

info = mp.get_logger().info

reader = ClaimReader(snopeDataPath, googleDataPath)
filePaths = os.listdir(googleDataPath)

def main():
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)
    '''
    # create shared array
    N, M = 100, 11
    shared_arr = mp.Array(ctypes.c_double, N)
    arr = tonumpyarray(shared_arr)

    # fill with random values
    arr[:] = np.random.uniform(size=N)
    arr_orig = arr.copy()
    '''
    shared_claimX = mp.Array(ctypes.c_double, 1)
    shared_relatedSnippetsX = mp.Array(ctypes.c_double, 1)
    # write to arr from different processes
    with closing(mp.Pool(processes=4, initializer=init, initargs=(shared_claimX,shared_relatedSnippetsX))) as p:
        '''
        # many processes access the same slice
        stop_f = N // 10
        p.map_async(f, [slice(stop_f)]*M)

        # many processes access different slices of the same array
        assert M % 2 # odd
        step = N // 10
        p.map_async(g, [slice(i, i + step) for i in range(stop_f, N, step)])
        '''
    p.join()
    #assert np.allclose(((-1)**M)*tonumpyarray(shared_arr), arr_orig)

def init(claimX_, relatedSnippetsX_):
    global shared_claimX
    global shared_relatedSnippetsX
    shared_claimX = claimX_
    shared_relatedSnippetsX = relatedSnippetsX_
    '''
    global shared_arr
    shared_arr = shared_arr_ # must be inhereted, not passed as an argument
    '''

def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj())

def f(i):
    """synchronized."""
    with shared_arr.get_lock(): # synchronize access
        g(i)

def g(i):
    """no synchronization."""
    info("start %s" % (i,))
    arr = tonumpyarray(shared_arr)
    arr[i] = -1 * arr[i]
    info("end   %s" % (i,))

def read(i):
    filePath = filePaths[i]
    articles_, sources_ = reader.readGoogle(filePath)
    claim, cred = reader.readSnopes(filePath)
    claimX = tonumpyarray(shared_claimX)
    relatedSnippetsX = tonumpyarray(shared_relatedSnippetsX)
    for article, source in zip(articles_, sources_):
        claimX_ = np.ones((1,300))
        relatedSnippetsX_ = np.ones((1,300))
        #claimX_, relatedSnippetsX_, relatedSnippets_, _ = rsExtractor.extract(claim, article)
        if relatedSnippets_ is not None:
            if (claimX is None):
                claimX = claimX_
                relatedSnippetsX = relatedSnippetsX_
            else:
                np.vstack((claimX, claimX_))
                np.vstack((relatedSnippetsX, relatedSnippetsX_))


if __name__ == '__main__':
    mp.freeze_support()
    main()