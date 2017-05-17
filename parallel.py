import os
import numpy as np

import ctypes
import logging
import multiprocessing as mp

from contextlib import closing

from src.ClaimReader import ClaimReader



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
    j = mp.Value('i', 0)
    k = mp.Value('i', 0)
    shared_claimX = mp.Array(ctypes.c_double, 57)
    shared_relatedSnippetsX = mp.Array(ctypes.c_double, 3600980100)
    # write to arr from different processes
    with closing(mp.Pool(processes=2, initializer=init, initargs=(shared_claimX, shared_relatedSnippetsX, j, k))) as p:
        '''
        # many processes access the same slice
        stop_f = N // 10
        p.map_async(f, [slice(stop_f)]*M)

        # many processes access different slices of the same array
        assert M % 2 # odd
        step = N // 10
        p.map_async(g, [slice(i, i + step) for i in range(stop_f, N, step)])
        '''
        p.map(readFile_synced, [i for i in range(1)])
    p.join()
    wocao = tonumpyarray(shared_claimX)
    print (wocao.shape)
    print ((wocao==1).shape)
    print (tonumpyarray(shared_relatedSnippetsX).shape)
    print (j)
    print (k)

    #assert np.allclose(((-1)**M)*tonumpyarray(shared_arr), arr_orig)

def init(claimX_, relatedSnippetsX_, j_, k_):
    global shared_claimX
    global shared_relatedSnippetsX
    global j
    global k
    shared_claimX = claimX_
    shared_relatedSnippetsX = relatedSnippetsX_
    j = j_
    k = k_
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

def readFile_synced(i):
    # synchronize access
    with shared_claimX.get_lock() and shared_relatedSnippetsX.get_lock(): 
        readFile(i)

def readFile(i):
    info("start %s" % (i,))

    claimX = tonumpyarray(shared_claimX)
    relatedSnippetsX = tonumpyarray(shared_relatedSnippetsX)


    filePath = filePaths[i]
    if not filePath.endswith('.json'):
        return 
    articles_, sources_ = reader.readGoogle(filePath)
    claim, cred = reader.readSnopes(filePath)

    for article, source in zip(articles_, sources_):
        claimX_ = np.ones((1,3))
        relatedSnippetsX_ = np.ones((2,3))
        #claimX_, relatedSnippetsX_, relatedSnippets_, _ = rsExtractor.extract(claim, article)
        claimX[j.value : j.value+3] = claimX_.reshape(claimX_.size)
        relatedSnippetsX[k.value : k.value+relatedSnippetsX_.size] = relatedSnippetsX_.reshape(relatedSnippetsX_.size)
        j.value += 3; k.value += relatedSnippetsX_.size
        '''
        if (claimX.size == 1):
            claimX = claimX_.reshape(claimX_.size)
            relatedSnippetsX = relatedSnippetsX_.reshape(relatedSnippetsX_.size)
        else:
            claimX = np.append(claimX, claimX_)
            relatedSnippetsX = np.append(relatedSnippetsX, relatedSnippetsX_)
        '''
    print (claimX)

    info("end   %s" % (i,))




if __name__ == '__main__':
    mp.freeze_support()
    main()