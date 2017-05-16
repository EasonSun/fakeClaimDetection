from multiprocessing import Pool, Array, Manager, Process
import numpy as np

unshared_arr = np.ones(3)
a = Array('d', unshared_arr)
#print "Originally, the first two elements of arr = %s"%(a[:2])
global b
b = np.frombuffer(a.get_obj())

def f(obj):
    #a[0] = -a[0]
    b=obj[1]
    b = np.append(b, np.ones(3))
    print (b.shape)

if __name__ == '__main__':
    # Create the array
    shared_arr = mp.Array(ctypes.c_double, 3)
    arr = np.frombuffer(shared_arr.get_obj())
    # Create, start, and finish the child process
    pool = Pool(processes=4)
    pool.map(f, [(i,b) for i in range(5)])
    '''
    p = Process(target=f, args=(b,))
    p.start()
    p.join()
    '''
    # Print out the changed values
    # print "Now, the first two elements of arr = %s"%a[:2]


    print (b.shape)