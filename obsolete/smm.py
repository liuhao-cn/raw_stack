import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

def pow2(i):
    # for convenience, make a local version of the shared memory
    local_var = np.frombuffer(shm_a.buf, dtype=np.float32).reshape(n1, n2)
    
    # modify the shared memory, which can be seen elsewhere
    local_var[i,:] = i**2
    return i, 2*i

n1 = 7
n2 = 6
data_type = "float32"

nproc = 4

# start the memory manager
smm = SharedMemoryManager()
smm.start()

n_elements = n1 * n2
size_in_byte = n_elements * np.dtype(data_type).itemsize


if __name__ == '__main__':
    # initialize a block of shared memory names shm_a. The name "shm_a" can be
    # used in other processes directly.
    shm_a = smm.SharedMemory(size=size_in_byte)
    
    # for convenience, make a local version of the shared memory
    shm_a_as_np = np.frombuffer(shm_a.buf, dtype=data_type).reshape(n1, n2)
    
    # use mp.Pool to set the content of the shared memory in parallel
    with mp.Pool(nproc) as pool:
        output = pool.map(pow2, range(n1))

# check the content
print(shm_a_as_np)

# check the output of the subroutine
print("---")
print(output)

# shutdown the shared memory
smm.shutdown()
