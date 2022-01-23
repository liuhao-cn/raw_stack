##############################################################
import os, rawpy, imageio, time, sys
import scipy.fft as fft
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()


##############################################################
# working directory, all raw files in this directory will be processed
working_dir = "/work/astro/temp/"

# define the raw file extension
extension = "CR3"

# specify the raw file data type, 16-bit unsigned integer by default.
raw_data_type = 'uint16'

# If true, do not report the alignment result
less_report = False

# the file of reference frames. If set to None, then use the first frame in the list.
reference_file = 'IMG_0000.CR3'

# fraction of frames that will not be used
bad_fraction = 0.4

# dark_fac means the fraction of pixels that will be ignored due to "too dark".
dark_frac = 5e-4

# bright_frac means the fraction of pixels that will be ignored due to "too bright".
bright_frac = 5e-4

# The red, green and blue pixels are amplified by this factor for a custom white balance.
color_enhance_fac = [1.05, 0.90, 1.05]

# final gamma
gamma = [0.5,0.5,0.5]

# name of the final image
final_file = "final.tiff"

# save aligned binary files or not. Note that for multiprocessing, this must be True
save_aligned_binary = True

# save aligned images?
save_aligned_image = False

# number of ADC digit. The true maximum value should be 2**adc_digit
adc_digit_max = 16


##############################################################
def raw_to_swp():
    tst = time.time()
    for i in range(n_files):
        # read the raw data as an object, obtain the image and compute its fft
        with rawpy.imread(file_lst[i]) as raw:
            raw.raw_image.tofile(file_swp[i])


# subroutine that aligns all frames to the reference frame
def align_frames(ida, frame):
    tst = time.time()
    # read the raw data as an object, obtain the image and compute its fft
    frame_fft = fft.fft2(np.float64(frame))
    # compute the frame offset
    cache = np.abs(fft.ifft2(ref_fft*frame_fft))
    local_index = np.unravel_index(np.argmax(cache, axis=None), cache.shape)
    s1, s2 = -local_index[0], -local_index[1]
    # make sure that the Bayer matrix will not be corrupted
    s1 = s1 - np.mod(s1, 2)
    s2 = s2 - np.mod(s2, 2)
    # fix the offset and save into the result array
    frame_aligned = np.roll(frame, (s1, s2), axis=(0,1))
    # save the aligned images and binaries if necessary
    if save_aligned_binary is True:
        frame_aligned.tofile(file_bin[ida])
    if not less_report:
        print("\nFrame %6i (%s) aligned in %8.2f sec, (sx, sy) = (%8i,%8i)." 
              %(ida, file_lst[ida], time.time()-tst, s1, s2))
    return ida, s1, s2, frame_aligned



import heapq as hp
# subroutine for adjusting the colors 
def adjust_color(i, samp):
    # number of "too dark" pixels and threshold
    samp = samp.reshape(m1*m2)
    ndark = int(npix*dark_frac)
    d1 = hp.nsmallest(ndark, samp)[ndark-1]*1.
    # number of "too bright" pixels and threshold
    nbright = int(npix*bright_frac)
    d2 = hp.nlargest(nbright, samp)[nbright-1]*1.
    # rescaling, note that this requires 16-bit to save weak signal from bright sky-light
    # note that val is expected to be in range [0,1]. Out-of-range values will be truncated.
    val = (samp-d1)/(d2-d1)
    val = np.where(val<0, 0, val)
    val = np.where(val>1, 1, val)
    samp = (val**gamma[i])*256*color_enhance_fac[i]
    samp = np.where(samp<  0,   0, samp)
    samp = np.where(samp>255, 255, samp)
    return samp.reshape(m1, m2)


os.chdir(working_dir)


file_lst, file_bin, file_swp, file_tif = [], [], [], []
if rank==0:
    # make a list of woking files
    for file in os.listdir():
        if file.endswith(extension):
            file_lst.append(file)
            file_swp.append(os.path.splitext(file)[0] + '_aligned.swp')
            file_bin.append(os.path.splitext(file)[0] + '_aligned.bin')
            file_tif.append(os.path.splitext(file)[0] + '_aligned.tif')

file_lst = comm.bcast(file_lst, root=0)
file_bin = comm.bcast(file_bin, root=0)
file_swp = comm.bcast(file_swp, root=0)

n_files = np.int64(len(file_lst))
if nproc > n_files+1:
    nproc = n_files+1

# prepare the reference frame in the Fourier domain and broadcast the frame,
# it fft and shape to all ranks as "ref_frame", "ref_fft", "n1", "n2"
if rank==0:
    if reference_file==None:
        reference_file = file_lst[0]
    ref_raw = rawpy.imread(reference_file)
    ref_frame = ref_raw.raw_image
    frame_shape = np.int64(np.shape(ref_frame))
    ref_fft = np.conjugate(fft.fft2(np.float64(ref_frame)))
else:
    frame_shape = np.int64(np.zeros(2))

comm.Bcast(frame_shape, root=0); comm.Barrier()
n1, n2 = frame_shape[0], frame_shape[1]

if rank!=0:
    ref_frame = np.zeros([n1, n2], dtype=raw_data_type)
    ref_fft = np.zeros([n1, n2], dtype='complex128')

comm.Bcast(ref_frame, root=0); comm.Barrier()
comm.Bcast(ref_fft,   root=0); comm.Barrier()


# read all frames in the root rank
if rank==0:
    tst = time.time()
    frames = np.zeros([n_files, n1, n2], dtype=raw_data_type)
    for i in range(n_files):
        with rawpy.imread(file_lst[i]) as raw:
            frames[i,:,:] = raw.raw_image.copy()
    print("Data file read by rank 0 in %8.2f sec." %(time.time()-tst))


RANK_IDLE = 1
RANK_BUSY = 2
TAG_ROOT2WORKER_INFO = 13
TAG_ROOT2WORKER_DATA = 17
TAG_WORKER2ROOT_INFO = 19
TAG_WORKER2ROOT_DATA = 23

# root2worker_info contains: job_current, exit flag
# root2worker_data contains an input frame
root2worker_info = np.zeros(2, dtype="int32")
root2worker_data = np.zeros([n1, n2], dtype=raw_data_type)

# worker2root_info contains: job_current, shift 1, shift 2
# worker2root_data contains an aligned frame
worker2root_info = np.zeros(3, dtype="int32")
worker2root_data = np.zeros([n1, n2], dtype=raw_data_type)

if rank==0:
    # initialize the rank status
    rank_status = np.zeros(nproc, dtype="int32") + RANK_IDLE
    rank_status[0] = RANK_BUSY

    rank_on_job = np.zeros(nproc, dtype="int32")

    # initialize the job status
    job_status = np.zeros(n_files)

    # "job_current" saves the current job that needs to be done (rise monotonically)
    job_current = 0
    
    # make a list of requests to be used to receive results from worker
    reqs_info, reqs_data = [], []
    for i in range(n_files):
        reqs_info.append(None)
        reqs_data.append(None)
    
    # prepare the array for aligned frames and shifts
    frames_aligned = np.zeros([n_files, n1, n2], dtype=np.float64)
    sx = np.zeros(n_files, dtype="int32")
    sy = sx.copy()

    # exit while only when all jobs are done
    while (1):
        # go through all ranks except root, if the rank (worker) is idle,
        # then assign a job to it and make a request to receive the result
        # (no waiting).
        for i in range(1, nproc):
            if (rank_status[i]==RANK_IDLE) and (job_current<n_files):
                print(job_current)
                # send a package to worker with the job number and frame
                # to be aligned
                root2worker_info[0] = job_current
                root2worker_info[1] = 0
                root2worker_data[:,:] = frames[job_current,:,:]

                tmp1 = comm.Isend(root2worker_info, dest=i, tag=TAG_ROOT2WORKER_INFO*i)
                tmp2 = comm.Isend(root2worker_data, dest=i, tag=TAG_ROOT2WORKER_DATA*i)

                # print("Frame %i sent (async) to rank %i." %(job_current, i))
                # start to receive the result from the worker (no wait).
                reqs_info[job_current] = comm.Irecv(worker2root_info, source=i, tag=TAG_WORKER2ROOT_INFO*i)
                reqs_data[job_current] = comm.Irecv(worker2root_data, source=i, tag=TAG_WORKER2ROOT_DATA*i)

                # mark the worker as busy.
                rank_status[i] = RANK_BUSY
                rank_on_job[i] = job_current
                job_current = job_current + 1

        # wait for 50 ms before checking the result of receiving
        # print(rank_on_job); sys.exit()
        time.sleep(6)

        # check if anything was received.
        for i in range(1, nproc):
            j = rank_on_job[i]
            if reqs_info[j] != None and reqs_data[j] != None and j>=0:
                if reqs_info[j].Test():
                    if reqs_data[j].Test():
                        print("check rank on job:", i, j)
                        reqs_info[j], reqs_data[j] = None, None
                        # print("reqs_info after test", i, reqs_info[i])
                        # reqs_info[i].Wait()
                        # print("reqs_info after wait", i, reqs_info[i])
                        # stat, _ = reqs_data[i].test()
                        # if stat:
                        #     reqs_data[i].Wait(); 
                        #     # save results
                        #     # print(worker2root_info)
                        index = worker2root_info[0]
                        print("test of root receival consistency", index, i, j)
                        time.sleep(0.5)

                        sx[index], sy[index] = worker2root_info[1], worker2root_info[2]
                        # print(i, index, sx[index], sy[index], job_current)
                        frames_aligned[index,:,:] = worker2root_data
                        # set the worker to idle and the request to None.
                        rank_status[i] = RANK_IDLE
                        job_status[index] = 1
                        rank_on_job[i] = -1

        print("root status:", np.sum(job_status))
        # print(rank, "Get results from rank %i" %(i)); sys.exit()
        # again sleep for 50 ms.
        time.sleep(0.1)
        if np.sum(job_status)==n_files:
            break

    for i in range(1, nproc):
        root2worker_info[:] = 1
        req_tmp1 = comm.Isend(root2worker_info, dest=i, tag=TAG_ROOT2WORKER_INFO)
    # print("Stop signal sent to all workers"); sys.exit()
else:
    # get the package from root. note that we should ignore ranks higher than
    # nproc because they receive no job.
    while(1):
        req1 = comm.Irecv(root2worker_info, source=0, tag=TAG_ROOT2WORKER_INFO*rank)
        req1.Wait()
        if root2worker_info[1] == 1:
            break
        else:
            req2 = comm.Irecv(root2worker_data, source=0, tag=TAG_ROOT2WORKER_DATA*rank)
            req2.Wait()

            worker2root_info[0], worker2root_info[1], worker2root_info[2], worker2root_data[:,:] = \
                align_frames(root2worker_info[0], root2worker_data)

            req3 = comm.Isend(worker2root_info, dest=0, tag=TAG_WORKER2ROOT_INFO*rank)
            req3.Wait()

            req4 = comm.Isend(worker2root_data, dest=0, tag=TAG_WORKER2ROOT_DATA*rank)
            req4.Wait()

            print(rank, "worker sent result: ", worker2root_info, np.sum(root2worker_data))

        time.sleep(0.05)

    # if rank<nproc:
    #     working = True
    #     while working:
    #         req1 = comm.Irecv(root2worker_info, source=0, tag=TAG_ROOT2WORKER_INFO); req1.Wait()#; req1.Free()
    #         if root2worker_info[1] == 1:
    #             working = False
    #         else:
    #             req2 = comm.Irecv(root2worker_data, source=0, tag=TAG_ROOT2WORKER_DATA); req2.Wait()#; req2.Free()
    #             worker2root_info[0], worker2root_info[1], worker2root_info[2], worker2root_data[:] = \
    #                 align_frames(root2worker_info[0], root2worker_data.copy())
    #             req1 = comm.Isend(worker2root_info, dest=0, tag=TAG_WORKER2ROOT_INFO); req1.Wait()#; req1.Free()
    #             req2 = comm.Isend(worker2root_data, dest=0, tag=TAG_WORKER2ROOT_DATA); req2.Wait()#; req2.Free()
    #             print(rank, "worker sent result: ", worker2root_info, np.sum(root2worker_data))

    #         time.sleep(0.05)

comm.barrier()

if rank==0:
    print(sx, sy)

sys.exit()


if rank==0:
    tst = time.time()
    
    output_arr = np.array(output)
    sx, sy = output_arr[0,:,1], output_arr[0,:,2]
    sx = np.where(sx >  n1/2, sx-n1, sx)
    sx = np.where(sx < -n1/2, sx+n1, sx)
    sy = np.where(sy >  n2/2, sy-n2, sy)
    sy = np.where(sy < -n2/2, sy+n2, sy)
    
    # for i in range(n_files):
    #     align_frames(i)

    print("Alignment done. Time cost: %9.2f" %(time.time()-tst))

    # read the alignment results of multiple processes
    for i in range(n_files):
        frame = np.fromfile(file_bin[i],dtype=raw_data_type)
        frames_aligned[i,:,:] = frame.reshape(n1, n2)



    # stack the frames with weights computed from the summed cross
    # covariance between frames (ignore auto covariance)
    tst = time.time()
    for i in range(0, n_files):
        frames_aligned[i,:,:] = frames_aligned[i,:,:] - np.median(frames_aligned[i,:,:])
    
    # compute the covariance matrix
    frames_aligned = frames_aligned.reshape(n_files, n1*n2)
    cov = np.dot(frames_aligned, frames_aligned.transpose())
    
    # compute weights from the covariance matrix
    aa = np.diag(cov)[0:n_files]
    bb = np.sum(cov, axis=1)[0:n_files]
    w = np.zeros(n_files)
    for i in range(n_files):
        w[i] = bb[i]/aa[i] - 1

    # exclude the low quality frames
    n_bad = int(n_files*bad_fraction)
    thr = hp.nsmallest(n_bad, w)[n_bad-1]
    if thr < 0: thr = 0
    w = np.where(w <= thr, 0, w)
    w = w / np.sum(w)
    
    # plot the weights for test 
    plt.figure(figsize=(4,2),dpi=200)
    plt.title(r'Stacking weights ($w\times N_{frames}$)')
    plt.xlabel('Frame number',fontsize=9)
    plt.ylabel(r'$w\times N_{frames}$',fontsize=9)
    w1 = np.where(w==0, np.nan, w)
    w2 = np.where(w==0, np.median(w), np.nan)
    plt.plot(w1*n_files, marker="o", label='Valid')
    plt.plot(w2*n_files, marker="*", label='Invaid')
    plt.legend()
    plt.savefig('weights.pdf')
    
    # plot the XY-shifts
    plt.figure(figsize=(4,2),dpi=200)
    plt.title('XY shifts in pixel')
    plt.xlabel('Y shifts',fontsize=9)
    plt.ylabel('X shifts',fontsize=9)
    plt.scatter(sx, sy, s=50, alpha=.5)
    plt.savefig('xy-shifts.pdf')

    # stack the frames with weights.
    # note that we must normalize the stacked result.
    frame_stacked = np.dot(w, frames_aligned).reshape(n1, n2)
    fmin = np.amin(frame_stacked)
    fmax = np.amax(frame_stacked)
    cache = (frame_stacked-fmin)/(fmax-fmin)
    tmax = 2.**(adc_digit_max) - 1.
    frame_stacked = np.floor(cache*tmax)
    print("Weights computed and %i/%i best frames used for stacking. Time cost: %6.2f" 
        %(n_files-n_bad, n_files, time.time()-tst))



    # adjust the color and prepare for 8-bit output, do it on linear rgb so that we still have 
    # linear response, but don't have to worry about the Bayer matrix.
    # note that "gamma=(1,1), no_auto_bright=True" means to get linear rgb
    # rememer to use output_bps=16 for 16-bit output depth
    print("Ready to adjust the colors:")
    ref_raw.raw_image[:,:] = frame_stacked
    rgb = ref_raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
    # note that the image size is different after converting from raw to rgb image
    m1, m2, npix = np.shape(rgb)[0], np.shape(rgb)[1], rgb.size
    tst = time.time()
    for i in range(3):
        rgb[:,:,i] = adjust_color(i, rgb[:,:,i])
        print("Color adjusted for color channel %i, time cost: %8.2f" %(i, time.time()-tst))
        tst = time.time()

    # save the final figure
    imageio.imsave(final_file, np.uint8(rgb))

    # show the final figure
    plt.figure(figsize=(6,4),dpi=200)
    plt.xlabel('Y',fontsize=12)
    plt.ylabel('X',fontsize=12)
    plt.imshow(np.uint8(rgb))

    print("Done!")
    
    
