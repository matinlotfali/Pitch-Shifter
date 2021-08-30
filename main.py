from scipy.io import wavfile
import numpy as np
from tqdm import trange


def process_chunk(input, shift_amount):
    data = np.fft.rfft(input)                                           # discrete fourier transform (find frequencies)
    data = np.roll(data, -shift_amount)                                 # roll the array (remove shorter frequencies and pretend that remaining frequencies are them)
    data[-shift_amount: len(data)] = 0                                  # add zeros to the end (shorter frequencies are rolled over to the end, let's remove them)
    data = np.fft.irfft(data)                                           # inverse descrete fourier transform
    data = data.astype(input.dtype)                                     # float array to int16
    return data


def main():
    sample_rate, data = wavfile.read('input.wav')  # read a mono wave file

    chunks_count = 100                      # split the wave in chunks (think of it as real-time frames)
    chunk_len = len(data) // chunks_count   # the length of each chunk
    shift_amount = 4000 // chunks_count     # amount of pitch shifting
    result = np.array([], dtype=data.dtype) # output container

    for i in trange(0, chunks_count):                       # show a progress while looping through each chunk
        chunk = data[i * chunk_len: (i + 1) * chunk_len]    # get a chunk
        chunk = process_chunk(chunk, shift_amount)          # calculate the change
        result = np.append(result, chunk)                   # keep it in the result container

    wavfile.write('output.wav', sample_rate, result)   # write a mono wave file


if __name__ == '__main__':
    main()
