import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from numpy import genfromtxt
import matplotlib.pyplot as plt
from util import moving_average,vector_magnitude,mode1,getName
from sklearn.ensemble import RandomForestClassifier
from scipy.io import wavfile
from util import moving_average,vector_magnitude,mode1,getName
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler




###


from scipy.io import wavfile
from scipy import signal
import scipy
import numpy as np


def get_sample_rate(audio_file):
    rate, data = wavfile.read(audio_file)
    return rate


def analyze(audio_file):
    downscaling = 11
    rate, data = wavfile.read(audio_file)
    time = calculate_time(len(data), rate)
    data = signal.decimate(data, downscaling)
    time = signal.decimate(time[:], downscaling)
    spectrogram = signal.spectrogram(data, fs=rate, mode='psd')
    return data.tolist(), time.tolist(), spectrogram


def calculate_time(number_of_samples, sample_rate):
    time = np.arange(0, number_of_samples)
    dT = 1.0 / sample_rate
    time = np.dot(time, dT)
    return time


import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np

sample_rate, samples = wavfile.read('1.wav') # ./output/audio.wav
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, np.log(spectrogram))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
##

###
sample_rate= 10
window_size=sample_rate
treeLength=100
percent_overlap=0.5
moveavg=100
max_depth=10
###
def window(arr, w=window_size, window=int(window_size*percent_overlap), copy=False):

    nparray = np.asarray(arr)
    view = np.lib.stride_tricks.as_strided(nparray, strides=nparray.strides * 2, shape=(nparray.size - w + 1, w))[0::window]
    if copy:
        return view.copy()
    else:
        return view

imu_data = genfromtxt('1.2.csv', delimiter=',')

accelermoter_data = imu_data[:,1:4]
gyroscope_data = imu_data[:,4:7]


magData_acc = vector_magnitude(accelermoter_data)
magData_gyr = vector_magnitude(gyroscope_data)

windowed_acc=window(magData_acc)
windowed_gyr=window(magData_gyr)

zeros_acc=np.zeros(windowed_acc.shape)
zeros_gyr=np.zeros(windowed_gyr.shape)

## PCA part
standard_imu = StandardScaler().fit_transform((imu_data[:,1:7]))
pca = PCA(n_components=4)
pca_transformed_imu=pca.fit_transform(standard_imu)

num=0
for acc,gyr in zip(windowed_acc,windowed_gyr):
    sample_acc = acc
    sample_gyr = gyr

    sample_edf_acc = ECDF(sample_acc)
    sample_edf_gyr = ECDF(sample_gyr)

    slope_changes_acc = sorted(set(sample_acc))
    slope_changes_gyr = sorted(set(sample_gyr))

    sample_edf_values_at_slope_changes_acc = [ sample_edf_acc(item) for item in slope_changes_acc]
    inverted_edf_acc = interp1d(sample_edf_values_at_slope_changes_acc, slope_changes_acc)

    sample_edf_values_at_slope_changes_gyr = [ sample_edf_gyr(item) for item in slope_changes_gyr]
    inverted_edf_gyr = interp1d(sample_edf_values_at_slope_changes_gyr, slope_changes_gyr)


    even_spaced_q_acc=np.linspace(min(sample_edf_values_at_slope_changes_acc),1,num=window_size)
    zeros_acc[num]=inverted_edf_acc(even_spaced_q_acc)

    even_spaced_q_gyr=np.linspace(min(sample_edf_values_at_slope_changes_gyr),1,num=window_size)
    zeros_gyr[num]=inverted_edf_gyr(even_spaced_q_gyr)
    num+=1

classifier = RandomForestClassifier(n_estimators = treeLength, criterion= "entropy", random_state= 0,verbose=True,warm_start=True,max_depth=max_depth)
test_label= np.ones(zeros_gyr.shape[0])
classifier.fit(zeros_gyr,test_label)
classifier.n_estimators+=100
classifier.fit(zeros_acc,test_label)
print('')