import numpy as np
import pandas as pd
import pylab as pl
from random import seed, random
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import rcParams
rcParams['figure.figsize'] = 30,10

# Cut points are defined globally
cpoints = {'3' : [-0.43, 0.43],
           '4' : [-0.67, 0, 0.67],
           '5' : [-0.84, -0.25, 0.25, 0.84],
           '6' : [-0.97, -0.43, 0, 0.43, 0.97],
           '7' : [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
           '8' : [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
           '9' : [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
           '10': [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28],
           '11': [-1.34, -0.91, -0.6, -0.35, -0.11, 0.11, 0.35, 0.6, 0.91, 1.34],
           '12': [-1.38, -0.97, -0.67, -0.43, -0.21, 0, 0.21, 0.43, 0.67, 0.97, 1.38],
           '13': [-1.43, -1.02, -0.74, -0.5, -0.29, -0.1, 0.1, 0.29, 0.5, 0.74, 1.02, 1.43],
           '14': [-1.47, -1.07, -0.79, -0.57, -0.37, -0.18, 0, 0.18, 0.37, 0.57, 0.79, 1.07, 1.47],
           '15': [-1.5, -1.11, -0.84, -0.62, -0.43, -0.25, -0.08, 0.08, 0.25, 0.43, 0.62, 0.84, 1.11, 1.5],
           '16': [-1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53],
           '17': [-1.56, -1.19, -0.93, -0.72, -0.54, -0.38, -0.22, -0.07, 0.07, 0.22, 0.38, 0.54, 0.72, 0.93, 1.19, 1.56],
           '18': [-1.59, -1.22, -0.97, -0.76, -0.59, -0.43, -0.28, -0.14, 0, 0.14, 0.28, 0.43, 0.59, 0.76, 0.97, 1.22, 1.59],
           '19': [-1.62, -1.25, -1, -0.8, -0.63, -0.48, -0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.48, 0.63, 0.8, 1, 1.25, 1.62],
           '20': [-1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.25, 0.39, 0.52, 0.67, 0.8, 1.04, 1.28, 1.64]
            }

# normalize sensor data
def znormalization(ts):
    # ts: sensor data. np.ndarray
    mus = ts.mean(axis=0)
    stds = ts.std(axis=0)
    return (ts - mus)/stds

def paa_transform(ts,n):
    # ts: timeseries data(normalized)
    # n: number of chunks
    means = list()
    splits = np.array_split(ts,n)

    for split in range(len(splits)):
        means.append((splits[split].mean(),len(splits[split])))
    #means = np.repeat(means,len(splits[0]))
    return means

def sax_transform(ts,n,alphabet):
    # ts: time series(raw)
    # n: number of chunks
    # alphabet: alphabet to be used for transformation - string(must be competible with n)
    # First normalize time series such a way that it has zero mean and unit standard deviation
    norm_ts = znormalization(ts)
    # PAA transformation of normalized time series
    paa_nts = paa_transform(norm_ts,n)
    paa = np.asarray(list())
    for mean,split in paa_nts:
        paa = np.append(paa,np.repeat(mean,split))
    #print('paa: {}'.format(paa_nts))

    # Compute cut points for SAX transformation
    #cpoints = norm.ppf(np.linspace(1./len(alphabet),1-1./len(alphabet),len(alphabet)-1))
    # or use them predefined

    first_letter = 'a'
    # SAX transformation
    # choose cut value based on alphabet size
    cut = cpoints[str(alphabet)]
    #saxts = np.asarray([(alphabet[0] if value < cpoints[0]
    #                     else (alphabet[-1] if value > cpoints[-1]
    #                           else alphabet[np.where(cpoints <= value)[0][-1] + 1]))
    #                        for value in paa_nts])
    saxts = list()
    for i in range(0,len(paa)):
        found = False
        for j in range(len(cut)):
            if paa[i] < cut[j]:
                saxts.append(chr(ord(first_letter) + j))
                found = True
                break
        if not found:
            saxts.append(chr(ord(first_letter) + len(cut)))
    saxts_df = pd.DataFrame(saxts,index=norm_ts.index)
    paa_df = pd.DataFrame(paa,index=norm_ts.index)

    #plt.plot(norm_ts.ix[0:500],alpha=0.6,label='Normalized ts')
    #plt.plot(paa_df.ix[0:500],'r--',alpha=0.5,label='PAA Transformation')
    #plt.plot(saxts_df.ix[0:500],'g--',label='SAX Transformation')
    #plt.legend(loc='best')
    #plt.show()
    return saxts

def letter_dict(alphaSize):
    # Compute letter vise distances
    number_rep = range(alphaSize)
    letters = [chr(x + ord('a')) for x in number_rep]
    compareDict = {}
    cuts = cpoints[str(alphaSize)]
    for i in range(0, len(letters)):
        for j in range(0, len(letters)):
            if np.abs(number_rep[i]-number_rep[j]) == 0:
                compareDict[letters[i]+letters[j]] = 0
            else:
                compareDict[letters[i]+letters[j]] = np.abs(number_rep[i]-number_rep[j])
    return compareDict

def compareSeq(trainSeq, newSeq,dict):

    if len(trainSeq) != len(newSeq):
        print('Strings must be in the same length')
        return -1
    list_letters_a = [x for x in trainSeq]
    list_letters_b = [x for x in newSeq]
    dist = 0.0
    for i in range(0, len(list_letters_a)):
        dist += dict[list_letters_a[i]+list_letters_b[i]]**2
    return dist

def generate_ngram(seq, n=3):
    # Generate n-grams
    ngrams = list()
    for i in range(len(seq)-(n-1)):
        ngrams.append(seq[i:i+n])
    #print('n-grams:\n{}'.format(ngrams))
    return ngrams


ts = pd.read_csv('..\dataset\BATADAL_dataset03.csv')
ts_test = pd.read_csv('..\dataset\BATADAL_dataset04.csv')
#print(ts.head())
#print(ts_test.head())
ts.index = pd.DatetimeIndex(start='20140106T00',end='20150106T00',freq='1H')
ts_test.index = pd.DatetimeIndex(start='20160704T00',end='20161225T00',freq='1H')

# Generate a look up dictionary for letters to compare different sequences
dict = letter_dict(20)

# n-grams
n = 3

# Sensor T1
ts_sax = sax_transform(ts['L_T1'],500,5)
ts_test_sax = sax_transform(ts_test[' L_T1'],500,5)

# Start reading data
test_reading = ts_test_sax[0:n]
prediction = list()
#print(test_reading)
# Generate n-grams for train dataset(3-gram default)
ts_ngram = generate_ngram(ts_sax,n)

#thresh = range(20,40) -- used for threshold selection
dists = list()
#perf = list()
#for th in thresh:
th = 20
tp = tn = fn = fp = 0
idx = -1
for i in range(len(ts_test_sax)-n):
    #print(test_reading[i:i + n])
    if idx == -1:
        for ngram in ts_ngram:
            dists.append(compareSeq(ngram,test_reading[i:i+n],dict))
        # Find the closest match
        idx = dists.index(min(dists))
        test_reading.append(ts_test_sax[i + n])
    else:
        # compare sequences
        dist = compareSeq(ts_ngram[idx],test_reading[i:i+n],dict)
        if dist > th:
            # if test sequence is too different from the next trained ngram raise an alarm
            prediction.append(1)
            if ts_test[' ATT_FLAG'][i] == 1:
                tp += 1
            if ts_test[' ATT_FLAG'][i] == -999:
                fp += 1
        else:
            prediction.append(-999)
            if ts_test[' ATT_FLAG'][i] == 1:
                tn += 1
            if ts_test[' ATT_FLAG'][i] == -999:
                fn += 1
        test_reading.append(ts_test_sax[i + n])
        idx += 1
print('---Sensor T1---\nTP: {}\tTN: {}\tFP: {}\tFN: {}'.format(tp,tn,fp,fn))
# used for threshold selection
#perf.append((th, tp / fp,prediction))
#best = perf.index(max(perf, key = lambda t: t[1]))
#thresh = perf[best][0]
#print('---Sensor T1---\nBest perf: {}\n Threshold: {}'.format(perf[1],perf[best][0]))

# plot predictions
prediction = pd.DataFrame(prediction,index=ts_test.index[n+1:])
fig ,((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4,2,sharex=False,sharey=False)
ax1.plot(prediction,label='Sensor T1',alpha=0.7)
ax1.plot(ts_test[' ATT_FLAG'],'r--',label='Actual Attacks')
ax1.legend(loc='best')


# Sensor T2
ts_sax = sax_transform(ts['L_T2'],700,6)
ts_test_sax = sax_transform(ts_test[' L_T2'],700,6)

test_reading = ts_test_sax[0:n]
prediction = list()
#print(test_reading)
# Generate n-grams for train dataset(3-gram default)
ts_ngram = generate_ngram(ts_sax,n)

#thresh = range(36,37)
dists = list()
perf = list()
th = 36
prediction = []
tp = tn = fn = fp = 0
#for th in thresh:
idx = -1
for i in range(len(ts_test_sax)-n):
    #print(test_reading[i:i + n])
    if idx == -1:
        for ngram in ts_ngram:
            dists.append(compareSeq(ngram,test_reading[i:i+n],dict))
            # Find the closest match
        idx = dists.index(min(dists))
        test_reading.append(ts_test_sax[i + n])
    else:
        dist = compareSeq(ts_ngram[idx],test_reading[i:i+n],dict)
        #print(dist)
        if dist > th:
            # if test sequence is too different from the next trained ngram raise an alarm
            prediction.append(1)
            if ts_test[' ATT_FLAG'][i] == 1:
                tp += 1
            if ts_test[' ATT_FLAG'][i] == -999:
                fp += 1
        else:
            prediction.append(-999)
            if ts_test[' ATT_FLAG'][i] == 1:
                tn += 1
            if ts_test[' ATT_FLAG'][i] == -999:
                fn += 1
        test_reading.append(ts_test_sax[i + n])
        idx += 1
print('---Sensor T2---\nTP: {}\tTN: {}\tFP: {}\tFN: {}'.format(tp, tn, fp, fn))
#perf.append((th, tp / fp,prediction))
#best = perf.index(max(perf, key = lambda t: t[1]))
#thresh = perf[best][0]
#print('---Sensor T2---\nBest perf: {}\n Threshold: {}'.format(perf[best][1],perf[best][0]))
prediction = pd.DataFrame(prediction,index=ts_test.index[n+1:])
prediction = pd.DataFrame(prediction,index=ts_test.index[n+1:])
ax2.plot(prediction,label='Sensor T2',alpha=0.7)
ax2.plot(ts_test[' ATT_FLAG'],'r--',label='Actual Attacks')
ax2.legend(loc='best')

# Sensor T3
ts_sax = sax_transform(ts['L_T3'],800,6)
ts_test_sax = sax_transform(ts_test[' L_T3'],800,6)

test_reading = ts_test_sax[0:n]
prediction = list()
#print(test_reading)
# Generate n-grams for train dataset(3-gram default)
ts_ngram = generate_ngram(ts_sax,n)

#thresh = range(19,20)
dists = list()
perf = list()
th = 19
prediction = []
#for th in thresh:
tp = tn = fn = fp = 0
idx = -1
for i in range(len(ts_test_sax)-n):
    #print(test_reading[i:i + n])
    if idx == -1:
        for ngram in ts_ngram:
            dists.append(compareSeq(ngram,test_reading[i:i+n],dict))
            # Find the closest match
        idx = dists.index(min(dists))
        test_reading.append(ts_test_sax[i + n])
    else:
        dist = compareSeq(ts_ngram[idx],test_reading[i:i+n],dict)
        if dist > th:
            # if test sequence is too different from the next trained ngram raise an alarm
            prediction.append(1)
            if ts_test[' ATT_FLAG'][i] == 1:
                tp += 1
            if ts_test[' ATT_FLAG'][i] == -999:
                fp += 1
        else:
            prediction.append(-999)
            if ts_test[' ATT_FLAG'][i] == 1:
                tn += 1
            if ts_test[' ATT_FLAG'][i] == -999:
                fn += 1
        test_reading.append(ts_test_sax[i + n])
        idx += 1
print('---Sensor T3---\nTP: {}\tTN: {}\tFP: {}\tFN: {}'.format(tp, tn, fp, fn))
#perf.append((th, tp / fp,prediction))
#best = perf.index(max(perf, key = lambda t: t[1]))
#thresh = perf[best][0]
#print('---Sensor T3---\nBest perf: {}\n Threshold: {}'.format(perf[best][1],perf[best][0]))
prediction = pd.DataFrame(prediction,index=ts_test.index[n+1:])
prediction = pd.DataFrame(prediction,index=ts_test.index[n+1:])
ax3.plot(prediction,label='Sensor T3',alpha=0.7)
ax3.plot(ts_test[' ATT_FLAG'],'r--',label='Actual Attacks')
ax3.legend(loc='best')


# Sensor T4
ts_sax = sax_transform(ts['L_T4'],500,6)
ts_test_sax = sax_transform(ts_test[' L_T4'],500,6)

test_reading = ts_test_sax[0:n]
prediction = list()
#print(test_reading)
# Generate n-grams for train dataset(3-gram default)
ts_ngram = generate_ngram(ts_sax,n)

thresh = range(19,20)
dists = list()
perf = list()
th = 19
prediction = []
#for th in thresh:
tp = tn = fn = fp = 0
idx = -1
for i in range(len(ts_test_sax)-n):
    #print(test_reading[i:i + n])
    if idx == -1:
        for ngram in ts_ngram:
            dists.append(compareSeq(ngram,test_reading[i:i+n],dict))
        # Find the closest match
        idx = dists.index(min(dists))
        test_reading.append(ts_test_sax[i + n])
    else:
        dist = compareSeq(ts_ngram[idx],test_reading[i:i+n],dict)
        #print(dist)
        if dist > th:
            # if test sequence is too different from the next trained ngram raise an alarm
            prediction.append(1)
            if ts_test[' ATT_FLAG'][i] == 1:
                tp += 1
            if ts_test[' ATT_FLAG'][i] == -999:
                fp += 1
        else:
            prediction.append(-999)
            if ts_test[' ATT_FLAG'][i] == 1:
                tn += 1
            if ts_test[' ATT_FLAG'][i] == -999:
                fn += 1
        test_reading.append(ts_test_sax[i + n])
        idx += 1
print('---Sensor T4---\nTP: {}\tTN: {}\tFP: {}\tFN: {}'.format(tp, tn, fp, fn))
#perf.append((th, tp / fp,prediction))
#best = perf.index(max(perf, key = lambda t: t[1]))
#thresh = perf[best][0]
#print('---Sensor T4---\nBest perf: {}\n Threshold: {}'.format(perf[best][1],perf[best][0]))
prediction = pd.DataFrame(prediction,index=ts_test.index[n+1:])
ax4.plot(prediction,label='Sensor T4',alpha=0.7)
ax4.plot(ts_test[' ATT_FLAG'],'r--',label='Actual Attacks')
ax4.legend(loc='best')


# Sensor T5
ts_sax = sax_transform(ts['L_T5'],600,6)
ts_test_sax = sax_transform(ts_test[' L_T5'],600,6)

test_reading = ts_test_sax[0:n]
prediction = list()
#print(test_reading)
# Generate n-grams for train dataset(3-gram default)
ts_ngram = generate_ngram(ts_sax,n)

thresh = range(19,20)
dists = list()
perf = list()
th = 19
prediction = []
#for th in thresh:
tp = tn = fn = fp = 0
idx = -1
for i in range(len(ts_test_sax)-n):
    #print(test_reading[i:i + n])
    if idx == -1:
        for ngram in ts_ngram:
            dists.append(compareSeq(ngram,test_reading[i:i+n],dict))
        # Find the closest match
        idx = dists.index(min(dists))
        test_reading.append(ts_test_sax[i + n])
    else:
        dist = compareSeq(ts_ngram[idx],test_reading[i:i+n],dict)
        #print(dist)
        if dist > th:
            # if test sequence is too different from the next trained ngram raise an alarm
            prediction.append(1)
            if ts_test[' ATT_FLAG'][i] == 1:
                tp += 1
            if ts_test[' ATT_FLAG'][i] == -999:
                fp += 1
        else:
            prediction.append(-999)
            if ts_test[' ATT_FLAG'][i] == 1:
                tn += 1
            if ts_test[' ATT_FLAG'][i] == -999:
                fn += 1
        test_reading.append(ts_test_sax[i + n])
        idx += 1
print('---Sensor T5---\nTP: {}\tTN: {}\tFP: {}\tFN: {}'.format(tp, tn, fp, fn))
#perf.append((th, tp / fp,prediction))
#best = perf.index(max(perf, key = lambda t: t[1]))
#thresh = perf[best][0]
#print('---Sensor T5---\nBest perf: {}\n Threshold: {}'.format(perf[best][1],perf[best][0]))
prediction = pd.DataFrame(prediction,index=ts_test.index[n+1:])
ax5.plot(prediction,label='Sensor T5',alpha=0.7)
ax5.plot(ts_test[' ATT_FLAG'],'r--',label='Actual Attacks')
ax5.legend(loc='best')


# Sensor T6
ts_sax = sax_transform(ts['L_T6'],500,5)
ts_test_sax = sax_transform(ts_test[' L_T6'],500,5)

test_reading = ts_test_sax[0:n]
prediction = list()
#print(test_reading)
# Generate n-grams for train dataset(3-gram default)
ts_ngram = generate_ngram(ts_sax,n)

#thresh = range(11,12)
dists = list()
perf = list()
th = 11
prediction = []
#for th in thresh:
tp = tn = fn = fp = 0
idx = -1
for i in range(len(ts_test_sax)-n):
    #print(test_reading[i:i + n])
    if idx == -1:
        for ngram in ts_ngram:
            dists.append(compareSeq(ngram,test_reading[i:i+n],dict))
        # Find the closest match
        idx = dists.index(min(dists))
        test_reading.append(ts_test_sax[i + n])
    else:
        dist = compareSeq(ts_ngram[idx],test_reading[i:i+n],dict)
        #print(dist)
        if dist > th:
            # if test sequence is too different from the next trained ngram raise an alarm
            prediction.append(1)
            if ts_test[' ATT_FLAG'][i] == 1:
                tp += 1
            if ts_test[' ATT_FLAG'][i] == -999:
                fp += 1
        else:
            prediction.append(-999)
            if ts_test[' ATT_FLAG'][i] == 1:
                tn += 1
            if ts_test[' ATT_FLAG'][i] == -999:
                fn += 1
        test_reading.append(ts_test_sax[i + n])
        idx += 1
print('---Sensor T6---\nTP: {}\tTN: {}\tFP: {}\tFN: {}'.format(tp, tn, fp, fn))
#perf.append((th, tp / fp,prediction))
#best = perf.index(max(perf, key = lambda t: t[1]))
#thresh = perf[best][0]
#print('---Sensor T6---\nBest perf: {}\n Threshold: {}'.format(perf[best][1],perf[best][0]))
prediction = pd.DataFrame(prediction,index=ts_test.index[n+1:])
ax6.plot(prediction,label='Sensor T6',alpha=0.7)
ax6.plot(ts_test[' ATT_FLAG'],'r--',label='Actual Attacks')
ax6.legend(loc='best')



# Sensor T7
ts_sax = sax_transform(ts['L_T7'],600,5)
ts_test_sax = sax_transform(ts_test[' L_T7'],600,5)

test_reading = ts_test_sax[0:n]
prediction = list()
#print(test_reading)
# Generate n-grams for train dataset(3-gram default)
ts_ngram = generate_ngram(ts_sax,n)

#thresh = range(11,12)
dists = list()
perf = list()
th = 11
prediction = []
#for th in thresh:
tp = tn = fn = fp = 0
idx = -1
for i in range(len(ts_test_sax)-n):
    #print(test_reading[i:i + n])
    if idx == -1:
        for ngram in ts_ngram:
            dists.append(compareSeq(ngram,test_reading[i:i+n],dict))
        # Find the closest match
        idx = dists.index(min(dists))
        test_reading.append(ts_test_sax[i + n])
    else:
        dist = compareSeq(ts_ngram[idx],test_reading[i:i+n],dict)
        #print(dist)
        if dist > th:
            # if test sequence is too different from the next trained ngram raise an alarm
            prediction.append(1)
            if ts_test[' ATT_FLAG'][i] == 1:
                tp += 1
            if ts_test[' ATT_FLAG'][i] == -999:
                fp += 1
        else:
            prediction.append(-999)
            if ts_test[' ATT_FLAG'][i] == 1:
                tn += 1
            if ts_test[' ATT_FLAG'][i] == -999:
                fn += 1
        test_reading.append(ts_test_sax[i + n])
        idx += 1
print('---Sensor T1---\nTP: {}\tTN: {}\tFP: {}\tFN: {}'.format(tp, tn, fp, fn))
#perf.append((th, tp / fp,prediction))
#best = perf.index(max(perf, key = lambda t: t[1]))
#thresh = perf[best][0]
#print('---Sensor T7---\nBest perf: {}\n Threshold: {}'.format(perf[best][1],perf[best][0]))
prediction = pd.DataFrame(prediction,index=ts_test.index[n+1:])
ax7.plot(prediction,label='Sensor T7',alpha=0.7)
ax7.plot(ts_test[' ATT_FLAG'],'r--',label='Actual Attacks')
ax7.legend(loc='best')
fig.suptitle('Predictions')
plt.savefig('disc_pred.png')
plt.show()