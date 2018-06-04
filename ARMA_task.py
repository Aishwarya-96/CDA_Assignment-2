import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 30,8
from statsmodels.graphics.api import qqplot
from numpy.linalg import LinAlgError
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
from datetime import datetime

def mean_forecast_err(y, yhat,fit_model):
    # Predicted values are less then the real values
    return y.sub(yhat).mean()

def mean_absolute_error(y,yhat,fit_model):
    return np.square((np.abs(mean_forecast_err(y,yhat,fit_model))))

def compute_errors(data,predictions,fit_model):
    # -model_fit.k_ma
    if model_fit.k_ma == 0:
        data = data[fit_model.k_ma:]
    else:
        data = data[fit_model.k_ar:-model_fit.k_ma]
    MFE = mean_forecast_err(data, predictions,fit_model)
    MAE = mean_absolute_error(data,predictions,fit_model)
    print('Mean Forecast Error: {}\nMean Square Error: {}'.format(MFE, MAE))

def visualize_autocorr(data):
    # Serial correlation
    print(sm.stats.durbin_watson(data))

    # Plot correlation graphs
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data.values.squeeze(), lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data, lags=40, ax=ax2)
    plt.show()

def selectARMA(data):
    # Compute autocorreleation/partial autocorrelation and select an ARMA model based on AIC value
    #visualize_autocorr(data)
    # Try
    p = [1,2,3,4,5]
    q = [0,1,2,3,4]
    print('Start searching for optimal ARMA model')
    min_aic = list()
    for p_val in p:
        for q_val in q:
            try:
                model = sm.tsa.ARMA(data,(p_val,q_val)).fit(disp=-1)
            except LinAlgError:
                continue
            min_aic.append((p_val,q_val,model.aic))
    best_model = min_aic.index(min(min_aic, key=lambda t: t[2]))
    #print('Best AIC value: {}'.format(min_aic[best_model][2]))
    #print(model.params)
    #print('AIC: {}\tBIC: {}\tHQIC: {}'.format(model.aic,model.bic,model.hqic))
    print('p: {}\tq:{}'.format(min_aic[best_model][0],min_aic[best_model][1]))

    # Analyze residuals
    #print('Analyse residuals of found ARMA model')
    #analyze_residuals(model.resid)
    return model



def analyze_residuals(residuals,s_name):
    # Autocorrelation after fitting a model
    print('Durbin Watson Statistics:\n{}'.format(sm.stats.durbin_watson(residuals.values)))
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111)
    ax = residuals.plot(ax=ax,label='Sensor {} Residuals'.format(s_name))
    plt.show()

    print('Normal Test: {}'.format(stats.normaltest(residuals)))
    #fig = plt.figure(figsize=(20, 8))
    #ax = fig.add_subplot(111)
    #fig = qqplot(residuals, line='q', ax=ax, fit=True)
    #plt.show()

def prediction_ondata(data,start_date,end_date,model):
    # Predictions on train data
    # to check if model can predict the normal time series
    predictions= model.predict(start_date,end_date ,dynamic=False)
    #print(predictions)
    # Plot Predictions
    '''
    ax = data.ix['20140106T00':].plot(figsize=(20, 8))
    ax = predictions.plot(ax=ax, style='r--', label='Prediction on Train Data')
    ax.legend()
    plt.show()
    '''
    return predictions

def prediction_ontest(test,model_fit):
    # Make predictions on test data using a learned ARMA model
    # AR part
    y_ar = float(0)
    y_ma = float(0)

    predictions = np.array(list())
    ar_order = model_fit.k_ar
    ma_order = model_fit.k_ma
    hist = np.array(test[-ar_order:])
    if ma_order == 0:
        res = np.array([0])
    else:
        res = np.array(test[-ma_order:])
    ar_params = model_fit.arparams
    ma_params = model_fit.maparams
    # Compute AR prediction
    for i in range(len(test)-(ar_order+ma_order)):
        for j in range(len(ar_params)):
            y_ar += ar_params[j] * hist[i+j]
    # Compute MA prediction
        for j in range(len(ma_params)):
            y_ma += ma_params[j] * res[i+j]
        hist = np.insert(hist,ar_order+i,test[ar_order+i])
        predictions = np.insert(predictions,i,round(y_ar+y_ma,2))
        #predictions[i] = y_ar + y_ma
        y_ar = y_ma = 0.0
        res = np.append(res,(test[i] - predictions[i]))
    res_df = pd.DataFrame(res,index=test.index[:len(res)])

    return (predictions,res_df)


def detect_anomalies(data,predictions,ma,k,fit_model):
    anomalies = pd.DataFrame(index=data.index)
    #print(anomalies.index)
    anomalylist = list()
    tn = tp = fn = fp = 0
    abserr = mean_squared_error(data,predictions)
    stdma = np.std(ma)
    for sample in range(len(predictions)):
        # Prediction error
        pe = np.abs(predictions[sample]-data[sample])**2
        if pe > k * stdma + abserr:
            anomalylist.append(1)
            if batadal_04[' ATT_FLAG'][sample] == 1:
                tp += 1
            elif batadal_04[' ATT_FLAG'][sample] == -999:
                fp += 1
        else:
            anomalylist.append(-999)
            if batadal_04[' ATT_FLAG'][sample] == 1:
                tn += 1
            elif batadal_04[' ATT_FLAG'][sample] == -999:
                fn += 1

    print('TP: {}\tTN: {}\tFP: {}\tFN: {}'.format(tp,tn,fp,fn))
    anomalies['a'] = anomalylist
    return anomalies

def find_k(data,predictions,ma,fit_model):
    k = np.arange(30,40,0.1)
    min_rate = list()
    anomalylist = list()
    anomalies = pd.DataFrame(index=data.index)
    abserr = mean_squared_error(data, predictions)
    print('SRE: {}'.format(abserr))
    stdma = np.std(ma)
    tp =  tn = fp = fn = 0
    for k_val in k:
        for sample in range(len(predictions)):
            pe = np.abs(predictions[sample] - data[sample])**2
            if pe > k_val * stdma + abserr:
                anomalylist.append(1)
                if batadal_04[' ATT_FLAG'][sample] == 1:
                    tp += 1
                elif batadal_04[' ATT_FLAG'][sample] == -999:
                    fp += 1
            else:
                anomalylist.append(-999)
                if batadal_04[' ATT_FLAG'][sample] == 1:
                    tn += 1
                if batadal_04[' ATT_FLAG'][sample] == -999:
                    fn += 1
        rate = tp / fp;
        min_rate.append((rate,k_val,anomalylist))
        print('k: {}\ttp: {}\ttn: {}\nfp: {}\tfn: {}'.format(k_val,tp,tn,fp,fn))
        anomalylist = []
        tp = tn = fp = fn = 0
    idx = min_rate.index(max(min_rate, key=lambda t: t[0]))
    anomalies['a'] = min_rate[idx][2]
    print('k: {}'.format(min_rate[idx][1]))
    return (min_rate[idx][1],anomalies)

def test(sensor, model_fit,k_pred):
    # test trained model
    # sensor: array. data to be tested
    # t_model: trained model parameters
    # k: parameter required for threshold
    predictions_on_test, residuals = prediction_ontest(sensor,model_fit)
    # Forecast errors
    # Not the all data points are predicted due to sliding windows
    # Remove the data not predicted on the prediction set -model_fit.k_ma
    if model_fit.k_ma == 0:
        sensor = sensor[model_fit.k_ar:]
    else:
        sensor = sensor[model_fit.k_ar:-model_fit.k_ma]
    print('Forecast Errors on Test set')
    print(mean_squared_error(sensor, predictions_on_test))

    # Detect Anomalies
    rolMean = pd.rolling_mean(sensor, window=24)
    # used to choose k value
    #k,anomalies = find_k(sensor,predictions_on_test,rolMean, model_fit)
    anomalies = detect_anomalies(sensor, predictions_on_test, rolMean, k_pred,model_fit)

    return (anomalies,residuals)


# Read train and test data
batadal_03 = pd.read_csv('..\dataset\BATADAL_dataset03.csv')
batadal_03.dropna()
batadal_04 = pd.read_csv('..\dataset\BATADAL_dataset04.csv')
batadal_04.dropna()
#print(batadal_04.head())
batadal_03.index = pd.DatetimeIndex(start='20140106T00',end='20150106T00',freq='1H')
batadal_04.index = pd.DatetimeIndex(start='20160704T00',end='20161225T00',freq='1H')
#print(batadal_03.index)
#print(batadal_04.index)
del batadal_03['DATETIME']
del batadal_04['DATETIME']
#print(batadal_03.columns)
#print(batadal_04.columns)

# Sensor L_T1
print('----Sensor T1----')
#visualize_autocorr(batadal_03['L_T1'])
# ARMA paramters can be selected via a small search. Yet, not always found model converges
# or proveides better performance
#model_fit = selectARMA(batadal_03['L_T1'])
# Rather, parameter selection has been done via autocorrelation plots
arma_model = sm.tsa.ARMA(batadal_03['L_T1'],(2,0))
model_fit = arma_model.fit(disp=-1)
fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4,2,sharex=False,sharey=False)
ax1.plot(model_fit.resid,label='Sensor T1')
ax1.legend(loc='best')

# Analyse Train Residuals
#analyze_residuals(model_fit.resid,'L_T1')
# Predictions(on train data) -- to check is model can represent ts well enough
#predictions_on_train = prediction_ondata(batadal_03['L_T1'],'20140602T00','20150106T00',model_fit)
# Forecast errors
#print('Forecast Errors on Train Data')
#compute_errors(batadal_03['L_T1'],predictions_on_train,model_fit)

# Predications on batadal_04(validation Set)
an_t1,res_t1 = test(batadal_04[' L_T1'],model_fit,2)

# Sensor L_T2
print('----Sensor T2----')
#visualize_autocorr(batadal_03['L_T2'])
#model_fit = selectARMA(batadal_03['L_T2'])

arma_model = sm.tsa.ARMA(batadal_03['L_T2'],(2,1))
model_fit = arma_model.fit(disp=-1)
ax2.plot(model_fit.resid,label='Sensor T2')
ax2.legend(loc='best')

# Predictions(on train data) -- to check is model can represent ts well enough
predictions_on_train = prediction_ondata(batadal_03['L_T2'],'20140602T00','20150106T00',model_fit)
# Forecast errors
#print('Forecast Errors on Train Data')
#compute_errors(batadal_03['L_T2'],predictions_on_train,model_fit)

# Predications on batadal_04(validation Set)
an_t2,res_t2 = test(batadal_04[' L_T2'],model_fit,6.2)

# Sensor L_T3
print('----Sensor T3----')
#visualize_autocorr(batadal_03['L_T3'])
#model_fit = selectARMA(batadal_03['L_T3'])

arma_model = sm.tsa.ARMA(batadal_03['L_T3'],(2,0))
model_fit = arma_model.fit(disp=-1)
ax3.plot(model_fit.resid,label='Sensor T3')
ax3.legend(loc='best')

# Predictions(on train data) -- to check is model can represent ts well enough
#predictions_on_train = prediction_ondata(batadal_03['L_T2'],'20140602T00','20150106T00',model_fit)
# Forecast errors
#print('Forecast Errors on Train Data')
#compute_errors(batadal_03['L_T3'],predictions_on_train,model_fit)

# Predications on batadal_04(validation Set)
an_t3,res_t3 = test(batadal_04[' L_T3'],model_fit,4)

# Sensor L_T4
print('----Sensor T4----')
#visualize_autocorr(batadal_03['L_T4'])
#model_fit = selectARMA(batadal_03['L_T4'])

arma_model = sm.tsa.ARMA(batadal_03['L_T4'],(1,0))
model_fit = arma_model.fit(disp=-1)
ax4.plot(model_fit.resid,label='Sensor T4')
ax4.legend(loc='best')

# Predictions(on train data) -- to check is model can represent ts well enough
#predictions_on_train = prediction_ondata(batadal_03['L_T4'],'20140602T00','20150106T00',model_fit)
# Forecast errors
#print('Forecast Errors on Train Data')
#compute_errors(batadal_03['L_T4'],predictions_on_train)

# Predications on batadal_04(validation Set)
an_t4,res_t4 = test(batadal_04[' L_T4'],model_fit,11.5)

# Sensor L_T5
print('----Sensor T5----')
#visualize_autocorr(batadal_03['L_T5'])
#model_fit = selectARMA(batadal_03['L_T5'])

arma_model = sm.tsa.ARMA(batadal_03['L_T5'],(1,0))
model_fit = arma_model.fit(disp=-1)
ax5.plot(model_fit.resid,label='Sensor T5')
ax5.legend(loc='best')

# Predictions(on train data) -- to check is model can represent ts well enough
#predictions_on_train = prediction_ondata(batadal_03['L_T5'],'20140602T00','20150106T00',model_fit)
# Forecast errors
#print('Forecast Errors on Train Data')
#compute_errors(batadal_03['L_T5'],predictions_on_train)

# Predications on batadal_04(validation Set)
an_t5,res_t5 = test(batadal_04[' L_T5'],model_fit,13.4)

# Sensor L_T6
print('----Sensor T6----')
#visualize_autocorr(batadal_03['L_T6'])
#model_fit = selectARMA(batadal_03['L_T6'])

arma_model = sm.tsa.ARMA(batadal_03['L_T6'],(1,1))
model_fit = arma_model.fit(disp=-1)
ax6.plot(model_fit.resid,label='Sensor T6')
ax6.legend(loc='best')

# Predictions(on train data) -- to check is model can represent ts well enough
#predictions_on_train = prediction_ondata(batadal_03['L_T6'],'20140602T00','20150106T00',model_fit)
# Forecast errors
#print('Forecast Errors on Train Data')
#compute_errors(batadal_03['L_T6'],predictions_on_train,model_fit)

# Predications on batadal_04(validation Set)
an_t6,res_t6 = test(batadal_04[' L_T6'],model_fit,45.2)

# Sensor L_T7
print('----Sensor T7----')
#visualize_autocorr(batadal_03['L_T7'])
#model_fit = selectARMA(batadal_03['L_T7'])

arma_model = sm.tsa.ARMA(batadal_03['L_T7'],(2,1))
model_fit = arma_model.fit(disp=-1)
ax7.plot(model_fit.resid,label='Sensor T7')
ax7.legend(loc='best')

# Predictions(on train data) -- to check is model can represent ts well enough
#predictions_on_train = prediction_ondata(batadal_03['L_T7'],'20140602T00','20150106T00',model_fit)
# Forecast errors
#print('Forecast Errors on Train Data')
#compute_errors(batadal_03['L_T7'],predictions_on_train,model_fit)

# Predications on batadal_04(validation Set)
an_t7,res_t7 = test(batadal_04[' L_T7'],model_fit,33.6)

# Show train residuals
plt.suptitle('Train Residuals')
#plt.savefig('trainresid.png')
plt.show()


# Test Residuals
fig ,((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4,2,sharex=False,sharey=False)
ax1.plot(res_t1,label='Sensor T1')
ax1.legend(loc='best')
ax2.plot(res_t2,label='Sensor T2')
ax2.legend(loc='best')
ax3.plot(res_t3,label='Sensor T3')
ax3.legend(loc='best')
ax4.plot(res_t4,label='Sensor T4')
ax4.legend(loc='best')
ax5.plot(res_t5,label='Sensor T5')
ax5.legend(loc='best')
ax6.plot(res_t6,label='Sensor T6')
ax6.legend(loc='best')
ax7.plot(res_t7,label='Sensor T7')
ax7.legend(loc='best')
fig.suptitle('Test Residuals')
#plt.savefig('testresid.png')
plt.show()

# Predictions
fig ,((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4,2,sharex=False,sharey=False)
ax1.plot(an_t1,label='Sensor T1')
ax1.plot(batadal_04[' ATT_FLAG'],label='Actual Attacks',alpha=0.7)
ax1.legend(loc='best')
ax2.plot(an_t2,label='Sensor T2')
ax2.plot(batadal_04[' ATT_FLAG'],label='Actual Attacks',alpha=0.7)
ax2.legend(loc='best')
ax3.plot(an_t3,label='Sensor T3')
ax3.plot(batadal_04[' ATT_FLAG'],label='Actual Attacks',alpha=0.7)
ax3.legend(loc='best')
ax4.plot(an_t4,label='Sensor T4')
ax4.plot(batadal_04[' ATT_FLAG'],label='Actual Attacks',alpha=0.7)
ax4.legend(loc='best')
ax5.plot(an_t5,label='Sensor T5')
ax5.plot(batadal_04[' ATT_FLAG'],label='Actual Attacks',alpha=0.7)
ax5.legend(loc='best')
ax6.plot(an_t6,label='Sensor T6')
ax6.plot(batadal_04[' ATT_FLAG'],label='Actual Attacks',alpha=0.7)
ax6.legend(loc='best')
ax7.plot(an_t7,label='Sensor T7')
ax7.plot(batadal_04[' ATT_FLAG'],label='Actual Attacks',alpha=0.7)
ax7.legend(loc='best')
fig.suptitle('Predictions')
#plt.savefig('predictions.png')
plt.show()

