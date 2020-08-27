import pandas as pd
from models import *
import matplotlib.pyplot as plt
import json

def process_data_USA(dailycts=False,norm_indiv=False,past_days=120):
    apl_datab = pd.read_csv("data/applemobility3.csv")
    appledata = apl_datab.loc[(apl_datab['country'] == 'United States') & (apl_datab['geo_type'] == 'sub-region'),
                :].drop(
        columns=['geo_type',
                 'transportation_type',
                 'alternative_name',
                 'sub-region',
                 'country'])
    appledata.drop(appledata.columns[1:10], axis=1, inplace=True)
    appledata = appledata.set_index('region')
    appledata.drop(["Guam", "Virgin Islands"], inplace=True)
    appledata = appledata.fillna(100)

    cvd_datab = pd.read_csv('data/coviddata3.csv')
    coviddata = cvd_datab.drop(columns=['UID',
                                        'iso2',
                                        'iso3',
                                        'code3',
                                        'FIPS',
                                        'Country_Region',
                                        'Lat',
                                        'Long_',
                                        'Combined_Key',
                                        'Admin2'])
    coviddata.drop(coviddata.columns[(appledata.shape[1]+1):], axis=1, inplace=True)
    coviddata = coviddata.set_index('Province_State')

    coviddata.drop(coviddata.columns[0:-past_days], axis=1,inplace=True)
    appledata.drop(appledata.columns[0:-past_days], axis=1,inplace=True)
    print("Processed data from " + str(coviddata.columns[0]) + " to " + str(coviddata.columns[-1]))

    irgs = np.intersect1d(coviddata.index, appledata.index)
    sumcoviddata = np.zeros((len(irgs), coviddata.values.shape[1]))
    i = 0
    for state in irgs:
        sumcoviddata[i] = coviddata.loc[state, :].sum(axis=0)
        i += 1

    cv_arr = sumcoviddata
    ap_arr = appledata.values

    x_data = np.zeros((len(irgs), cv_arr.shape[1], 2))
    x_data[:, :, 0] = cv_arr
    x_data[:, :, 1] = ap_arr

    #THIS LINE USES NEW CASES PER DAY AS OPPOSED TO TOTAL CASES:
    if dailycts:
        x_data[:,:,0] -= np.concatenate((np.zeros((x_data.shape[0],1)), x_data[:,:-1,0]),1)

    for i in range(x_data.shape[0]): #plot train data
        plt.plot(x_data[i,:,0])
    plt.show()

    #scaling/normalization:
    if not norm_indiv:
        mx0,mn0 = x_data[:, :, 0].flatten().max(),x_data[:, :, 0].flatten().min()
        mx1,mn1 = x_data[:, :, 1].flatten().max(),x_data[:, :, 1].flatten().min()
        rng = 10
        x_data[:, :, 0] = (rng*(x_data[:, :, 0] - mn0) / (mx0 - mn0))
        x_data[:, :, 1] = (rng*(x_data[:, :, 1] - mn1) / (mx1 - mn1))
        y_data = x_data[:, :, 0]
    else:
        mx0, mn0 = x_data[:, :, 0].max(axis=1), x_data[:, :, 0].min(axis=1)
        mx1, mn1 = x_data[:, :, 1].max(axis=1), x_data[:, :, 1].min(axis=1)
        rng = 10
        x_data[:, :, 0] = (rng*(x_data[:, :, 0].transpose() - mn0) / (mx0 - mn0)).transpose()
        x_data[:, :, 1] = ((x_data[:, :, 1].transpose() - mn1) / (mx1 - mn1)).transpose()
        y_data = x_data[:, :, 0]

    # x_data: (places, timeline, 2)
    # y_data: (places, timeline) where covid counts have NOT been shifted
    # Training example is segment of x_data and corresponding segment in y_data
    return x_data, y_data, (mx0, mn0,rng), irgs

def runcovid():
    x_data,y_data,scale,places = process_data_USA(dailycts=False,norm_indiv=False, past_days=150)
    x_train,y_train = x_data, y_data

    sqlen = 15
    offst = 15
    model = train_rnn(x_train,y_train,
                      scale=scale,
                      batch_size=20,
                      seq_len=sqlen,
                      offset=offst,
                      num_samples=4000)

    lines = []
    for i in range(0,51): #REMOVE FIRST FEW POINTS, CONNECT WITH REST!
        prd = model.predict(torch.from_numpy(x_data[i:i+1, -100:]).float(),st_idx=None).squeeze().detach().numpy()
        #appd = np.concatenate((model.invert_scale(y_data[i,:]),prd[-offst:]),0)
        appd = np.concatenate((model.invert_scale(y_data[i, :],st_idx=None), prd[-offst:]), 0)
        #prd = model.predict(torch.from_numpy(x_data[i:i+1, :]).float(),i).squeeze().detach().numpy()
        #appd = np.concatenate((model.invert_scale(y_data[i,:],i),prd[-offst:]),0)
        lines.append(np.round(appd).tolist())

    with open('linedata10.txt','w') as fp:
        json.dump(lines,fp)

    return lines,places

'''
def process_data_state():
    apl_datab = pd.read_csv("data/applemobility.csv")
    appledata = apl_datab.loc[(apl_datab['sub-region'] == 'Texas') & (apl_datab['geo_type'] == 'county')].drop(
        columns=['geo_type',
                 'transportation_type',
                 'alternative_name',
                 'sub-region',
                 'country'])
    appledata.drop(appledata.columns[1:10], axis=1, inplace=True)
    appledata['region'] = appledata['region'].apply(lambda x: x.split(" ")[0])

    cvd_datab = pd.read_csv('data/coviddata.csv')
    coviddata = cvd_datab.loc[cvd_datab['Province_State'] == 'Texas'].drop(columns=['UID',
                                                                                    'iso2',
                                                                                    'iso3',
                                                                                    'code3',
                                                                                    'FIPS',
                                                                                    'Province_State',
                                                                                    'Country_Region',
                                                                                    'Lat',
                                                                                    'Long_',
                                                                                    'Combined_Key'])
    coviddata.drop(coviddata.columns[183:], axis=1, inplace=True)

    appledata = appledata.set_index('region')
    coviddata = coviddata.set_index('Admin2')

    irgs = list(np.intersect1d(appledata.index, coviddata.index))
    coviddata = coviddata.loc[irgs]
    appledata = appledata.loc[irgs]

    appledata = appledata.fillna(100)

    cv_arr = coviddata.values
    ap_arr = appledata.values

    x_data = np.zeros((len(irgs), cv_arr.shape[1]-50, 2))
    x_data[:,:,0] = cv_arr[:,50:]
    x_data[:,:,1] = ap_arr[:,50:]

    mx0, mn0 = x_data[:, :, 0].max(axis=1), x_data[:, :, 0].min(axis=1)
    mx1, mn1 = x_data[:, :, 1].max(axis=1), x_data[:, :, 1].min(axis=1)
    x_data[:, :, 0] = ((x_data[:, :, 0].transpose() - mn0) / (mx0 - mn0)).transpose()
    x_data[:, :, 1] = ((x_data[:, :, 1].transpose() - mn1) / (mx1 - mn1)).transpose()
    y_data = x_data[:, :, 0]

    #x_data: (counties, timeline, 2)
    #y_data: (counties, timeline) where covid counts have NOT been shifted
    #Training example is segment of x_data and corresponding segment in y_data
    return x_data,y_data,mx0,mn0,irgs,cv_arr

def process_data_USA_scaled():
    apl_datab = pd.read_csv("data/applemobility.csv")
    appledata = apl_datab.loc[(apl_datab['country'] == 'United States') & (apl_datab['geo_type'] == 'sub-region'), :].drop(
        columns=['geo_type',
                 'transportation_type',
                 'alternative_name',
                 'sub-region',
                 'country'])
    appledata.drop(appledata.columns[1:10], axis=1, inplace=True)
    appledata = appledata.set_index('region')
    appledata.drop(["Guam","Virgin Islands"],inplace=True)
    appledata = appledata.fillna(100)

    cvd_datab = pd.read_csv('data/coviddata.csv')
    coviddata = cvd_datab.drop(columns=['UID',
                                                                                    'iso2',
                                                                                    'iso3',
                                                                                    'code3',
                                                                                    'FIPS',
                                                                                    'Country_Region',
                                                                                    'Lat',
                                                                                    'Long_',
                                                                                    'Combined_Key',
                                        'Admin2'])
    coviddata.drop(coviddata.columns[183:], axis=1, inplace=True)
    coviddata = coviddata.set_index('Province_State')

    irgs = np.intersect1d(coviddata.index,appledata.index)
    sumcoviddata = np.zeros((len(irgs),coviddata.values.shape[1]))
    i = 0
    for state in irgs:
        sumcoviddata[i] = coviddata.loc[state,:].sum(axis=0)
        i +=  1

    cv_arr = sumcoviddata
    ap_arr = appledata.values

    x_data = np.zeros((len(irgs), cv_arr.shape[1] - 50, 2))
    x_data[:, :, 0] = cv_arr[:, 50:]
    x_data[:, :, 1] = ap_arr[:, 50:]
    mx0,mn0 = x_data[:, :, 0].max(axis=1),x_data[:, :, 0].min(axis=1)
    mx1,mn1 = x_data[:, :, 1].max(axis=1),x_data[:, :, 1].min(axis=1)
    x_data[:,:,0] = ((x_data[:,:,0].transpose()-mn0)/(mx0-mn0)).transpose()
    x_data[:,:,1] = ((x_data[:,:,1].transpose()-mn1)/(mx1-mn1)).transpose()
    y_data = x_data[:, :, 0]

    # x_data: (counties, timeline, 2)
    # y_data: (counties, timeline) where covid counts have NOT been shifted
    # Training example is segment of x_data and corresponding segment in y_data
    return x_data, y_data, mx0,mn0,irgs, cv_arr
'''

if __name__ == '__main__':
    runcovid()

#https://medium.com/@canerkilinc/hands-on-multivariate-time-series-sequence-to-sequence-predictions-with-lstm-tensorflow-keras-ce86f2c0e4fa
