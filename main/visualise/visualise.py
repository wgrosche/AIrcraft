import pandas as pd

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import json
BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('main')[0]
sys.path.append(BASEPATH)
# from src.dynamics import LinearisedAircraft, CD_alpha, CL_alpha
DATA_DIR = os.path.join(BASEPATH, 'data')

data_real = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'data_real.csv'))
data_sim = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'data_sim.csv'))
data_full = pd.concat([data_real, data_sim], axis=0)
com_offset = [0.133, 0, 0.003]

def linearised_model(filepath, inputs, feature):
    coeff_table = json.load(filepath)
    (q, alpha, beta, aileron, elevator) = inputs

    if feature =='CX':
        CZ = np.dot(np.array(coeff_table['CZ']['coefs']), inputs.T) + coeff_table['CZ']['intercept']

        return coeff_table['CX']['k'] * CZ ** 2 + coeff_table['CX']['CD0']
    return np.dot(np.array(coeff_table[feature]['coefs']), inputs.T) + coeff_table[feature]['intercept']



print(data_sim.head())
print(data_real['Cm'].min(), data_real['Cm'].max())
print(data_sim['Cm'].min(), data_sim['Cm'].max())


fig = plt.figure(figsize=(18, 10))
for i in range(6):
    ax = fig.add_subplot(2, 3, i+1, projection='3d')
    if i == 0:
        ax.scatter(data_real['alpha'], data_real['beta'], linearised_model(open('data/glider/linearised.json'), data_real[['q', 'alpha', 'beta', 'aileron', 'elevator']], data_real.columns[i+6]), marker = 'o', label='linear')
    else:
        ax.scatter(data_real['alpha'], data_real['beta'], linearised_model(open('data/glider/linearised.json'), data_real[['q', 'alpha', 'beta', 'aileron', 'elevator']], data_real.columns[i+6]), marker = 'o', label='linear')
    
    ax.scatter(data_real['alpha'], data_real['beta'], data_real.iloc[:, i+6], marker='o', label='real')
    ax.scatter(data_sim['alpha'], data_sim['beta'], data_sim.iloc[:, i+6], marker='o', label='sim')
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    ax.set_zlabel(data_real.columns[i+6])
    ax.legend()
plt.show()
