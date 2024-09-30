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

data_real = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'data_sim.csv'))
data_sim = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'data_sim.csv'))

params = json.load(open(os.path.join(DATA_DIR, 'glider', 'problem_definition.json')))['aircraft']
com = np.array(params['aero_centre_offset'])
data_full = pd.concat([data_real, data_sim], axis=0)
com_offset = [0.0067578, 0, -0.000556231]#[0.133, 0, 0.003]



print(data_sim.head())
print(data_real['Cm'].min(), data_real['Cm'].max())
print(data_sim['Cm'].min(), data_sim['Cm'].max())


data_real = data_real.sample(frac=.1).reset_index(drop=True)
data_sim = data_sim.sample(frac=.1).reset_index(drop=True)
# visualise 3d
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(231, projection='3d')
ax.scatter(data_real['alpha'], data_real['beta'], data_real['CL'] * data_real['q'] * 1.4**2 * params['reference_area'], label='real', color='red')
ax.scatter(data_sim['alpha'], data_sim['beta'], data_sim['CL'] * data_sim['q'] * params['reference_area'], label='sim', color='blue')
    
    
ax.set_xlabel('alpha')
ax.set_ylabel('beta')
ax.set_zlabel('CL')

ax2 = fig.add_subplot(232, projection='3d')
ax2.scatter(data_real['alpha'], data_real['beta'], data_real['CD'], label='real', color='red')
ax2.scatter(data_sim['alpha'], data_sim['beta'], data_sim['CD'], label='sim', color='blue')
ax2.set_xlabel('alpha')
ax2.set_ylabel('beta')
ax2.set_zlabel('CD')

ax3 = fig.add_subplot(233, projection='3d')
ax3.scatter(data_real['alpha'], data_real['beta'], data_real['CY'], label='real', color='red')
ax3.scatter(data_sim['alpha'], data_sim['beta'], data_sim['CY'], label='sim', color='blue')
ax3.set_xlabel('alpha')
ax3.set_ylabel('beta')
ax3.set_zlabel('CY')

Cl_from_forces = com_offset[1] * data_full['CL'] - com_offset[2] * data_full['CY']
Cm_from_forces = -com_offset[0] * data_full['CL'] + com_offset[2] * data_full['CD']
Cn_from_forces = com_offset[0] * data_full['CY'] - com_offset[1] * data_full['CD']


ax4 = fig.add_subplot(234, projection='3d')
ax4.scatter(data_real['alpha'], data_real['beta'], data_real['Cl'], label='real', color='red')
ax4.scatter(data_full['alpha'], data_full['beta'], Cl_from_forces, label='real', color='green')
ax4.scatter(data_sim['alpha'], data_sim['beta'], data_sim['Cl'], label='sim', color='blue')
ax4.set_xlabel('alpha')
ax4.set_ylabel('beta')
ax4.set_zlabel('Cl')


ax5 = fig.add_subplot(235, projection='3d')
ax5.scatter(data_real['alpha'], data_real['beta'], data_real['Cm'], label='real', color='red')
ax5.scatter(data_full['alpha'], data_full['beta'], Cm_from_forces, label='real', color='green')
ax5.scatter(data_sim['alpha'], data_sim['beta'], data_sim['Cm'], label='sim', color='blue')
ax5.set_xlabel('alpha')
ax5.set_ylabel('beta')
ax5.set_zlabel('Cm')

ax6 = fig.add_subplot(236, projection='3d')
ax6.scatter(data_real['alpha'], data_real['beta'], data_real['Cn'], label='real', color='red')
ax6.scatter(data_full['alpha'], data_full['beta'], Cn_from_forces, label='real', color='green')
ax6.scatter(data_sim['alpha'], data_sim['beta'], data_sim['Cn'], label='sim', color='blue')
ax6.set_xlabel('alpha')
ax6.set_ylabel('beta')
ax6.set_zlabel('Cn')

plt.show()
