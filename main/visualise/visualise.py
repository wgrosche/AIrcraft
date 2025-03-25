import pandas as pd

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import json
BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('main')[0]
sys.path.append(BASEPATH)
# from aircraft.dynamics import LinearisedAircraft, CD_alpha, CL_alpha
DATA_DIR = os.path.join(BASEPATH, 'data')

from aircraft.dynamics.aircraft import Aircraft


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

import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
# Customize seaborn aesthetics
sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})

data_real = data_real.sample(frac=0.5)
data_sim = data_sim.sample(frac=0.5)

data_linear = data_real.copy()
data_linear['CX'] = linearised_model(open('data/glider/linearised.json'), data_real[['q', 'alpha', 'beta', 'aileron', 'elevator']], 'CX')
# Example 1: Pairwise Comparisons (Alpha vs CX)
plt.figure(figsize=(8, 6))
sns.scatterplot(x="alpha", y="CX", data=data_real, label="Real", color="C0", alpha=1.0, marker='o', edgecolor='black')
sns.scatterplot(x="alpha", y="CX", data=data_sim, label="Sim", color="C1", alpha=0.6, marker='^', edgecolor='black')

# sns.lineplot(x="alpha", y="CX", data=data_sim, label="Linear", color="C2")
plt.title(r"Angle of Attack vs Drag Coefficient", fontsize = 16)
plt.xlabel(r"Angle of Attack ($\alpha$ [rad])", fontsize = 16)
plt.ylabel(r"$C_X$ Coefficient", fontsize = 16)
plt.legend(fontsize = 14)
# set tight layout
plt.tight_layout()
plt.savefig(os.path.join(BASEPATH, 'figures', 'alpha_vs_cx.svg'), dpi=300, format='svg')
plt.show( block = True)
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(x="alpha", y="CZ", data=data_real, label="Real", color="C0", alpha=1.0, marker='o', edgecolor='black')
sns.scatterplot(x="alpha", y="CZ", data=data_sim, label="Sim", color="C1", alpha=0.6, marker='^', edgecolor='black')
plt.title(r"Angle of Attack vs Lift Coefficient", fontsize = 16)
plt.xlabel(r"Angle of Attack ($\alpha$ [rad])", fontsize = 16)
plt.ylabel(r"$C_Z$ Coefficient", fontsize = 16)
plt.legend(fontsize = 14)
# set tight layout
plt.tight_layout()
plt.savefig(os.path.join(BASEPATH, 'figures', 'alpha_vs_cz.svg'), dpi=300, format='svg')
plt.show( block = True)
plt.close()


# # Example 2: Beta vs CX
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x="beta", y="CY", data=data_real, label="Real", color="blue", alpha=0.6)
# sns.scatterplot(x="beta", y="CY", data=data_sim, label="Sim", color="orange", alpha=0.6)
# plt.title("Beta vs CX")
# plt.xlabel("Beta (Sideslip Angle)")
# plt.ylabel("CX Coefficient")
# plt.legend()
# plt.show(block = True)

# # data_real.where(data_real['ailerons']==0).where(data_real['elevator'] == 0).dropna()

# fig = plt.figure(figsize=(18, 10))
# for i in range(6):
#     ax = fig.add_subplot(2, 3, i+1, projection='3d')
#     # if i == 0:
#     #     ax.scatter(data_real['alpha'], data_real['beta'], linearised_model(open('data/glider/linearised.json'), data_real[['q', 'alpha', 'beta', 'aileron', 'elevator']], data_real.columns[i+6]), marker = 'o', label='linear')
#     # else:
#     #     ax.scatter(data_real['alpha'], data_real['beta'], linearised_model(open('data/glider/linearised.json'), data_real[['q', 'alpha', 'beta', 'aileron', 'elevator']], data_real.columns[i+6]), marker = 'o', label='linear')
    
#     ax.scatter(data_real['alpha'], data_real['beta'], data_real.iloc[:, i+6], marker='^', label='real', alpha = 1.0)
#     ax.scatter(data_sim['alpha'], data_sim['beta'], data_sim.iloc[:, i+6], marker='o', label='sim', alpha = 0.6)
#     ax.set_xlabel(r"Angle of Attack $\alpha$ [rad]")
#     ax.set_ylabel(r"Angle of Sideslip $\beta$ [rad]")
#     ax.set_title(data_real.columns[i+6])
#     ax.set_zlabel(data_real.columns[i+6])
#     if i == 0:
#         ax.legend()
# plt.show(block = True)



# # Example 3: Heatmap of Differences (Optional)
# df['CX_diff'] = df['CX'] - df['CX_sim']
# heatmap_data = df.pivot_table(index="alpha", columns="beta", values="CX_diff")
# plt.figure(figsize=(10, 8))
# sns.heatmap(heatmap_data, cmap="coolwarm", center=0)
# plt.title("Heatmap of Real vs Sim Differences (CX)")
# plt.xlabel("Beta")
# plt.ylabel("Alpha")
# plt.show()




# from scipy.interpolate import griddata

# fig = plt.figure(figsize=(18, 10))
# for i in range(6):
#     ax = fig.add_subplot(2, 3, i+1, projection='3d')
    
#     # Prepare grid for surface plot
#     grid_alpha, grid_beta = np.meshgrid(
#         np.linspace(data_real['alpha'].min(), data_real['alpha'].max(), 50),
#         np.linspace(data_real['beta'].min(), data_real['beta'].max(), 50)
#     )
    
#     # Interpolate data for smooth surface
#     grid_coeff_real = griddata(
#         (data_real['alpha'], data_real['beta']), 
#         data_real.iloc[:, i+6], 
#         (grid_alpha, grid_beta), 
#         method='cubic'
#     )
    
#     ax.plot_surface(grid_alpha, grid_beta, grid_coeff_real, cmap='viridis', alpha=0.8, label='real')
#     ax.scatter(data_real['alpha'], data_real['beta'], data_real.iloc[:, i+6], color='r', label='real (points)')
    
#     ax.set_xlabel('alpha')
#     ax.set_ylabel('beta')
#     ax.set_zlabel(data_real.columns[i+6])
#     ax.legend()
# plt.show(block =True)