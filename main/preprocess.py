"""
Preprocesses windtunnel and freestreem simulation data:
    - Loads simulation data from npz
    - Converts angles to radians
    - Flips signs of x and z axes to get from drag, lift to fx, fz (NED)
    - Flips sign of alpha to account convention differences.
    - Converts freestream simulation forces into the body frame
    - Centres control surface deflections around 0 (range -5, 5)
    - Converts from aerodynamic forces to coefficients
    - Scales velocities to account for size (Change in Reynolds)
    - Augments data by exploiting symmetries in xz-plane


NOTE:
The resulting data are inconsistent in the sideforce CY and yawing moment Cn.
A correction has been applied to account for this in accordance with my 
understanding of the definition of beta, however this is not a principled
solution to the problem.

The windtunnel data is in the wind frame whereas the freestream data is in the 
body frame.

TODO: Double-check results


CONVENTION FOR END RESULT:
    pos,    vel,    force,  omega,  moment
N   x       u       X       p       l
E   y       v       Y       q       m
D   z       w       Z       r       n

sign of moment and angular velocity -> positive right handed rotation, 
negative left handed.

alpha = atan(w / u)
beta = asin(v / |(u, v, w)^T|)

CONVENTION IN USE FOR REAL WINDTUNNEL DATA:

Signs give relation to the end result.

    pos,    vel,    force,  omega,  moment
S   -x      -u      -X      -p       -l
E   y       v       Y       -q       -m
U   -z      -w      -Z      -r       -n

alpha_r = atan(-w / u) = -alpha
beta = asin(-v / |(u, v, w)^T|) = -beta

NOTE: The results only seem to make sense when the moments are flipped

CONVENTION IN USE FOR SIMULATED WINDTUNNEL DATA:

Signs give relation to the end result.

    pos,    vel,    force,  omega,  moment
S   -x      -u      -X      -p       -l
E   -y      -v      -Y       q        m
U   -z      -w      -Z      -r       -n

alpha_r = atan(-w / u) = -alpha
beta = asin(-v / |(u, v, w)^T|) = -beta


CONVENTION IN USE FOR SIMULATED FREESTREAM DATA:

Signs give relation to the end result.

    pos,    vel,    force,  omega,  moment
S   -x      -u      -X      -p       -l
E   -y      -v      -Y       q        m
U   -z      -w      -Z      -r       -n

alpha_r = atan(-w / u) = -alpha
beta = asin(-v / |(u, v, w)^T|) = -beta


"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import json
from copy import deepcopy

# Constants
RHO = 1.225 # air density
IN_TO_M = 0.0254
LBF_TO_N = 4.44822
LBS_TO_KG = 0.453592
MACH_TO_MS = 343

process_real = False # flag to include real windtunnel data


def R(alpha, beta):
    """
    Rotation matrix from wind to body axes. It is transposed for np.einsum

    Parameters
    ----------
    alpha : float
        Angle of attack in radians
    beta : float
        Sideslip angle in radians

    Returns
    -------
    R : numpy array
        Rotation matrix
    """
    R = np.zeros((3, 3, len(alpha)))
    R[0, 0] = np.cos(alpha) * np.cos(beta)
    R[0, 1] = - np.sin(beta) * np.cos(alpha)
    R[0, 2] = - np.sin(alpha)
    R[1, 0] = np.sin(beta)
    R[1, 1] = np.cos(beta)
    R[1, 2] = 0.
    R[2, 0] = np.cos(beta) * np.sin(alpha)
    R[2, 1] = - np.sin(beta) * np.sin(alpha)
    R[2, 2] = np.cos(alpha)
    return np.transpose(R, (1, 0, 2))

def process_sim_dataset(
        input:np.lib.npyio.NpzFile, 
        params:dict, 
        goal_params:dict, 
        degrees:bool = True,
        body:bool = False,
        augment:bool = True,
        axes:np.ndarray = np.ones((6, 1))
        ):
    """
    Takes as input a simulation dataset containing aerodynamic forces 
    and moments at given:
        - angles of attack
        - angles of sideslip
        - velocity
        - control surface deflection

    The forces and moments are assumed to be in a wind aligned (USE) frame.
    Moments are taken around 25% MAC.

    Parameters:
        - params: dictionary containing information about the aircraft for the 
                    data (span, chord, reference area, etc.)
        - goal_params: dictionary containing information about the aircraft 
                    to be simulated
        - degrees : flag to convert aerodynamic angles to radians
        - body : flag stating whether forces are already in the body frame
        - augment : flag for exploiting mirror symmetry in xz-plane
        - axes : custom flip for certain axes to account for inconsitencies in
                    the data

    Returns an array of aerodynamic coefficients defined in the NED coordinate 
    system. Moments are defined positive in the right handed sense.

    Conventions for the output:
        - alpha = atan(w / u) is positive for the nose facing towards the 
                                ground (down)
        - beta = asin(v / V) is positive for sideslip towards the right 
                            (as viewed from the perspective of a pilot)
        - delta_aileron is defined positive if the right wing 
                            (positive y direction) is deflected downward 
                            (positive z direction)
        - delta_elevator is defined positive if the elevator is deflected 
                            downward (positive z direction)
    """
    
    output = pd.DataFrame(columns = ['q', 'alpha', 'beta', 'aileron', 
                                     'elevator', 'windtunnel', 'CX', 'CY', 'CZ', 
                                     'Cl', 'Cm', 'Cn'])

    S = params['reference_area']
    b = params['span']
    c = params['chord']

    scale_factor = b / goal_params['span']
    
    q = np.array(input['vel'], dtype=float) ** 2 * 0.5 * RHO

    # sign flip for alpha convention in sim data vs control
    alpha = np.array(input['alpha'], dtype=float) 
    beta = np.array(input['beta'], dtype=float)

    if degrees:
        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)

    if body:
        rotmat = R(np.zeros_like(alpha), np.zeros_like(beta))
    else:
        # rotmat = R(alpha, beta)
        rotmat = R(-alpha, -beta)

    forces = np.array([
            np.array(input['fx'], dtype=float),
            np.array(input['fy'], dtype=float),
            np.array(input['fz'], dtype=float)
        ])
    forces *= axes[:3]

    moments = np.array([
            np.array(input['mx'], dtype=float),
            np.array(input['my'], dtype=float), 
            np.array(input['mz'], dtype=float)
            ])
    
    moments *= axes[3:]

    print(rotmat)

    forces = np.einsum('ijk,jk->ik', rotmat, forces).T
    moments = np.einsum('ijk,jk->ik', rotmat, moments).T

    output['CX'] = forces[:, 0] / (q * S)
    # output['CX'] = - forces[:, 0] / (q * S)
    output['CY'] = forces[:, 1] / (q * S)
    output['CZ'] = - forces[:, 2] / (q * S)

    output['Cl'] = - moments[:, 0] / (q * S * b)
    output['Cm'] = moments[:, 1] / (q * S * c)
    output['Cn'] = - moments[:, 2] / (q * S * b)

    # centre control surface deflections
    output['aileron'] = -(np.array(input['ctrl1'], dtype=float) - 5.0)
    output['elevator'] = -(np.array(input['ctrl2'], dtype=float) - 5.0)

    output['beta'] = - beta
    output['alpha'] = - alpha
    output['windtunnel'] = not body

    output['q'] = q * scale_factor ** 2

    if augment:
        mirrored = output.copy()

        mirrored['beta'] = - mirrored['beta']
        mirrored["CY"] = - mirrored["CY"]
        mirrored["Cl"] = - mirrored["Cl"]
        mirrored["Cn"] = - mirrored["Cn"]
        mirrored['aileron'] = - mirrored['aileron']

        output = pd.concat([output, mirrored], ignore_index=True)

    return output

def process_wt_dataset(
        input:pd.DataFrame, 
        params:dict, 
        goal_params:dict, 
        degrees:bool = True,
        body:bool = False,
        augment:bool = False,
        axes:np.ndarray = np.ones((6, 1))
        ):
    """
    Takes as input a windtunnel dataset containing aerodynamic coefficients 
    at given:
        - angles of attack
        - angles of sideslip
        - velocities

    The forces and moments are assumed to be in a wind aligned (USE) frame.
    Moments are taken around 25% of MAC.

    Parameters:
        - params: dictionary containing information about the aircraft for the 
                    data (span, chord, reference area, etc.)
        - goal_params: dictionary containing information about the aircraft 
                    to be simulated
        - degrees : flag to convert aerodynamic angles to radians
        - body : flag stating whether forces are already in the body frame
        - augment : flag for exploiting mirror symmetry in xz-plane

    Returns an array of aerodynamic coefficients defined in the NED coordinate 
    system. Moments are defined positive in the right handed sense.

    Conventions for the output:
        - alpha = atan(w / u) is positive for the nose facing towards the 
                                ground (down)
        - beta = asin(v / V) is positive for sideslip towards the right 
                            (as viewed from the perspective of a pilot)
        - delta_aileron is defined positive if the right wing 
                            (positive y direction) is deflected downward 
                            (positive z direction)
        - delta_elevator is defined positive if the elevator is deflected 
                            downward (positive z direction)
    """
    
    output = pd.DataFrame(columns = ['q', 'alpha', 'beta', 'aileron', 
                                     'elevator', 'windtunnel', 'CX', 'CY', 'CZ', 
                                     'Cl', 'Cm', 'Cn'])

    S = params['reference_area']

    scale_factor = params['span'] / goal_params['span']

    q = 0.5 * RHO * S * (input['MACH'] * MACH_TO_MS) ** 2

    # sign flip for alpha convention in sim data vs control
    alpha = np.array(input['ALPHAC'], dtype=float) 
    beta = np.array(input['PSI'], dtype=float)

    if degrees:
        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)

    if body:
        rotmat = R(np.zeros_like(alpha), np.zeros_like(beta))
        beta = beta
    else:
        rotmat = R(alpha, beta)
        # rotmat = R(alpha, beta)

    forces = np.array([
            np.array(input['CDWA'], dtype=float),
            np.array(input['CYWA'], dtype=float),
            np.array(input['CLWA'], dtype=float)
        ])
    forces *= axes[:3]
    
    moments = np.array([
            np.array(input['CRWA25'], dtype=float), # rolling coeff
            np.array(input['CMWA25'], dtype=float), # pitching coeff
            np.array(input['CNWA25'], dtype=float)  # yawing coeff
            ])

    moments *= axes[3:]
    print(rotmat)

    forces = np.einsum('ijk,jk->ik', rotmat, forces).T
    moments = np.einsum('ijk,jk->ik', rotmat, moments).T

    output['CX'] = -forces[:, 0]
    output['CY'] = forces[:, 1]
    output['CZ'] = - forces[:, 2]

    output['Cl'] = moments[:, 0]
    output['Cm'] = moments[:, 1] * 4 # NOTE: don't know whats going on here, 
                                     # maybe the moment arm used in calculating 
                                     # it was divided by 4 to account for 25%MAC
    output['Cn'] = moments[:, 2]

    # no control deflections in dataset
    output['aileron'] = 0
    output['elevator'] = 0

    output['beta'] = - beta
    output['alpha'] = - alpha
    output['windtunnel'] = True

    output['q'] = q * scale_factor ** 2

    if augment:
        mirrored = output.copy()

        mirrored['beta'] = - mirrored['beta']
        mirrored["CY"] = - mirrored["CY"]
        mirrored["Cl"] = - mirrored["Cl"]
        mirrored["Cn"] = - mirrored["Cn"]
        mirrored['aileron'] = - mirrored['aileron']

        output = pd.concat([output, mirrored], ignore_index=True)

    return output

def plot(fig, data:pd.DataFrame, label:str = 'sim'):
    data_wt = data.where(data['windtunnel'] == True).sample(frac=.1).dropna()
    data_fs = data.where(data['windtunnel'] == False).dropna()
    print(data_wt.head())
    print(data_fs.head())
    for i, ax in enumerate(fig.axes[:6]):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        if not data_wt.empty:
            ax.scatter(data_wt['alpha'], data_wt['beta'], data_wt.iloc[:, i+6], 
                    marker='o', label=f'{label} windtunnel')
        if not data_fs.empty:   
            ax.scatter(data_fs['alpha'], data_fs['beta'], data_fs.iloc[:, i+6], 
                    marker='o', label=f'{label} freestream')
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')
        ax.set_zlabel(data.columns[i+6])
        ax.legend()

def create_interactive_aero_plot(data):
        fig = plt.figure(figsize=(18, 10))
        plt.subplots_adjust(bottom=0.2)
        
        # Pre-compute all possible filtered datasets
        aileron_values = np.unique(data['aileron'])
        elevator_values = np.unique(data['elevator'])
        filtered_datasets = {}
        for a in aileron_values:
            for e in elevator_values:
                key = (a, e)
                filtered_datasets[key] = data[
                    (data['aileron'].between(a-0.1, a+0.1)) & 
                    (data['elevator'].between(e-0.1, e+0.1))
                ]
        
        scatter_plots = []
        for i in range(6):
            ax = fig.add_subplot(2, 3, i+1, projection='3d')
            ax.set_title(f"{data.columns[i + 6]}")
            scatter = ax.scatter(data['alpha'], data['beta'], 
                                    data.iloc[:, i+6], marker='o', label='sim')
            scatter_plots.append((scatter, i))
            ax.set_xlabel('alpha')
            ax.set_ylabel('beta')
            ax.set_zlabel(data.columns[i+6])
            ax.legend()
        
        ax_aileron = plt.axes([0.2, 0.1, 0.6, 0.03])
        ax_elevator = plt.axes([0.2, 0.05, 0.6, 0.03])
        
        s_aileron = Slider(ax_aileron, 'Aileron', data['aileron'].min(), data['aileron'].max(), 
                        valinit=0, dragging=True)
        s_elevator = Slider(ax_elevator, 'Elevator', data['elevator'].min(), data['elevator'].max(), 
                        valinit=0, dragging=True)
        
        def update(val):
            # Find nearest pre-computed dataset
            nearest_a = aileron_values[np.abs(aileron_values - s_aileron.val).argmin()]
            nearest_e = elevator_values[np.abs(elevator_values - s_elevator.val).argmin()]
            filtered_data = filtered_datasets[(nearest_a, nearest_e)]
            
            for scatter, i in scatter_plots:
                scatter._offsets3d = (filtered_data['alpha'], filtered_data['beta'], 
                                    filtered_data.iloc[:, i+6])
            fig.canvas.draw_idle()
    
        s_aileron.on_changed(update)
        s_elevator.on_changed(update)
        update(None)
        plt.show(block=True)

def main():
    BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('main')[0]
    DATA_DIR = os.path.join(BASEPATH, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PARAMS_DIR = os.path.join(DATA_DIR, 'glider')

    # windtunnel simulation data (wind frame)
    wt_path = os.path.join(RAW_DATA_DIR, 'windtunnel', 'sim', 'forces_wt.npz')
    wt_sim = np.load(wt_path, allow_pickle=True)
    wt_params = json.load(open(os.path.join(PARAMS_DIR, 'glider_wt.json')))

    # freestream simulation data (body frame)
    fs_path = os.path.join(RAW_DATA_DIR, 'freestream', 'sim', 
                           'forces_912sims_avg500_symextended.npz')
    fs_sim = np.load(fs_path, allow_pickle=True) 
    fs_params = json.load(open(os.path.join(PARAMS_DIR, 'glider_fs.json')))

    
    fs_params_2 = deepcopy(fs_params)
    fs_params_2['reference_area'] = 0.225454


    
    data_wt = process_sim_dataset(wt_sim, wt_params, fs_params, 
                                  axes = np.array([[-1, -1, 1, 1, 1, 1]]).T, body=True)
    
    data_fs = process_sim_dataset(fs_sim, fs_params_2, fs_params, 
                                  axes = np.array([[-1, 1, 1, -1, 1, -1]]).T, body=False)
    
    # data_fs['Cl'] *= 4 # TODO: Why?
    data_wt['Cl'] /= 4 # TODO: Why?

    data =  pd.concat([data_fs, data_wt], ignore_index=True)

    

    if process_real:
        wt_raw_path = os.path.join(
            RAW_DATA_DIR, 
            'windtunnel', 
            'real', 
            'ProcessedData', 
            'UW2344', 
            'FinalData', 
            'finaldata_uw2344.csv'
        )
        wt_real = pd.read_csv(wt_raw_path) # load windtunnel data (wind frame)

        data_real = process_wt_dataset(wt_real, fs_params, fs_params,
                                   axes = np.array([[1, 1, 1, -1, -1, -1]]).T, body=True)
        data_real.to_csv(os.path.join(DATA_DIR, 'processed', 'data_real.csv'), index=False)
        data =  pd.concat([data, data_real], ignore_index=True)




    """To switch frames to body"""
    rotmat = R(-data['alpha'], -data['beta'])
    forces = np.einsum('ijk,jk->ik', rotmat, data.iloc[:, 6:9].to_numpy().T).T
    moments = np.einsum('ijk,jk->ik', rotmat, data.iloc[:, 9:12].to_numpy().T).T

    data['alpha'] *= -1
    data['CX'] = forces[:, 0]
    data['CY'] = forces[:, 1]
    data['CZ'] = forces[:, 2]
    data['Cl'] = moments[:, 0] * -1 / 16
    data['Cm'] = moments[:, 1]
    data['Cn'] = moments[:, 2] * -1
    
    create_interactive_aero_plot(data)

    output_path = os.path.join(DATA_DIR, 'processed', 'data_sim.csv')
    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
