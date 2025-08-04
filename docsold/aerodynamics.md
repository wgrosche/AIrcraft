# Aerodynamics

This section concerns itself with the function `coefficients` in the [aircraft dynamics](https://github.com/wgrosche/AIrcraft/blob/main/src/aircraft/dynamics/aircraft.py). 

Aerodynamic coefficients formalise the dependence of the aerodynamic forces and moments on the [dynamic pressure](https://en.wikipedia.org/wiki/Dynamic_pressure) and the reference area. An example can be found in the page on the [lift coefficient](https://en.wikipedia.org/wiki/Lift_coefficient).

We derive forces from the coefficients as:

$F_i = \bar{q}SC_i$

And moments:

$M_i = \bar{q}SC_ir$

Where r is the moment arm for the lifting surface (span for roll and length for pitch and yaw)


We implement 4 approaches for modelling aerodynamic coefficients.

## Constant

A simplified model that uses standard values for the coefficients.


        ```
        CD0 = 0.02
        CD_alpha = 0.3

        CL0 = 0.0
        CL_alpha = 5.0  # lift per rad

        CY_beta = -0.98

        Cl_aileron = 0.08
        Cl_p = -0.05  # roll damping

        Cm_elevator = -1.2
        Cm_q = -.5  # pitch damping

        Cn_rudder = -0.1
        Cn_r = -0.05  # yaw damping

        # Core coefficient calculations
        CD = CD0 + CD_alpha * self._alpha**2
        CL = CL0 + CL_alpha * self._alpha
        CY = CY_beta * self._beta

        Cl = Cl_aileron * self._aileron  / (2*np.pi)+ Cl_p * p
        Cm = Cm_elevator * self._elevator  / (2*np.pi) + Cm_q * q
        Cn = Cn_rudder * self._rudder / (2*np.pi) + Cn_r * r

        outputs = ca.vertcat(-CD, CY, -CL, Cl, Cm, Cn)
        ```

## [Linear](https://github.com/wgrosche/AIrcraft/blob/main/main/linear_coefficients.py)

In the linear case we model aerodynamic coefficients as functions $C_i(\bar{q}, \alpha, \beta, \delta a, \delta e) = a_1\bar{q} + a_2\alpha + a_3\beta + a_4\delta a + a_5 \delta e$. 

Drag is modelled separately as $C_D = C_{D,0} + kC_L^2$ to capture the quadratic lift dependency.

## [Polynomial](https://github.com/wgrosche/AIrcraft/blob/main/main/polynomial_model.py)

Allows higher order terms in the expansion and neglects contribution of the dynamic pressure $\bar{q}$. The force dependence on the dynamic pressure should mostly be captured by the definition of the aerodynamic coefficients and we are likely capturing undesirable dataset fluctuations that do not extrapolate beyond the fairly narrow band of velocities for which data are available.

## [Neural Surrogate](https://github.com/wgrosche/AIrcraft/blob/main/main/train.py)

Here the coefficients are modelled by a small mlp. Due to the difficulties in integrating larger models with traditional MPC only fairly limited work has been done on developing this model. Further work will be necessary to embed shape information at a later date.

# Damping

The models above are formed from a static dataset and do not capture dynamic effects at the lifting surfaces. This underestimates the effective angles of attack and sideslip when calculating the aerodynamic moments. To counteract this we introduce a simple approximation of the effective angles:

### Rolling Moment $\(C_l\)$

The rolling moment is the result of lift differences between the left and right wings. When in motion behaviour deviates from the static case primarily due to the roll-and yaw-rates ($p$, $r$ resp.). 

When rolling the effective angle of attack for the wings changes as:

Right Wing:
$\alpha_{W,r} = \arctan(\frac{v_z + bp/4}{v_x})$

Left Wing:
$\alpha_{W,l} = \arctan(\frac{v_z - bp/4}{v_x})$

And similarly the dynamic pressure due to yaw rate:

Right Wing:
$\bar{q}_r= 0.5 * 1.225 * (v_x - br/4)$

Left Wing:
$\bar{q}_l= 0.5 * 1.225 * (v_x + br/4)$

The lever arm of b/4 is chosen for simplicity (b is full span of the plane and we assume the lift attacks at mid-wing). The roll rate thus damps further roll whereas the yaw rate accelerates it.

### Pitching Moment

The relevant surface for the pitching moment is the elevator. It is primarily affected by the pitching rate and the effect is damping in nature.

The angle of attack at the elevator changes as:
$\alpha_{e} = \arctan(\frac{v_z + r_e q/4}{v_x})$

, where $r_e$ is the moment arm for the elevator (distance between centre of pressure and elevator).

### Yawing Moment

With similar reasoning the damping effects of the rudder's yaw rate yield the 'effective angle of attack' for the rudder (which amounts to an angle of sideslip in the dataset):

$\arcsin\(\frac{v_y + r_rr}{|\vec{v}|r_rr}\)$

Where the confusingly chosen $r_r$ is the rudder moment arm.

# Dataset limitations

Due to limits on the data collected from CFD simulations and windtunnel tests the accuracy of our models is limited outside of the flight envelope:

$\alpha \in [-10\degree, 10\degree]$

$\beta \in [-10\degree, 10\degree]$

$v_x \in [20m/s, 80m/s]$

To account for this we introduce clipping on the derived values above and simulate stall:

We define the stall thresholds for $\(\alpha\)$ and $\(\beta\)$ as:

$\alpha_{\text{stall}} = \beta_{\text{stall}} = \pm 10\degree$

A sigmoid function is used to smoothly decrease effectiveness as the magnitude of the angle exceeds the stall threshold. This is applied to the pitch moment as well as the lift.
