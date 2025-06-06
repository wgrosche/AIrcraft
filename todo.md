# TODO

## Initialisation

Initialise as point mass model with thruster that can deliver the maximal forces from our dataset to give us min-radius turns.

Assume orientation is just aligned with velocity (alpha, beta = 0).

Think about what is necessary to initialise controls.

## Plotting

Update the rpg plotting routine to show all the faff from the current plotting routine as quickly as possible.

## Waypoints

Investigate TOGT style formulation for waypoint constraint.

## Control Node dt

Play with ways to assign control nodes durations (dt) based on the initialised trajectory (eg. dt inversely propto change in velocity (thrust))

## Control smoothing

Play with different methods of penalising control input changes to see whether they can be smoothed.

## ablation

Ablation results:


opts = AircraftOpts(coeff_model_type='default', coeff_model_path='', aircraft_config=aircraft_config, physical_integration_substeps=1)
trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
controller_opts = {'time':'fixed', 'quaternion':'', 'integration':'explicit’}

final time: 2.98s
Result:
Number of Iterations....: 90 (scaled) (unscaled) Objective...............: 5.4240283206263242e-21 5.4240283206263242e-21 Dual infeasibility......: 1.6327679951854356e-07 1.6327679951854356e-07 Constraint violation....: 5.0681809498135983e-07 1.3553530469446428e-06 Variable bound violation: 0.0000000000000000e+00 0.0000000000000000e+00 Complementarity.........: 1.0000000000000003e-11 1.0000000000000003e-11 Overall NLP error.......: 5.0681809498135983e-07 1.3553530469446428e-06 Number of objective function evaluations = 236 Number of objective gradient evaluations = 91 Number of equality constraint evaluations = 250 Number of inequality constraint evaluations = 250 Number of equality constraint Jacobian evaluations = 91 Number of inequality constraint Jacobian evaluations = 91 Number of Lagrangian Hessian evaluations = 90 Total seconds in IPOPT = 423.580 EXIT: Optimal Solution Found. solver : t_proc (avg) t_wall (avg) n_eval callback_fun | 411.49 s ( 4.52 s) 412.21 s ( 4.53 s) 91 nlp_f | 336.00us ( 1.42us) 267.09us ( 1.13us) 236 nlp_g | 163.84ms (655.36us) 164.38ms (657.52us) 250 nlp_grad_f | 954.00us ( 10.37us) 895.74us ( 9.74us) 92 nlp_hess_l | 5.01 s ( 55.72ms) 5.02 s ( 55.79ms) 90 nlp_jac_g | 1.31 s ( 14.19ms) 1.31 s ( 14.24ms) 92 total | 425.22 s (425.22 s) 423.59 s (423.59 s) 1 {'success': True} state (13, 298) control (6, 298) lam: None mu: None nu: None Final State: [-4.91443960e-04 3.00007355e+01 -1.79999243e+02 -3.40474317e+01 -3.08113900e+01 -1.36658629e+01 1.07592461e-01 -6.54142188e-01 6.65115558e-01 -3.43760932e-01 -3.30126758e-02 6.28510264e-02 8.32389179e-02] Final Control: [-0.48060452 -0.48060452 -0.48060452 -0.48060452 -0.48060452 -0.48060452] Final Forces: [-7.48988, 4.39679, -94.534]





Variable time (progress): added time to loss

traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    # poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
    opts = AircraftOpts(coeff_model_type='default', coeff_model_path='', aircraft_config=aircraft_config, physical_integration_substeps=1)

    aircraft = Aircraft(opts = opts)
    trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
    aircraft.com = np.array(trim_state_and_control[-3:])
    filepath = Path(DATAPATH) / 'trajectories' / 'basic_test.h5'

    # controller_opts = {'time':'fixed', 'quaternion':'', 'integration':'explicit'}
    controller_opts = {'time':'progress', 'quaternion':'', 'integration':'explicit'}
    controller = Controller(aircraft=aircraft, filepath=filepath, implicit=True, progress = False, opts = controller_opts)
    guess = controller.initialise(ca.DM(trim_state_and_control[:aircraft.num_states]))
    controller.setup(guess)
    
    controller.solve()
    final_state = controller.opti.debug.value(controller.state)[:, -1]
    final_control = controller.opti.debug.value(controller.control)[:, -1]
    print("Final State: ", final_state, " Final Control: ", final_control, " Final Forces: ", aircraft.forces_frd(final_state, final_control))
    plt.show(block = True)


Cannot call restoration phase at point that is almost feasible (violation 5.684342e-14).
Abort in line search due to no other fall back.

Number of Iterations....: 506

                                   (scaled)                 (unscaled)
Objective...............:   9.1306926583149572e-01    9.1306926583149572e-01
Dual infeasibility......:   3.2541029993617926e-03    3.2541029993617926e-03
Constraint violation....:   5.6843418860808015e-14    5.6843418860808015e-14
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   1.0000002167517520e-11    1.0000002167517520e-11
Overall NLP error.......:   1.0168435686491789e-04    3.2541029993617926e-03


Number of objective function evaluations             = 600
Number of objective gradient evaluations             = 485
Number of equality constraint evaluations            = 601
Number of inequality constraint evaluations          = 601
Number of equality constraint Jacobian evaluations   = 509
Number of inequality constraint Jacobian evaluations = 509
Number of Lagrangian Hessian evaluations             = 507
Total seconds in IPOPT                               = 117.526

EXIT: Error in step computation!
      solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
callback_fun  |  11.46 s ( 23.72ms)  11.40 s ( 23.60ms)       483
       nlp_f  |   2.09ms (  3.49us)   1.93ms (  3.21us)       600
       nlp_g  | 402.59ms (669.87us) 399.87ms (665.34us)       601
  nlp_grad_f  |   6.20ms ( 12.76us)   5.83ms ( 11.99us)       486
  nlp_hess_l  |  31.02 s ( 61.30ms)  30.85 s ( 60.97ms)       506
   nlp_jac_g  |   7.81 s ( 15.32ms)   7.76 s ( 15.21ms)       510
       total  | 134.42 s (134.42 s) 117.53 s (117.53 s)         1
Solver failed: Error in Opti::solve [OptiNode] at .../casadi/core/optistack.cpp:217:
.../casadi/core/optistack_internal.cpp:1336: Assertion "return_success(accept_limit)" failed:
Solver failed. You may use opti.debug.value to investigate the latest values of variables. return_status is 'Error_In_Step_Computation'
{'success': False}
state (13, 298) control (6, 298)
lam:  None mu:  None nu:  None
Final State:  [ 2.71825097e-10  3.00000000e+01 -1.80000000e+02 -3.73314043e+01
  6.39928972e+01  2.77826364e+01 -2.65774055e-01 -6.11580371e-01
  4.45413540e-01  5.97477466e-01 -5.58006954e-01  7.18165869e-01
  1.04268472e+00]  Final Control:  [-4.99999779 -4.99999779 -4.9999978  -5.         -4.99999999 -5.00000001]  Final Forces:  [-51.5765, -156.479, -1472.69]

Variable time (variable): Final time: 0.83 s (very strange results)
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    # poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
    opts = AircraftOpts(coeff_model_type='default', coeff_model_path='', aircraft_config=aircraft_config, physical_integration_substeps=1)

    aircraft = Aircraft(opts = opts)
    trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
    aircraft.com = np.array(trim_state_and_control[-3:])
    filepath = Path(DATAPATH) / 'trajectories' / 'basic_test.h5'

    # controller_opts = {'time':'fixed', 'quaternion':'', 'integration':'explicit'}
    controller_opts = {'time':'variable', 'quaternion':'', 'integration':'explicit'}
    controller = Controller(aircraft=aircraft, filepath=filepath, progress = False, opts = controller_opts)
    guess = controller.initialise(ca.DM(trim_state_and_control[:aircraft.num_states]))
    controller.setup(guess)
    controller.logging = False
    
    controller.solve()
    final_state = controller.opti.debug.value(controller.state)[:, -1]
    final_control = controller.opti.debug.value(controller.control)[:, -1]
    print("Final State: ", final_state, " Final Control: ", final_control, " Final Forces: ", aircraft.forces_frd(final_state, final_control))
    plt.show(block = True)


Number of Iterations....: 2200

                                   (scaled)                 (unscaled)
Objective...............:   4.2094131267795581e-01    1.2544051117803083e+02
Dual infeasibility......:   8.5167225740273526e-03    2.5379833270601511e+00
Constraint violation....:   1.4210854715202004e-14    2.8421709430404007e-14
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   1.0000000668132952e-11    2.9800001991036201e-09
Overall NLP error.......:   1.0000000668132952e-11    2.5379833270601511e+00


Number of objective function evaluations             = 2635
Number of objective gradient evaluations             = 2201
Number of equality constraint evaluations            = 2649
Number of inequality constraint evaluations          = 2649
Number of equality constraint Jacobian evaluations   = 2202
Number of inequality constraint Jacobian evaluations = 2202
Number of Lagrangian Hessian evaluations             = 2200
Total seconds in IPOPT                               = 5470.085

EXIT: Solved To Acceptable Level.
      solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
callback_fun  |  14.24 s (  6.47ms)  15.00 s (  6.82ms)      2200
       nlp_f  |  11.69ms (  4.44us)  11.12ms (  4.22us)      2635
       nlp_g  |   2.47 s (931.20us)   2.57 s (969.00us)      2649
  nlp_grad_f  |  34.14ms ( 15.50us)  34.27ms ( 15.56us)      2202
  nlp_hess_l  | 176.98 s ( 80.48ms)   1.23ks (560.43ms)      2199
   nlp_jac_g  |  45.23 s ( 20.53ms)  47.36 s ( 21.50ms)      2203
       total  |   3.11ks (  3.11ks)   5.47ks (  5.47ks)         1
{'success': True}
state (13, 298) control (6, 298)
lam:  None mu:  None nu:  None
Final State:  [ 3.75432597e-32  3.00000000e+01 -1.80000000e+02 -3.76429340e+01
  6.47602149e+01  2.58801853e+01 -2.83827959e-01 -6.36278543e-01
  4.24668182e-01  5.78203029e-01 -5.56987777e-01  7.14029880e-01
  1.04398353e+00]  Final Control:  [-4.9999613  -4.99996123 -4.99996145 -4.99999856 -4.99999694 -4.99999946]  Final Forces:  [-49.6078, -157.228, -1421.45]