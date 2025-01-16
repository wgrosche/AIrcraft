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
