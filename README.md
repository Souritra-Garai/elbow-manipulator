# 2R Elbow Manipulator

Simulation of 2R Elbow Manipulator Robot

![Alt Text](https://github.com/Souritra-Garai/elbow-manipulator/blob/main/docs/2R-sim.gif)

## Simulation Hierarchy

The underlying differential equations for the system is solved at very high frequency compared to controller frequency. This, in priciple, emulates the real world condition, where the torque-based robot controller runs at a finite frequency while the system continuosly evolves with time.