# Wake-T: A fast tracking code for plasma-based accelerators.

<p align="center">
  <img alt="Wake-T logo" src="other/WakeT_logo_dot.png" width="300px" />
</p>

## Overview
 `Wake-T` (**Wake**field particle **T**racker) is a tracking code for plasma wakefield accelerators which aims at providing a fast alternative to Particle-in-Cell (PIC) simulations. Instead of relying on the computationally-expensive PIC algorithm for simulating the plasma wakefields and the beam evolution, `Wake-T` uses an analytical or numerical (Runge-Kutta) solver to track the evolution of the beam electrons in the wakefields, which, at the same time, are computed from reduced models. This allows for a significant speed-up of the simulations, which can be performed in a matter of seconds instead or hours/days. An overview of this strategy can be seen in the following figure:
<p align="center">
  <img alt="Wake-T logo" src="other/plasma_tracking.png" width="600px" />
</p>

The main drawback of this approach is a reduced accuracy of the results, compared to a PIC code, particularly if the assumptions of the reduced wakefield models are not satisfied. Although more models are planned to be included in the future, some of the main current limitations of the code are the lack of beam-loading effects, realistic laser evolution and electron self-injection.

In addition to plasma-acceleration stages, `Wake-T` can also simulate active plasma lenses, drifts, dipoles, quadrupoles and sextupoles, allowing for the simulation of complex beamlines. The tracking along the drifts and magnets is performed using second-order transfer matrices, and CSR effects can be included by using a 1D model. This matrix approach and the CSR model are based on a streamlined version of the [`Ocelot`](https://github.com/ocelot-collab/ocelot) implementation.

## Installation
1) If you don't have Python 3 already installed, download the latest version, for example, from [here](https://www.python.org/downloads/release/python-352/). It is recommended to create a virtual environment for `Wake-T` (you can see how [here](https://docs.python.org/3/library/venv.html), for example). Remember to activate the new environment before proceeding with the installation.

2) Clone this repository to a directory in your computer using `git`
```bash
git clone https://github.com/AngelFP/Wake-T.git
```
or simply download the code from [here](https://github.com/AngelFP/Wake-T/archive/master.zip) and unzip it.

3) If you haven't already, open a terminal in the newly created folder and perform the installation with
```bash
python setup.py install
```

## References

[1] - A. Ferran Pousa et al., *Intrinsic energy spread and bunch length growth in plasma-based accelerators due to betatron motion*, [Sci. Rep. **9**, 17690 ](https://doi.org/10.1038/s41598-019-53887-8) (2019).

[2] - A. Ferran Pousa et al., *Wake-T: a fast particle tracking code for plasma-based accelerators*, [J. Phys.: Conf. Ser. **1350** 012056](https://iopscience.iop.org/article/10.1088/1742-6596/1350/1/012056) (2019).