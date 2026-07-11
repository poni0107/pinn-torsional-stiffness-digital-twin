# pinn-torsional-stiffness-digital-twin
A physics-informed neural network-based digital twin for estimating torsional stiffness degradation in motor-load oscillatory systems.

# PINN-Based Digital Twin for Torsional Stiffness Estimation

A Physics-Informed Neural Network (PINN)-based framework for digital twin
development and torsional stiffness estimation in motor-load oscillatory systems.

---

## Overview

This repository contains the implementation of a Physics-Informed Neural
Network (PINN) framework for estimating torsional stiffness in a motor-load
oscillatory system.

The proposed approach combines measured system responses with the governing
physical equations of motion to estimate unknown or degraded torsional
stiffness. The developed framework represents a digital twin capable of
monitoring drivetrain dynamics and supporting predictive maintenance in
electric vehicle applications.

---

## Features

- Numerical simulation of the motor-load oscillatory system
- Forward modelling using Physics-Informed Neural Networks (PINNs)
- Inverse estimation of torsional stiffness
- Digital twin development for drivetrain monitoring
- Stiffness degradation analysis
- Robustness evaluation under noisy measurements
- Sparse-data learning using embedded physical constraints

---

## Repository Structure

```
pinn-torsional-stiffness-digital-twin/
│
├── src/                 # Source code
├── data/                # Input datasets
├── results/             # Figures and experimental results
├── docs/                # Mathematical model and documentation
├── notebooks/           # Experimental notebooks
├── paper/               # Conference paper
├── requirements.txt
└── README.md
```

---

## Research Paper

This repository accompanies the research paper:

**Digital Twin Design for Torsional Stiffness Estimation in Motor-Load Oscillatory Systems Using Physics-Informed Neural Networks**

submitted to

**The 11th International Congress Motor Vehicles & Motors (MVM 2026)**

---

## Authors

**Marijana Jeremić**  
Faculty of Engineering, University of Kragujevac, Serbia

**Lazar Krstić**  
Faculty of Science, University of Kragujevac, Serbia

**Miloš Ivanović**  
Faculty of Information Studies, Novo Mesto, Slovenia

**Mihailo Lazarević**  
Faculty of Mechanical Engineering, University of Belgrade, Serbia

**Milan Matijević**  
Faculty of Engineering, University of Kragujevac, Serbia

---

## Citation

If you use this repository in your research, please cite:

Jeremić, M., Krstić, L., Ivanović, M., Lazarević, M., & Matijević, M.
*Digital Twin Design for Torsional Stiffness Estimation in Motor-Load Oscillatory Systems Using Physics-Informed Neural Networks.*
MVM 2026.

---

## Acknowledgment

This work builds upon previous research on Physics-Informed Neural Networks
for motor-load oscillatory systems. The implementation has been extended to
support inverse torsional stiffness estimation, digital twin development,
and stiffness degradation analysis.

---

## License

This repository is intended for academic and research purposes.
