{\rtf1\ansi\ansicpg936\cocoartf2869
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Perception-Aware Cooperative Path Planning for Multi-UAV Systems in Urban Wind Fields\
\
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)\
[![MATLAB](https://img.shields.io/badge/MATLAB-R2023a%2B-blue.svg)](https://www.mathworks.com/)\
\
This repository contains the official MATLAB implementation of the **NPD3QN** algorithm, as presented in the paper:\
**"Perception-Aware Cooperative Path Planning for Multi-UAV Systems in Urban Wind Fields via Deep Reinforcement Learning"**.\
\
## \uc0\u55357 \u56534  Overview\
Deploying multiple UAVs in low-altitude urban airspaces faces severe challenges due to complex building clusters and stochastic wind field disturbances. This project introduces **NPD3QN**, an enhanced Dueling Double Deep Q-Network. Key features include:\
* **Kinematic-Aware Environment:** Integrates 3D urban building models (`city.obj`) with spatial wind field vectors.\
* **$N$-step Update Strategy:** Enhances the characterization of long-term returns for cooperative trajectory optimality.\
* **Task-Oriented PER:** An improved Prioritized Experience Replay mechanism to suppress negative experiences and prevent conservative policy freezing.\
\
## \uc0\u55357 \u56514  Repository Structure\
* `main_npd3qn.m`: The main entry point for the project.\
* `src/npd3qn_env.m`: Custom Markov Decision Process (MDP) and reward function definition.\
* `src/npd3qn_net.m`: Dueling network architecture construction.\
* `src/npd3qn_training.m`: Training loop with $N$-step and improved PER logic.\
* `src/npd3qn_inference.m`: Evaluation script for pre-trained models.\
* `src/npd3qn_visualize.m`: 3D rendering of the urban terrain, wind field, and UAV trajectories.\
* `assets/`: Contains the 3D urban geometric models (`city.obj`, `city.mtl`).\
\
## \uc0\u55357 \u57056 \u65039  Prerequisites\
* **MATLAB** (R2023a or later recommended)\
* Deep Learning Toolbox\
* Reinforcement Learning Toolbox\
\
## \uc0\u55357 \u56960  How to Run\
\
**1. Clone the repository:**\
```bash\
git clone [https://github.com/YourUsername/NPD3QN.git](https://github.com/YourUsername/NPD3QN.git)\
cd NPD3QN}