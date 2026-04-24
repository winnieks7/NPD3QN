# Perception-Aware Cooperative Path Planning for Multi-UAV Systems in Urban Wind Fields

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2023a%2B-blue.svg)](https://www.mathworks.com/)

This repository contains the official MATLAB implementation of the **NPD3QN** algorithm, as presented in the paper:
**"Perception-Aware Cooperative Path Planning for Multi-UAV Systems in Urban Wind Fields via Deep Reinforcement Learning"**.

## 📖 Overview
Deploying multiple UAVs in low-altitude urban airspaces faces severe challenges due to complex building clusters and stochastic wind field disturbances. This project introduces **NPD3QN**, an enhanced Dueling Double Deep Q-Network. Key features include:
* **Kinematic-Aware Environment:** Integrates 3D urban building models (`city.obj`) with spatial wind field vectors.
* **$N$-step Update Strategy:** Enhances the characterization of long-term returns for cooperative trajectory optimality.
* **Task-Oriented PER:** An improved Prioritized Experience Replay mechanism to suppress negative experiences and prevent conservative policy freezing.

## 📂 Repository Structure
* `main_npd3qn.m`: The main entry point for the project.
* `npd3qn_env.m`: Custom Markov Decision Process (MDP) and reward function definition.
* `npd3qn_net.m`: Dueling network architecture construction.
* `npd3qn_training.m`: Training loop with $N$-step and improved PER logic.
* `npd3qn_inference.m`: Evaluation script for pre-trained models.
* `npd3qn_visualize.m`: 3D rendering of the urban terrain, wind field, and UAV trajectories.
* `city.obj` / `city.mtl`: 3D urban geometric models.

## 🛠️ Prerequisites
* **MATLAB** (R2023a or later recommended)
* Deep Learning Toolbox
* Reinforcement Learning Toolbox

## 🚀 How to Run

**1. Clone the repository:**
```bash
git clone [https://github.com/winnieks7/NPD3QN.git](https://github.com/winnieks7/NPD3QN.git)
cd NPD3QN
