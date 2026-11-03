# Genesis AI Waste Sorting Robot

Physical AI Application Course – Final Project

## Overview

An AI-powered robotic waste sorting system developed
using Genesis and the Franka Panda robotic arm.

The robot classifies three types of waste based on
shape features and sorts them into designated bins.

## Features

- Shape-based waste classification using Random Forest
- Three categories: Plastic Bottle, Paper, Metal Can
- Robotic pick-and-place using Franka Panda
- Inverse Kinematics (IK) for arm control
- Simulated suction grasping
- Color-coded collection bins
- Sorting result verification

## Technologies

- Python
- Google Colab
- Genesis
- scikit-learn
- OpenCV

## How to Run

1. Open Genesis_Waste_Sorting_Robust_Colab.ipynb
   in Google Colab.
2. Install the required dependencies.
3. Upload genesis_shape_sorter_robust.py.
4. Run the simulation.

## Demo Video

YouTube: 
Only movie
https://youtu.be/tXQ3KOx5UkY

Movie with cover & subtitles
https://youtu.be/4-L6U0694u4

## Results

Successfully sorted three waste objects
into their designated collection bins.

## Limitations

- Waste classification uses shape features extracted
  with simulator-provided segmentation.
- Grasping is simulated using a suction constraint.
- The system has not been tested on a physical robot.
