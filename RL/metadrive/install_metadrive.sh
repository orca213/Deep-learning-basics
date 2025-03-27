#!/bin/bash

apt-get update -y
apt-get install -y libgl1-mesa-glx
apt-get install -y libglib2.0-0

# requirements
pip install stable_baselines3

# install metadrive
git clone https://github.com/metadriverse/metadrive.git
cd metadrive

pip install -e .