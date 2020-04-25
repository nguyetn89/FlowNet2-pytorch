#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-0:20
#SBATCH --output=./log_install_%N-%j.out
module load python/3.6
module load gcc/8.3.0 cuda/10.1
source ~/ENV/bin/activate
pip install pypng

cd ./networks/correlation_package
rm -rf *_cuda.egg-info build dist __pycache__
python setup.py install

cd ../resample2d_package
rm -rf *_cuda.egg-info build dist __pycache__
python setup.py install

cd ../channelnorm_package
rm -rf *_cuda.egg-info build dist __pycache__
python setup.py install

cd ..
