# DDRM

## Usage:

### Training:

pip3 install -r requirements.txt

python3 setup.py

export XRT_TPU_CONFIG='localservice;0;localhost:51011'

python3 main.py -mp models -n 0 -d lsun -m tpu

### Testing: