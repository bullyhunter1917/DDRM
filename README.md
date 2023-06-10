# DDRM

## Usage:

export XRT_TPU_CONFIG='localservice;0;localhost:51011'

python3 main.py -m models -n 0 -d <dataset>

Where dataset is one of lsun, cifar10

### or directly from googlecloud shell:

gcloud compute tpus tpu-vm ssh ddrm-tpu --zone europe-west4-a --worker=all --command="cd test/DDRM;
export XRT_TPU_CONFIG='localservice;0;localhost:51011';  python3 main.py -m models -n 0 -d lsun
"
