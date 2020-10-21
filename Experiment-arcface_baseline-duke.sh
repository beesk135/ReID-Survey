# Dataset: dukemtmc
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# last stride 1
# with center loss
# weight regularized triplet loss
# generalized mean pooling
# non local blocks
python3 tools/main.py --config_file='configs/arcface_baseline.yml' MODEL.DEVICE_ID "('0')" \
DATASETS.NAMES "('dukemtmc')" OUTPUT_DIR "('./log/dukemtmc/arcface_baseline-1')"