# Dataset: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# last stride 1
# with center loss
# weight regularized triplet loss
# generalized mean pooling
# non local blocks
python3 tools/main.py --config_file='/Users/thanittativannarat/capstone/ReID-Survey/configs/AGW_circle.yml' MODEL.DEVICE_ID "('0')" \
DATASETS.NAMES "('market1501')" OUTPUT_DIR "('./log/market1501/AGW-circle-1')"