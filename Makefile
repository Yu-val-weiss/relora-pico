.PHONY: *

get_intr:
	sintr -A CAINES-SL3-GPU -p ampere -N1 -t 1:0:0 --qos=INTR --mail-type=ALL --gres=gpu:1