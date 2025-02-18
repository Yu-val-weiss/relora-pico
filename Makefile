.PHONY: *

get_intr:
	sintr -A CAINES-SL3-GPU -p ampere -N1 -t 1:0:0 --qos=INTR --mail-type=ALL --gres=gpu:1

get_two:
	sintr -A CAINES-SL3-GPU -p ampere -N1 -t 1:0:0 --qos=INTR --mail-type=ALL --gres=gpu:2

get_four:
	sintr -A CAINES-SL3-GPU -p ampere -N1 -t 1:0:0 --qos=INTR --mail-type=ALL --gres=gpu:4

get_excl:
	sintr -A CAINES-SL3-GPU -p ampere -N1 -t 1:0:0 --qos=INTR --mail-type=ALL --exclusive