import os
import sys
import json
import copy
import argparse
import subprocess

import torch

if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str,
					default='/misc/vlgscratch4/WilsonGroup/ngruver/logs/physics-uncertainty',
                   	help='directory containing configs and slurm scripts')

def write_config(exp_dir, C, num_epochs):
	config = {}

	exp_name = "calib_c_{}_e_{}".format(C, num_epochs)
	config["exp_dir"] = exp_dir
	config["exp_name"] = exp_name

	config["num_bodies"] = 2
	config["segment_len"] = C
	config["num_epochs"] = num_epochs
	config["posterior_samples"] = 10

	config_fn = os.path.join(exp_dir, "{}.json".format(exp_name))
	with open(config_fn, 'w') as fd:
	    json.dump(config, fd)

	return config

def run_slurm_script(config):
	py_command_str = \
"""/misc/vlgscratch4/WilsonGroup/ngruver/miniconda3/envs/phy-unc/bin/python3 \
calibration_exp.py --exp_dir {} --exp_name {}""".format(config["exp_dir"], config["exp_name"])

	basename = os.path.join(config["exp_dir"], config["exp_name"])
	output_file = "{}.out".format(basename)
	error_file = "{}.err".format(basename)

	slurm_script_str = \
"""#!/bin/sh
#SBATCH --job-name={}
#SBATCH --open-mode=append
#SBATCH --output={}
#SBATCH --error={}
#SBATCH --export=ALL
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_12gb
#SBATCH --mem=16G
#SBATCH -c 4

{}""".format(config["exp_name"], output_file,
			 error_file, py_command_str)

	script_fn = "{}.sh".format(basename)
	with open(script_fn, "w") as fd:
		fd.write(slurm_script_str)

	subprocess.run(["sbatch", script_fn])

def main():
	args = parser.parse_args()
	
	if not os.path.exists(args.exp_dir):
		os.mkdir(args.exp_dir)

	for C in [5, 10, 15, 20]:
		for num_epochs in [10, 50, 100]:
			config = write_config(args.exp_dir, C, num_epochs)
			run_slurm_script(config)

if __name__ == "__main__":
	main()