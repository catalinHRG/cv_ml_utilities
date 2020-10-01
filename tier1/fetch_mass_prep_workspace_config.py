from argparse import ArgumentParser, Namespace
from json import dump

from os.path import isdir, join, isfile
from os import listdir

tray_config_template = {

	"phase1":{

		"dataset_config": {

			"candidates": [],

			"label_groups":{

			},

			"aug_config": {

				"common":{

					"rotate_up_to_360_by": 0,
				    "baseline_augs_per_image": 0,
				    "flip_left_right": True,
				    "negative_filter": False,
				    "rotate_180": False,
				    "flip_up_down": True,
				    "clean_existing_augs": False

				},	

			},

			"base_image_height": 512,
			"base_image_width": 1024

		},

		"model_config": { 

			"retinanet_flag": True,
			"ssd_flag": False,
			"num_interpolated_convs_between_encoding_and_predictor": 2,
			"unmatched_threshold_values": [0.5, 0.45, 0.4, 0.35, 0.3],

			"faster_rcnn_flag": True,
			"striped_faster_rcnn_flag": True,

			"max_steps_per_attempt": -1,

			"blacklist": [],
			"target": []

		}
	
	},

	"phase2":{

		"dataset_config": {

			"candidates": [],

			"aug_config":{

				"common":{

					"rotate_up_to_360_by": 0,
				    "baseline_augs_per_image": 5,
				    "flip_left_right": False,
				    "negative_filter": True,
				    "rotate_180": False,
				    "flip_up_down": False,
				    "clean_existing_augs": False

				},

			},

			"square_images": False

		},

		"model_config": { 

			"retinanet_flag": False,
			"ssd_flag": False,
			"num_interpolated_convs_between_encoding_and_predictor": 2,
			"unmatched_threshold_values": [0.5, 0.45, 0.4, 0.35, 0.3],

			"faster_rcnn_flag": True,
			"striped_faster_rcnn_flag": True,

			"max_steps_per_attempt": -1,

			"blacklist": [],
			"target": []

		}
	
	},

}


def run(args):

	if not isdir(args.root_dir):

		msg = args.root_dir + ' does not exist or is not a directory.'
		raise IOError(msg)

	tray_ids = listdir(args.root_dir)

	mass_prep_workspace_config_dict = {}

	for tray_id in tray_ids:

		if tray_id == 'stash': continue

		_path = join(args.root_dir, tray_id)

		if isfile(_path): continue

		mass_prep_workspace_config_dict[ tray_id ] = tray_config_template

	with open(join(args.root_dir, 'config.json'), 'w') as fd:

		dump(mass_prep_workspace_config_dict, fd, indent = 3, sort_keys = True)

	return 0

def main():

	argp = ArgumentParser( description = '' )
    
	argp.add_argument('--root_dir', '-i', required = True, help = 'root directory where the exports generated using the Biomet AnnotationTool for multiple trays can be found.')
	
	args = argp.parse_args()

	_ = run(args)

if __name__ == '__main__': main()