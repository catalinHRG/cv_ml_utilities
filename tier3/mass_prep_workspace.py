from argparse import ArgumentParser, Namespace
from sys import path as PYTHONPATH

PYTHONPATH.insert(0, '/media/sml/catalinh/misc/utils/exe/tier2')

from prep_phase1_workspace import run as prep_p1w
from prep_phase2_workspace import run as prep_p2w

from os.path import isdir, isfile, join
from os import listdir

from json import load, dump

def run(args):

	# TODO: before running each particular prep_workspace utility for each subset, check wether it has been succsefully preped by interogting a json found in either p1 or p2 standardized workspace

	if not isdir(args.root_dir):

		msg = args.root_dir + ' does not exist or is not a directory.'
		raise IOError(msg)

	if args.config: config_file = args.config

	else: config_file = join(args.root_dir, 'config.json')

	if not isfile(config_file):

		msg = config_file + ' does not exist or is not a file.'
		raise IOError(msg)

	with open(config_file, 'r') as fd:

		config = load(fd)

	content = listdir(args.root_dir)

	for element in content:

		if element == "stash": continue

		if isfile( join(args.root_dir, element) ): continue

		tray_dataset = join( args.root_dir, element )

		tray_config = config[ element ]

		phase1_config = tray_config.get( 'phase1', {} )

		if phase1_config:

			p1_dataset_config = phase1_config.get('dataset_config', {})

			label_groups = p1_dataset_config.get('label_groups', None)

			base_image_height = p1_dataset_config.get('base_image_height', None)
			
			if base_image_height is None:

				msg = 'Expected key base_image_height with integer value'
				raise KeyError(msg)

			base_image_width = p1_dataset_config.get('base_image_width', None)		

			if base_image_width is None:

				msg = 'Expected key base_image_width with integer value'
				raise KeyError(msg)
				
			prep_p1w_args = Namespace(

				root_dir = tray_dataset,
				label_groups = label_groups,
				candidates = p1_dataset_config.get('candidates', []),
				base_image_height = base_image_height,
				base_image_width = base_image_width

			)

			p1_workspace_dir = prep_p1w( prep_p1w_args )

			p1_aug_config = p1_dataset_config.get('aug_config', {})

			with open( join( p1_workspace_dir, 'aug_config.json'), 'w' ) as fd:

				dump(p1_aug_config, fd, indent = 3, sort_keys = True)

			p1_model_config = phase1_config.get('model_config', {})

			with open( join( p1_workspace_dir, 'model_config.json' ), 'w') as fd:

				dump(p1_model_config, fd, indent = 3, sort_keys = True)

		else:

			print( 'There is no phase1 config for', element, 'skipping ...')
			continue

		phase2_config = tray_config.get( 'phase2', {} )

		if phase2_config:

			p2_dataset_config = phase2_config.get('dataset_config', {})

			candidates = p2_dataset_config.get('candidates', [])

			if not candidates:

				msg = element + ' requires list of candidates found at key \'candiates\' since phase2 is intended.'
				raise ValueError(msg)

			if not isinstance(candidates, list):

				msg = 'Expected ' + type( list ) + ' got ' + type( candidates )
				raise TypeError(msg)

			square_images = p2_dataset_config.get('square_images', False)

			prep_p2w_args = Namespace(

				root_dir = tray_dataset,
			    candidates = candidates,
		    	base = 64,
		    	square_images = square_images

			)

			p2_workspace_dir = prep_p2w( prep_p2w_args )

			p2_aug_config = p2_dataset_config.get('aug_config', {})

			with open( join( p2_workspace_dir, 'aug_config.json' ), 'w') as fd:

				dump(p2_aug_config, fd, indent = 3, sort_keys = True)

			p2_model_config = phase2_config.get('model_config', {})

			with open( join( p2_workspace_dir, 'model_config.json' ), 'w') as fd:

				dump(p2_model_config, fd, indent = 3, sort_keys = True)

	return 0

def main():

	argp = ArgumentParser( description = '' )
    
	argp.add_argument('--root_dir', '-i', required = True, help = 'root directory where the exports generated using the Biomet AnnotationTool for multiple trays can be found.')
	argp.add_argument('--config', '-c', required = False, help = 'config of how to go about preping both phase1 and phase2 worksapces for each tray found in root_dir, where each tray folder is the result of the Biomet AnnotationTool export')

	args = argp.parse_args()

	_ = run(args)

if __name__ == '__main__': main()