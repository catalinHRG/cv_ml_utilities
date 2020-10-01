from sys import path as PYTHONPATH

PYTHONPATH.insert(0, '/media/sml/catalinh/misc/utils/exe/tier1')

from paralel_augment_dataset import run as augment
from get_random_eval_selection import run as get_random_split
from balance_dataset_temporal_chunks import run as balance_dataset
from clean import run as clean_dataset

PYTHONPATH.insert(0, '/media/sml/catalinh/misc/utils/AnnotationConverter/AnnotationConverter')

from to_google_obj_detect import run as create_tfr

PYTHONPATH.insert(0, '/media/sml/catalinh/misc/utils/exe/tier2')

from find_optimal_resolution import run as find_optimal_image_shape

from argparse import ArgumentParser, Namespace
from subprocess import Popen, PIPE
from os.path import join, isdir, isfile
from os import mkdir, listdir
from json import load, dump

from glob import glob
from cv2 import imread

PREP_DATASET_AUTOGEN_SIGNATURE = 'prep_dataset_autogen_' # allows for easy cleanup using regex

def _rotate_up_to_360(root_dir, step):

	global PREP_DATASET_AUTOGEN_SIGNATURE

	augs_dir = join(root_dir, 'augs')
	if not isdir(augs_dir): mkdir(augs_dir)

	augment_args = Namespace(

		root_dir = root_dir,
		output_dir = augs_dir,

		num_augs_per_image = 1,
		image_file_prefix = PREP_DATASET_AUTOGEN_SIGNATURE + 'rotated_up_to_360_by_' + str(step),
		augmentation_pipe_root_dir = '/media/sml/catalinh/misc/utils/exe/tier1/samples',
		grayscale = False,
		negative_filter = False,
		rotate_n = step,
		flip_lr = False,
		flip_ud = False

	)

	iters = int( 360 / step ) - 1

	_step = step

	subset_output_dirs = []

	for i in range(iters):
		
		step_multiplier = i + 1

		_step = step * step_multiplier

		augment_args.rotate_n = _step

		subset_output_dir = join(augs_dir, str(_step))

		if not isdir(subset_output_dir): mkdir(subset_output_dir)

		augment_args.output_dir = subset_output_dir

		subset_output_dirs.append(subset_output_dir)

		augment_args.image_file_prefix = PREP_DATASET_AUTOGEN_SIGNATURE + 'rotated_up_to_360_by_' + str( _step )

		print('Rotating base images by', _step, 'degrees ...')

		augment(augment_args)

	annot_dir_destination = join(root_dir, 'ForTrainingAnnotations')

	# Note: mv command cannot merge, workaround is to manually relocate sub dir content to destination and cleaning

	# initiating both and then waiting for them to finish, might be better off like this in terms of IO load

	for subset_output_dir in subset_output_dirs:
		
		image_file_relocate_proc = Popen( 
		
			'mv ' + join(subset_output_dir, '*.jpg') + ' ' + root_dir, 
			bufsize = 2048,
			stdin = PIPE,
			shell = True # for regex arg to be expanded
		
		)
		
		annot_file_relocate_proc = Popen( 

			'mv ' +  join( subset_output_dir, 'ForTrainingAnnotations', '*.json' ) + ' ' + annot_dir_destination, 
			bufsize = 2048, 
			stdin = PIPE,
			shell = True # for regex arg to be expanded

		)

		image_file_relocate_proc.wait()
		annot_file_relocate_proc.wait()

		clean_proc = Popen( 

			['rm', '-r', subset_output_dir], 
			bufsize = 2048, 
			stdin = PIPE

		)

		clean_proc.wait()

def _rotate_180(root_dir):

	global PREP_DATASET_AUTOGEN_SIGNATURE

	augment_args = Namespace(

		root_dir = root_dir,
		output_dir = root_dir,

		num_augs_per_image = 1,
		image_file_prefix = PREP_DATASET_AUTOGEN_SIGNATURE + 'rotated_180',
		augmentation_pipe_root_dir = '/media/sml/catalinh/misc/utils/exe/tier1/samples',
		grayscale = False,
		negative_filter = False,
		rotate_n = 180,
		flip_lr = False,
		flip_ud = False

	)

	augment(augment_args)

def _negative_filter(root_dir):

	global PREP_DATASET_AUTOGEN_SIGNATURE

	augment_args = Namespace(

		root_dir = root_dir,
		output_dir = root_dir,

		num_augs_per_image = 1,
		image_file_prefix = PREP_DATASET_AUTOGEN_SIGNATURE + 'negative',
		augmentation_pipe_root_dir = '/media/sml/catalinh/misc/utils/exe/tier1/samples',
		grayscale = True,
		negative_filter = True,
		rotate_n = 0,
		flip_lr = False,
		flip_ud = False

	)

	augment(augment_args)

def _baseline_augs(root_dir, quantity):

	global PREP_DATASET_AUTOGEN_SIGNATURE

	augment_args = Namespace(

		root_dir = root_dir,
		output_dir = root_dir,

		num_augs_per_image = quantity,
		image_file_prefix = PREP_DATASET_AUTOGEN_SIGNATURE + 'rotated',
		augmentation_pipe_root_dir = '/media/sml/catalinh/misc/utils/exe/tier1/samples',
		grayscale = False,
		negative_filter = False,
		rotate_n = 0,
		flip_lr = False,
		flip_ud = False

	)

	augment(augment_args)

def _flips(root_dir, ud, lr):

	global PREP_DATASET_AUTOGEN_SIGNATURE

	output_dir = join(root_dir, 'augs')
	
	if lr:

		augment_args = Namespace(

			root_dir = root_dir,
			output_dir = output_dir,

			num_augs_per_image = 1,
			image_file_prefix = PREP_DATASET_AUTOGEN_SIGNATURE + 'flip_left_right',
			augmentation_pipe_root_dir = '/media/sml/catalinh/misc/utils/exe/tier1/samples',
			grayscale = False,
			negative_filter = False,
			rotate_n = 0,
			flip_lr = True,
			flip_ud = False

		)

		augment(augment_args)

	if ud:
		
		augment_args = Namespace(

			root_dir = root_dir,
			output_dir = output_dir,

			num_augs_per_image = 1,
			image_file_prefix = PREP_DATASET_AUTOGEN_SIGNATURE + 'flip_up_down',
			augmentation_pipe_root_dir = '/media/sml/catalinh/misc/utils/exe/tier1/samples',
			grayscale = False,
			negative_filter = False,
			rotate_n = 0,
			flip_lr = False,
			flip_ud = True

		)

		augment(augment_args)

	annot_dir_destination = join(root_dir, 'ForTrainingAnnotations')

	image_file_relocate_proc = Popen( 
		
		'mv ' + join(output_dir, '*.jpg') + ' ' + root_dir, 
		bufsize = 2048,
		stdin = PIPE,
		shell = True # for regex arg to be expanded
	
	)
	
	annot_file_relocate_proc = Popen( 

		'mv ' +  join( output_dir, 'ForTrainingAnnotations', '*.json' ) + ' ' + annot_dir_destination, 
		bufsize = 2048, 
		stdin = PIPE,
		shell = True # for regex arg to be expanded

	)

	image_file_relocate_proc.wait()
	annot_file_relocate_proc.wait()

	clean_proc = Popen( 

		['rm', '-r', output_dir], 
		bufsize = 2048, 
		stdin = PIPE

	)

	clean_proc.wait()


def _augment_dataset(root_dir, aug_config):

	''' aug_config dict
		
		{

			"rotate_up_to_360_by": 45,
			"rotate_180": false,
			"negative_filter": false,
			"flip_up_down": true,
    		"flip_left_right": true
			"baseline_augs_per_image": 5

		}

		TODO: add support for flips

	'''

	flip_ud = aug_config.get("flip_up_down", False)
	flip_lr = aug_config.get("flip_left_right", False)

	if flip_ud or flip_lr:

		_flips(root_dir, flip_ud, flip_lr )

	step = aug_config.get("rotate_up_to_360_by", None)

	if step is not None:

		if not isinstance(step, int):

			msg = "\'rotate_up_to_360_by\' should be type int"
			raise TypeError(msg)

		if 0 < step <= 180:

			_rotate_up_to_360(root_dir, step)

	rotate_180 = aug_config.get("rotate_180", None)

	if rotate_180 is not None: 

		if rotate_180:

			_rotate_180(root_dir)

	negative_filter = aug_config.get("negative_filter", None)

	if negative_filter is not None: 

		if negative_filter:

			_negative_filter(root_dir)

	baseline_augs_per_image = aug_config.get("baseline_augs_per_image", None)

	if baseline_augs_per_image is not None:

		if not isinstance(baseline_augs_per_image, int):

			msg = '\'baseline_augs_per_image\' should be type integer'
			raise TypeError(msg)

		if baseline_augs_per_image > 0:

			_baseline_augs(root_dir, baseline_augs_per_image)

def _clean_augs(dataset):

	global PREP_DATASET_AUTOGEN_SIGNATURE

	remnants = []

	candidate_train_set_dir = join( dataset, 'train')
	candidate_eval_set_dir = join( dataset, 'eval')

	_dirs = [candidate_train_set_dir, candidate_eval_set_dir]

	for _dir in _dirs:

		remnants += glob(join(_dir, '*' + PREP_DATASET_AUTOGEN_SIGNATURE + '*' ))
		remnants += glob(join(_dir, 'ForTrainingAnnotations', '*' + PREP_DATASET_AUTOGEN_SIGNATURE + '*' ))

	for remnant in remnants:

		if isfile(remnant):

			clean_proc = Popen( 
			
				'rm ' + remnant, 
				bufsize = 2048,
				stdin = PIPE
			
			)

			clean_proc.wait()

	# maybe return success / failed code


def run(args):

	if not isdir(args.root_dir):

		msg = args.root_dir + ' does not exist or is not a directory.'
		raise IOError(msg)

	aug_config = dict()
	
	if args.aug_config is not None:

		if not isfile(args.aug_config):

			msg = args.aug_config + ' does not exist or is not a file.'
			raise IOError(msg)

		with open(args.aug_config, 'r') as fd:

			aug_config = load(fd)

	candidates = listdir(args.root_dir)

	for candidate in candidates:

		if candidate == 'workspace': continue # skip workspace folder

		candidate_dataset = join(args.root_dir, candidate)

		if not isdir(candidate_dataset): continue # account for any miscellaneous workspace-specific files

		print('Preping', candidate, 'subset ...')

		candidate_dataset_status_file = join(candidate_dataset, 'status.json')

		candidate_dataset_status = {

			"balance": False,
			"split": False,
			"augment": False, # None is used when augments were not intended for candidate_dataset, whilst False is for unfinished attempt at augmenting and True for successfull attempt
			"tfrs": False

		}

		clean_dataset_args = Namespace(

			root_dir = join(candidate_dataset, 'info'),
			originating_image_dir = candidate_dataset

		)
		
		_ = clean_dataset(clean_dataset_args)

		
		if isfile(candidate_dataset_status_file):

			with open(candidate_dataset_status_file, 'r') as fd:

				candidate_dataset_status = load(fd)

		if not candidate_dataset_status['balance']:

			balance_dataset_args = Namespace(

				root_dir = candidate_dataset

			)

			_ = balance_dataset(balance_dataset_args)

			candidate_dataset_status["balance"] = True

			with open(candidate_dataset_status_file, 'w') as fd:

				dump(candidate_dataset_status, fd)

		if not candidate_dataset_status['split']:

			random_split_args = Namespace(

				root_dir = candidate_dataset,

				split_percent = 0.7,
				annot_file_suffix = '_forTraining'

			)
			_ = get_random_split(random_split_args)

			candidate_dataset_status['split'] = True

			with open(candidate_dataset_status_file, 'w') as fd:

				dump(candidate_dataset_status, fd)

		finished_augmenting = candidate_dataset_status['augment']

		candidate_train_set_dir = join( candidate_dataset, 'train')
		candidate_eval_set_dir = join( candidate_dataset, 'eval')

		if not finished_augmenting:

			_aug_config = None

			candidate_aug_config = aug_config.get(candidate, None)

			if candidate_aug_config:

				_aug_config = candidate_aug_config

			else:

				common_aug_config = aug_config.get('common', None)

				if common_aug_config:

					_aug_config = common_aug_config

			if _aug_config is not None:

				clean = _aug_config.get('clean_existing_augs', False)

				if clean: _clean_augs(candidate_dataset)	
				
				print('Augmenting train set ...')
				_augment_dataset( candidate_train_set_dir, _aug_config )

				print('Augmenting eval set ...')
				_augment_dataset( candidate_eval_set_dir, _aug_config )

				candidate_dataset_status['augment'] = True

			with open(candidate_dataset_status_file, 'w') as fd:

				dump(candidate_dataset_status, fd)

		######### 

		# TODO: have lists of values for the 3 params bellow, this will ultimatly yield in a mesh-fashion more relevant configs to try and therefore increase the likelyhood of finding an optimal solution in one sweep if enough time is available
		
		info_dir = join(candidate_dataset, 'info')
		stats_file = join(info_dir, 'stats.json')
		
		_base = args.base
		max_obj_size = args.max_obj_size
		min_obj_size = args.min_obj_size

		optimal_image_heights, optimal_image_widths = [], []

		find_optimal_resolution_args = Namespace(

			root_dir = join( candidate_dataset, 'train'),
			base = _base,
			max_object_target_size = max_obj_size,
			min_object_target_size = min_obj_size

		)

		_results = find_optimal_image_shape( find_optimal_resolution_args )

		optimal_image_heights.append( _results['square_boxes'][0] )
		optimal_image_widths.append( _results['square_boxes'][1] )

		if args.include_rectagular:

			optimal_image_heights.append( _results['rectangular_boxes'][0] )
			optimal_image_widths.append( _results['rectangular_boxes'][1] )

		#########

		_train_imgs = glob(join(candidate_train_set_dir, '*.jpg'))
		_random_img = _train_imgs[0]
		_img = imread(_random_img)
		
		base_image_height, base_image_width, _ = _img.shape		

		metadata = {

			"optimal_image_heights": optimal_image_heights,
			"optimal_image_widths": optimal_image_widths,
			"original_image_height": base_image_height,
			"original_image_width": base_image_width,
			"highest_multiple": _base,
			"average_object_size": (max_obj_size + min_obj_size) / 2

		}

		with open(stats_file, 'w') as fd:

			dump(metadata, fd, indent = 3, sort_keys = True)

		if not candidate_dataset_status['tfrs']:

			tfr_args = Namespace(

				input_dir = candidate_dataset,
				labelmap_file = join(candidate_dataset, 'lmap.pbtxt'),

				json_dir = "ForTrainingAnnotations",
				annot_file_suffix = "_forTraining",
				masks_dir = None,
				num_shards = 1,
				blacklist = [],
				class_weights = None

			)

			_ = create_tfr(tfr_args)

			candidate_dataset_status['tfrs'] = True

			with open(candidate_dataset_status_file, 'w') as fd:

				dump(candidate_dataset_status, fd)


	return 0

def main():

	argp = ArgumentParser( description = '' )
    
	argp.add_argument('--root_dir', '-i', required = True, help = 'Root directory with standard workspace layout.')
	argp.add_argument('--aug_config', '-ac', required = False, help = 'json config file illustrating how each datasubset is to be augmented.')
    
	argp.add_argument('--base', '-b', required = False, type = int, default = 64, help = 'Base used to round up to the nearest multiple of, for the optimal resolutions found.')
	argp.add_argument('--min_obj_size', '-min', required = False, type = int, default = 50, help = 'Max value for the biggest side for each box.')
	argp.add_argument('--max_obj_size', '-max', required = False, type = int, default = 45, help = 'Min value for the biggest side for each box.')

	argp.add_argument('--include_rectagular', '-rec', action = 'store_true', help = 'wether to include the optimal resolution that leads to rectangular boxes along side the restrictive square-box-only option used implicitly.')
    
	args = argp.parse_args()

	_ = run(args)

if __name__ == '__main__': main()


'''

	intended workflow:

		prep_p2_workspace.py

			get the standard layout to work with

		manual labor:

			clean dataset
			balance dataset
			add artificial images
			complex aug strategies offline

			...

		prep_phase2_dataset.py

			randomly split each subset
			
			based on a configuration file indicating what aug strategy is inteded for what subset, make use of paralel_augment_dataset.py and bash commands to bundle up each split
			
			get a number of resolutions that will yield objects of certain characteristics, mainly certain object sizes and aspect ratios that are well suited for the underlaying methodology for object localisation and classifcation techniques
			
			build tfrs

			considerations:

				one might end up employing cetain strategies for finding an optimal resolution by first investigating the dataset

					1. impose condition so that objets end up being square and a certain size
					2. impose that largest side found on average throught the dataset is going to fall between a min and max range imposed and just train with as few anchors as possible that would be squares with sizes that match the values that fall within the range

'''