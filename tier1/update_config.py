from sys import path as PYTHONPATH

PYTHONPATH.insert(0, "/media/sml/catalinh/misc/godapi/research/object_detection")

from utils.config_util import get_configs_from_pipeline_file as create_config
from utils.config_util import create_pipeline_proto_from_configs as create_pipeline
from utils.config_util import save_pipeline_config as save_pipeline

from argparse import ArgumentParser

from os import listdir
from os.path import join, isdir

from subprocess import Popen, PIPE

def run(args):
	
	'''

	pipeline = create_pipeline( retinanet_template_config )
	retinanet_template_config = create_config( join(args.base_configuration_directory, 'retinanet_pipeline.config') )
	save_pipeline( pipeline, base_folder, 'train.config' )

	'''

	if not isdir(args.root_dir):

		msg = args.root_dir + ' does not exist or is not a directory.'
		raise IOError(msg)


	batch_size, encoder_size, image_height, image_width = None, None, None, None

	if args.batch_size is not None:

		if not isinstance(args.batch_size, int):

			msg = '--batch_size should be int and > 0'
			raise TypeError(msg)

		batch_size = args.batch_size

	if args.encoder_size is not None:

		if not isinstance(args.encoder_size, float):

			msg = '--encoder_size should be float > 0'
			raise TypeError(msg)

		encoder_size = args.encoder_size

	if args.image_height is not None:

		if not isinstance(args.image_height, int):

			msg = '--image_height should be int > 0'
			raise TypError(msg)

		image_height = args.image_height

	if args.image_width is not None:

		if not isinstance(args.image_width, int):

			msg = '--image_width should be int > 0'
			raise TypeError(msg)

		image_width = args.image_width

	workbenches = listdir(args.root_dir)

	for workbench in workbenches:

		base_dir = join(args.root_dir, workbench, 'base')

		train_config = create_config( join(base_dir, 'train.config') )

		if batch_size is not None:

			train_config['train_config'].batch_size = batch_size

		if encoder_size is not None:

			train_config['model'].ssd.feature_extractor.depth_multiplier = encoder_size
			train_config['train_config'].fine_tune_checkpoint = '/media/sml/catalinh/model_zoo/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/model.ckpt' 

		if image_height is not None:

			train_config['model'].ssd.image_resizer.fixed_shape_resizer.height = image_height

		if image_width is not None:

			train_config['model'].ssd.image_resizer.fixed_shape_resizer.width = image_width

		pipeline = create_pipeline( train_config )
		save_pipeline( pipeline, base_dir, 'train.config' )
		save_pipeline( pipeline, base_dir, 'eval.config' )

		proc = Popen(['rm', '-r', join(base_dir, 'dump')], bufsize = 2048, stdin = PIPE)
		proc.wait()

		proc = Popen(['rm', join(base_dir, 'done_evaluating')], bufsize = 2048, stdin = PIPE)
		proc.wait()


	return 0

def main():

	parser = ArgumentParser( description = '' )
	
	parser.add_argument( '--root_dir', '-i', required = True, help = 'Root directory path for subset workbenches' )
	parser.add_argument( '--batch_size', '-bs', required = False, type = int, default = None, help = '@@')
	parser.add_argument( '--encoder_size', '-es', required = False, type = float, default = None, help = '@@')
	parser.add_argument( '--image_height', '-ih', required = False, type = int, default = None, help = '@@')
	parser.add_argument( '--image_width', '-iw', required = False, type = int, default = None, help = '@@')

	args = parser.parse_args()

	_ = run(args)

if __name__ == '__main__': main()