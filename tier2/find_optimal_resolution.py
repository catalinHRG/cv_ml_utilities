from sys import path as PYTHONPAT
from argparse import ArgumentParser, Namespace
from os.path import join, isfile, isdir
from json import load
from glob import glob
from cv2 import imread

PYTHONPAT.insert(0, '/media/sml/catalinh/misc/utils/exe/tier1')

from get_dataset_stats import run as get_stats

def run(args):

	if not isdir(args.root_dir):

		msg = args.root_dir + ' does not exist or is not a directory.'
		raise IOError(msg)

	imgs = glob(join(args.root_dir, '*.jpg'))

	img = imread(imgs[0])

	image_h, image_w, _ = img.shape # asumes images have the same shape

	results = dict()

	stats_dir = join(args.root_dir, 'info')
	box_stats_file = join(stats_dir, 'box_stats.json')

	if not isdir(stats_dir) or not isfile(box_stats_file):

		get_stats_args = Namespace(

			dataset_input_dir = args.root_dir,
			view_annots = False,
			scale_bins = 5,
			aspect_ratios_bins = 5,
			balance = False,
			multiscale_grid_anchor_generator = False,
			log = False,
			num_box_clusters = 1,
			annot_file_suffix = '_forTraining',
			include_masks = False

		)

		_ = get_stats(get_stats_args)

	with open(box_stats_file) as fd:

		box_stats = load(fd)

	max_target_size = args.max_object_target_size # px
	min_target_size = args.min_object_target_size # px
	avg_target_size = ( max_target_size + min_target_size ) / 2 # px

	biggest_box = box_stats['biggest_box_side']

	height_scale = avg_target_size / biggest_box
	width_scale = avg_target_size / biggest_box

	optimal_image_h = image_h * height_scale
	optimal_image_w = image_w * width_scale

	rounded_image_h = args.base * round(optimal_image_h / args.base)
	rounded_image_w = args.base * round(optimal_image_w / args.base)

	results['rectangular_boxes'] = (rounded_image_h, rounded_image_w)

	height, width, counter = 0, 0, 0

	for class_name, class_box_stats in box_stats['per_class_box_stats'].items():

		height += class_box_stats['height_mean']
		width += class_box_stats['width_mean']

		counter += 1

	height /= counter
	width /= counter

	height_scale = avg_target_size / height
	width_scale = avg_target_size / width

	optimal_image_h = image_h * height_scale
	optimal_image_w = image_w * width_scale

	rounded_image_h = args.base * round(optimal_image_h / args.base)
	rounded_image_w = args.base * round(optimal_image_w / args.base)

	results['square_boxes'] = (max(rounded_image_h, 64), max(rounded_image_w, 64))

	return results

def main():

	parser = ArgumentParser(description = 'Finds the optimal image width and height so that objects end up having the biggest side within a range of values while maintaining the original aspect ratio or even imposing that they end up being square')
	
	parser.add_argument('--root_dir', '-i', required = True, help = 'Root directory where AnnotationTool format dataset is found')
	parser.add_argument('--base', '-b', required = False, type = int, default = 64, help = 'Base used to round up to the nearest multiple of for the final resolution')
	parser.add_argument('--max_object_target_size', '-maxts', required = False, type = int, default = 50, help = 'Max value for the biggest side for each box')
	parser.add_argument('--min_object_target_size', '-mints', required = False, type = int, default = 45, help = 'Min value for the biggest side for each box')

	args = parser.parse_args()

	results = run(args)

	print('For square boxes the resolution is', results['square_boxes'], 'height x width format.')
	print('For rectangular boxes the resolution is', results['rectangular_boxes'], 'height x width format.')

if __name__ == '__main__': main()


'''
	square of target size
		
		anchors for frcnn are to be squares with the sides equal to the average of biggest sides found trought the dataset, if std dev is too big try to cover the range of values with as little anchors as possible

	keep ar but scale image so that biggest side overall is to fall between a range of values

		anchors bis

'''