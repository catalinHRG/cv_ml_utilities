
from argparse import ArgumentParser
from os import mkdir
from os.path import join, isdir, isfile, dirname, basename, splitext
from glob import glob

from subprocess import Popen, PIPE

def get_base(bad_crop_image_name):

	name_, _ext = splitext( bad_crop_image_name )

	_name_ = name_[0: name_.index('_forTraining')]
	
	return _name_ + '.jpg'


def put_quotes(string):

	return '"' + string + '"'

def run(args):

	originating_image_dir = args.originating_image_dir

	if originating_image_dir is not None:

		if not isdir(originating_image_dir): 

			msg = 'Originating image dir does not exist ' + originating_image_dir
			raise ValueError(msg)

	else:

		originating_image_dir = dirname(args.root_dir)

	dump_spot = join(originating_image_dir, 'bad')

	if not isdir(dump_spot): mkdir(dump_spot)

	content = glob(join(args.root_dir, '*'))

	orig_big_image_names = set()

	for item in content:

		if not isdir(item): continue

		bad_crops_dir = join(item, 'bad') # intended to completly remove the originating images associated with each bad crop from the dataset

		if not isdir(bad_crops_dir): continue # maybe this class does not have any bad annots

		bad_crops = glob(join(bad_crops_dir, '*'))

		for bad_crop in bad_crops:

			_bad_crop = basename(bad_crop)

			originating_image_name = get_base(_bad_crop)

			orig_big_image_names.add( originating_image_name ) 

			originating_image_path = join( originating_image_dir, originating_image_name)

			if not isfile(originating_image_path): 

				print('There must be something wrong here, either the originating image file name was not compiled properly or this image was relocated already')
				print('Crop name is', _bad_crop, 'found in', dirname(bad_crop))
				print('Originating image file path is', originating_image_path)
				print('#######################################################################')

				continue
			
			p = Popen( ['mv', originating_image_path, dump_spot], bufsize = 2048, stdin = PIPE)
			p.wait()

		explicit_negative_crop_dir = join(item, 'exneg') # intended to retain the images but remove the annotations associated with it from the dataset

		if not isdir(explicit_negative_crop_dir): continue

		explicit_negative_crops = glob(join(explicit_negative_crop_dir, '*'))

		for explicit_negative_crop in explicit_negative_crops:

			_explicit_crop = basename(explicit_negative_crop)

			originating_image_name = get_base(_explicit_crop)

			originating_image_annotation_file_name = splitext(originating_image_name)[0] + '_forTraining.json'

			originating_image_annotation_file_path = join( originating_image_dir, 'ForTrainingAnnotations', originating_image_annotation_file_name)

			if isfile(originating_image_annotation_file_path):

				annotation_dump_spot = join(dump_spot, 'ForTrainingAnnotations')
				if not isdir(annotation_dump_spot): mkdir(annotation_dump_spot)

				p = Popen( ['mv', originating_image_annotation_file_path, annotation_dump_spot], bufsize = 2048, stdin = PIPE )
				p.wait()


	with open( join(dump_spot, 'bad_image_names.txt'), 'w' ) as fd:

		for name in orig_big_image_names:

			fd.write(name + '\n')


def main():

	parser = ArgumentParser(description = 'Clean dataset of badly annotated images denotaed by user in a standard way by having a folder named bad \'bad\' within each class folder, where all the crops associated with images that are to be cleaned will be found')
	
	parser.add_argument('--root_dir', '-i', required = True, help='Input directory consiting of the crops gotten from image dataset using get_dataset_stats.py')
	parser.add_argument('--originating_image_dir', '-o', required = False, help = 'Directory where to find the images that are to be cleaned, based on bad_crops_dir')

	args = parser.parse_args()

	run(args)

if __name__ == '__main__': main()
