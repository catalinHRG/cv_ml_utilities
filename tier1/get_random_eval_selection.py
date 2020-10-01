from os import system
from glob import glob
from os.path import splitext, isdir, isfile, basename, dirname, join
from os import mkdir, listdir
from argparse import ArgumentParser
from random import shuffle
from tqdm import tqdm
from time import time
from multiprocessing import cpu_count, Process, Manager

image_file_formats = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', 'PNG', '.bmp', '.BMP', '.bitmap', '.BITMAP']

def relocate(

	image_files, 

	img_dir, 
	annot_dir,

	annot_file_suffix

	):

	### Helpers

	def _put_quotes(string):

		return '"' + string + '"'

	def _get_annotation_file(image_file):

		_image_file = basename(image_file)
		
		annotation_dir = dirname(image_file)
		
		name_, _ext = splitext(_image_file)

		_annotation_file = name_ + annot_file_suffix + '.json'

		annotation_file = join(annotation_dir, 'ForTrainingAnnotations', _annotation_file)

		if not isfile(annotation_file): 

			print('missing anont file')
			annotation_file = None

		return annotation_file

	def _relocate(image_files, image_destination_dir, annot_destination_dir):

		_image_files = tqdm(image_files)

		for image_file in _image_files:

			#_image_files.set_description("Relocating %s" % image_file)
			
			annotation_file = _get_annotation_file(image_file)

			if isfile(image_file):

				command = 'mv ' + _put_quotes(image_file) + " " + _put_quotes(image_destination_dir)
				system(command)

			if annotation_file is not None:

				if isfile(annotation_file):

					command = 'mv ' + _put_quotes(annotation_file) + ' ' + _put_quotes(annot_destination_dir)
					system(command)


	_relocate(image_files, img_dir, annot_dir)

def _assign_workloads(image_files, output_image_dir, output_annot_dir, annot_suffix, max_workers):

	image_workloads = [ [] for dontcare in range(max_workers) ]
	worker_id = 0

	for image_file in image_files:

		image_workloads[worker_id].append(image_file)
		worker_id += 1

		if worker_id == max_workers: worker_id = 0

	workers = []
	
	for idx, image_workload in enumerate(image_workloads):

		workers.append(

			Process(

				target = relocate, 
				args = (

					image_workload,
					output_image_dir,
					output_annot_dir,
					annot_suffix
				)
			)
		)

	return workers
	


def run(args):

	if not isdir(args.root_dir):

		msg = args.root_dir + ', root dir does not exist'
		raise IOError(msg)

	annotation_dir_name = 'ForTrainingAnnotations'
	annotation_dir = join(args.root_dir, annotation_dir_name)
	
	if not isdir(annotation_dir):

		msg = 'Annotation directory does not exist, aborting ...'
		raise IOError(msg)

	### 

	train_dir = join(args.root_dir, 'train')
		
	if not isdir(train_dir): mkdir(train_dir)

	train_annot_dir = join(train_dir, annotation_dir_name)

	if not isdir(train_annot_dir): mkdir(train_annot_dir)

	
	eval_dir = join(args.root_dir, 'eval')
		
	if not isdir(eval_dir): mkdir(eval_dir)

	eval_annot_dir = join(eval_dir, annotation_dir_name)

	if not isdir(eval_annot_dir): mkdir(eval_annot_dir)

	###
	
	train_split_file = join(args.root_dir, 'train.split')
	eval_split_file = join(args.root_dir, 'eval.split')

	train_image_files, eval_image_files = [], []

	if isfile(train_split_file) and isfile(eval_split_file):

		with open(train_split_file, 'r') as fd:

			for element in fd.readlines():

				train_image_files.append(join(args.root_dir, element.rstrip()))
	
		with open(eval_split_file, 'r') as fd:

			for element in fd.readlines():

				eval_image_files.append(join(args.root_dir, element.rstrip()))
		
	else:

		def _write_split_file(image_files, split_file):

			_image_files = []
			
			for image_file in image_files:

				_image_files.append(basename(image_file) + '\n')

			with open(split_file, 'w') as fd:

				fd.writelines(_image_files)

		image_files = []
		
		for image_file_format in image_file_formats: 

			image_files += glob(join(args.root_dir, '*' + image_file_format))

		shuffle(image_files)
		
		image_file_count = len(image_files)

		eval_count = int(image_file_count * args.split_percent)
		
		train_image_files = image_files[ 0 : eval_count ]
		eval_image_files = image_files[ eval_count : ]

		_write_split_file(train_image_files, train_split_file)
		_write_split_file(eval_image_files, eval_split_file)

	# TODO: make relocation optional, after revamping tfr generation tool

	max_workers = cpu_count()

	train_set_workers = _assign_workloads(

		train_image_files,
		train_dir,
		train_annot_dir,
		args.annot_file_suffix,
		max_workers

	)

	eval_set_workers = _assign_workloads(

		eval_image_files,
		eval_dir,
		eval_annot_dir,
		args.annot_file_suffix,
		max_workers

	)
	
	start = time()
	
	for worker in train_set_workers: worker.start()
	for worker in eval_set_workers: worker.start()

	for worker in train_set_workers: worker.join()
	for worker in eval_set_workers: worker.join()

	system('rm -r ' + annotation_dir)

	print('Job done in', time() - start, 'seconds.')

def main():

	argp = ArgumentParser(description = '@@')

	argp.add_argument('--root_dir', '-i', required = True, help = 'absolute path to root dir')
	argp.add_argument('--split_percent', '-p', default = 0.9, type = float)
	argp.add_argument('--annot_file_suffix', '-as', default = '_forTraining', help = '@@')

	args = argp.parse_args()

	run(args)

if __name__ == '__main__': 

	main()
