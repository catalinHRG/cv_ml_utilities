from argparse import ArgumentParser, Namespace

from glob import glob
from os.path import join, basename, isdir, splitext

from subprocess import Popen, PIPE

def run(args):

	if not isdir(args.root_dir):

		msg = args.root_dir + ' does not exist or is not a directory.'
		raise IOError(msg)

	'''

		the date associated with each chunk is encoded in the name of each image associated with a specific chunk

			2CL01011461-00001 2019-05-27 1427.20.jpg
			2CL01011461-00013 2020-07-24 1400.57.jpg


		fetch all images
		fetch all annotations

		split them into chunks using the date

		fill the missing amount for the smaller chunks relative to the biggest via uniform replication (shell cp command for each instance in the small chunks)


	'''

	annotation_directory_name = 'ForTrainingAnnotations'

	annotation_directory = join(args.root_dir, annotation_directory_name)

	if not isdir(annotation_directory):

		msg = 'It appears that there is not ' + annotation_directory_name + ' within the root directory. It was supposed to contain the annotation images'
		raise IOError(msg) 

	all_images = glob(join(args.root_dir, '*.jpg'))

	chunks_by_date = dict()
	chunks_count = dict()

	for image_file in all_images:

		image_name = basename(image_file)

		components = image_name.split(" ")

		date = components[ 1 ]

		_chunk = chunks_by_date.get(date, None) # if the key exists, then it should have atleast an empty list, which a reference for was returned, so that we can append to

		if _chunk is None: 

			chunks_by_date[ date ] = []
			chunks_count[ date ] = 0 

		else: 

			_chunk.append( image_file ) 
			chunks_count[ date ] += 1

	biggest_chunk_id, biggest_chunk_count = '', 0

	for _id, count in chunks_count.items():

		if count > biggest_chunk_count: 

			biggest_chunk_id = _id
			biggest_chunk_count = count

	for _id, chunk in chunks_by_date.items():

		if _id == biggest_chunk_id:	continue

		amount_to_balance = biggest_chunk_count - chunks_count[ _id ]

		no_need_for_balance = ( biggest_chunk_count / float(chunks_count[ _id ]) ) <= 2 # will balance only if the biggest chunk is at least twice as much as the chunk that needs balancing

		if no_need_for_balance:

			print('Skipping', _id, 'no need for balance')
			continue

		print('Balancing', _id, '...')

		amount_to_balance_for_each_element = int( amount_to_balance / chunks_count[ _id ] ) # will be at least 1 

		for image_file in chunk:

			image_file_name = basename( image_file )

			name_, _ext = splitext(image_file_name)

			annotation_file_name = name_ + '_forTraining.json' 
			
			annotation_file = join( annotation_directory, annotation_file_name )

			for i in range( amount_to_balance_for_each_element ):

				image_file_copy_name_ = name_ + '_balance_nr_' + str(i)
				
				image_copy_name = image_file_copy_name_ + _ext
				image_file_copy = join(args.root_dir, image_copy_name)
				
				annotation_file_copy_name = image_file_copy_name_ + '_forTraining.json'
				annotation_file_copy = join( annotation_directory, annotation_file_copy_name )

				copy_image_proc = Popen( ['cp', image_file, image_file_copy], bufsize = 2048, stdin = PIPE )
				copy_annot_proc = Popen( ['cp', annotation_file, annotation_file_copy], bufsize = 2048, stdin = PIPE )

				copy_image_proc.wait()
				copy_annot_proc.wait()



	return 0

def main():

	argp = ArgumentParser( description = 'Balance a dataset comprised of the different chunks sent by the client at different times, each of each, supposedly, containing relevant variety introduced in the final dataset, via small chunks relative to the initial imbalanced/incomplete dataset.' )
    
	argp.add_argument('--root_dir', '-i', required = True, help = 'Root directory for AnnotationTool format dataset')

	args = argp.parse_args()

	_ = run(args)

if __name__ == '__main__': main()
