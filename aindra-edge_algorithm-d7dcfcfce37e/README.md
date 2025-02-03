This repo contains code for deployment of the cervastra ai algorithm in the edge device

## Strucutre of repo
### Src
Contains source files. Will only contain the necessary files to run core algo

### Tests
Contains code and data for testing. Please note that files/data inside this folder is only for testing. You should not use then in production code. 

## Usage
A sample for using the code is given in tests/sample_usage.py.
Tiles directory to be processed must contain acq_finished.txt file

## API
The stable API of this project is

## Setup.py
Run this file to create wheel(.whl) file using this command in current directory

python2 setup.py sdist bdist_wheel

## Testing .whl package
1. Create a virtual environment 
2. Run pip install edge_algorithm-0.0.1-py2-none-any.whl from the directry of whl file
3. Run the command

python -m edge_algorithm.tests.sample_usage model.pb tileFiles/tiles_files/

where model.pb is model file
	  tileFiles/tiles_files : directory containing images of samples

'''python 

src.wsi_analyzer. Slide

src.wsi_analyzer.queue_processor(queue_in, queue_out, process_lock, model_path, logger,
                    per_process_gpu_memory_fraction):
       
'''

1. queue_in : Input queue. Queued items should be of type src.wsi_analyser.Slide with the tiles_dir pointing to a directory with images

2. queue_out : Output queue

3. process_lock : Lock to pause processing of slide. To pause the processing acquire the lock. To unpause release the lock

4. model_path: Path of model

5. logger : a logging object. This is used for logging

6. per_process_gpu_memory_fraction: Fraction og GPU memory to be used. Defaults to 1.0


Please use nothing other than this API from the project

