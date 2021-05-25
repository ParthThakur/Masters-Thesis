MSc Dissertation
Artefact submitted by: Parth Thakur
Student ID: 10520930

Note: This artefact requires an installation of Anaconda Python version 3.6.

Pre-requisites:
	Training the neural networks requires a CUDA compatible GPU. 
	If a GPU is not present, training will be slow.
	
	Install tensorflow using the following command:
		conda install tensorflow-gpu
		
	Alternatively, if no GPU is present:
		conda install tensorflow
		
Requirements:
	numpy
	hdf5
	keras
	scikit-learn
	
	Install all required libraries using:
		pip install requirements.txt
		
1.	To simulate FRB data:
		Run create_simulations.py
			python create_simulations
		Change file name in benchmark_models.py
			
	Alternatively, download data from https://mydbs-my.sharepoint.com/:u:/g/personal/10520930_mydbs_ie/Efv1UtOh40JGvH_akOkM8dcB18aCqwoxvsj0_bWigDLvoQ?e=DC6c5c
	and store in data subdirectory.
	
2.	To benchmark the models:
		run benchmark_models.py
			python benchmark_models.py