# Final Project: Information Retrieval and Web Analytics 

In this project we were asked to collect data from Twitter, build a search engine, and answer 3 research questions. The topic that we selected to perform our work is Covid-19.


Getting started
------
Download a copy of this repository. Extract the zip file and rename it to *FinalProject*. If not existent, create a directory in your root Google Drive named *IRWA* and upload the *FinalProject* directory to it.The path to the repository folder should be */content/drive/My Drive/IRWA/FinalProject/*

Also, go to https://drive.google.com/drive/folders/1K2fq_juqM5T0HRaHYtargFjzrHFyniOT?usp=sharing and right click on the data directory. Add a shortcut to this folder inside the *FinalProject* directory, or simply download the files and add them to a data directory inside *FinalProject*. The path to the data should be */content/drive/My Drive/IRWA/FinalProject/data/*


The expected directory structure is the following:

<p align="center">
  <img width="100%" src="https://github.com/claudia-hm/IRMAS_Deep_Learning/blob/master/directory_structure_2.png" >
</p>

Prerequisites
------
Google Colab

Running the tests
------
Run the IRMAS_Deep_Learning.ipynb notebook. Mount drive when prompted.

The code is structured in the following sections:

0. Preliminaries: mount drive, install required packages, imports, data checking and CPU/GPU warm-up.
1. Data loading and pre-processing: retrieve the audio filenames and compute the Log Mel Spectrograms for the whole dataset.
2. Create custom dataset used to gather batches of images.
3. Define the model: class creation and instantiation of the VGG-like model.
4. Hyperparameters setting
5. Actual training
6. Analysing the output incluiding testing

Also, we have included an extra python notebook with code to retrain and test the model with the original 11 instruments (IRMAS_Deep_Learning_bis_11.ipynb). The structure of the code is exactly the same as in IRMAS_Deep_Learning.ipynb.

### Results
In IRMAS_Deep_Learning.ipynb and IRMAS_Deep_Learning_bis_11.ipynb, we have reached an accuracy of approximately 83% and 64%, respectively. As the training and testing spliting is stochastic, the accuracy value will differ from run to run. We recommend restarting the notebook and running it again to achieve this results if they are not reflected in the first runs.

Built With
------
* [PyTorch](https://pytorch.org) - An open source machine learning framework
* [Essentia](https://essentia.upf.edu) - Open-source library and tools for audio and music analysis, description and synthesis

Authors
------
* [Claudia Herron Mulet](https://www.linkedin.com/in/claudiaherronmulet/) (claudia.herron01@estudiant.upf.edu)  
* Júlia Riera Perramón (julia.riera02@estudiant.upf.edu)
* [Sara Estévez Manteiga](www.linkedin.com/in/saraestevezmanteiga/) (sara.estevez02@estudiant.upf.edu)

Mathematical Engineering in Data Science students at UPF.
 

Acknowledgments
------
Xavier Favory, researcher at the Audio Signal Processing Lab at UPF, for the code provided to start this project.

Bosch, J. J., Janer, J., Fuhrmann, F., & Herrera, P. “A Comparison of Sound Segregation Techniques for Predominant Instrument Recognition in Musical Audio Signals”, in Proc. ISMIR (pp. 559-564), 2012


