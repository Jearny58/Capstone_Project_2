# Capstone Project #2 for Springboard 

## Pathology Prediction

With the [CheXpert data set](https://stanfordmlgroup.github.io/competitions/chexpert/) from Stanford University, we apply deep learning in an attempt to analyze and detect cardiomegaly from nearly 224,000+ medical images accurately. The purpose of CheXpert and this project is to research further the viability of deep learning in helping to screen and diagnose life-threatening diseases. While this particular model only focuses on one pathology, its performance is near that of Stanford's for cardiomegaly. 

Additionally, this project utilizes [fast.ai's deep learning library](https://www.fast.ai/), which is a research institute co-founded by [Jeremy Howard](https://www.fast.ai/about/#jeremy) and [Rachel Thomas](https://www.fast.ai/about/#rachel) with the mission "to make deep learning as accessible as possible." It is built on top of [PyTorch](https://pytorch.org/), and further information regarding the library can be found [here](https://docs.fast.ai/).

## Table of Contents

- `capstone` - Folder that contains python scripts utilized within various stages of the project to help with reproducibility and produce 'cleaner' notebooks, below are some of the more critical scripts and a quick description of their primary focus:
    - [`sample.py`](https://github.com/Jearny58/Springboard-DS-Portfolio/blob/master/capstone_2/capstone/sample.py): preps data from CSVs, sets seed for reproducibility, and creates sample data sets to more quickly iterate training
    - [`replicate.py`](https://github.com/Jearny58/Springboard-DS-Portfolio/blob/master/capstone_2/capstone/replicate.py): further helps prepare data by, splits data into a validation set, and label images based on the feature column (i.e., cardiomegaly)
    - [`deeplearning.py`](https://github.com/Jearny58/Springboard-DS-Portfolio/blob/master/capstone_2/capstone/deeplearning.py): applies any transformations to the images (if necessary), determines batch size based on GPU memory and creates ImageDataBunch that is ready to be fed into the deep learning model
- `data` - Folder containing medical images during the project; terms of services prohibited the distribution of the images so it __was not__ uploaded to GitHub however it can be accessed [here](https://stanfordmlgroup.github.io/competitions/chexpert/) for anybody interested
    - [`connect-storage-bucket-trial.ipynb`](https://github.com/Jearny58/Springboard-DS-Portfolio/blob/master/capstone_2/data/connect-storage-bucket-trial.ipynb): folder also contains the following Jupyter notebook that was used to move data from GCP storage bucket to compute instance
- `exploration` - Folder containing material from exploratory data analysis stage of the project
    - [`chexpert-eda.ipynb`](https://github.com/Jearny58/Springboard-DS-Portfolio/blob/master/capstone_2/exploration/chexpert-eda.ipynb): Jupyter Notebook that explores demographic aspects and pathology distribution within the training and validation data sets
- `playground_nbs` - Folder that contains experimental notebooks from various stages of the model creation process
- `reports` - Folder that contains [slide deck](https://github.com/Jearny58/Springboard-DS-Portfolio/blob/master/capstone_2/reports/capstone_2_presentation_draft.pdf) detailing findings, and [milestone report](https://github.com/Jearny58/Springboard-DS-Portfolio/blob/master/capstone_2/reports/milestone_report_1.pdf)/[final report](https://github.com/Jearny58/Springboard-DS-Portfolio/blob/master/capstone_2/reports/final_report.pdf) that provides comprehensive analysis of entire workflow from data acquisition to training the deep learning model.
- [`trial30.ipynb`](https://github.com/Jearny58/Springboard-DS-Portfolio/blob/master/capstone_2/trial30.ipynb) - Jupyter notebook that contains the entire model development process for the best performing model (AUC = ~ 0.82)

## Approach & Results

I used fastai to fine-tune a DenseNet121 architecture on CheXpert, a data set comprising ~224k medical images from Stanford, to detect cardiomegaly (i.e., enlarged heart). This dataset was also a competition, where the goal was to create a model that could perform as well as radiologists in detecting different pathologies in chest X-rays. Additionally, the metric that was being used to judge the various models submitted was the AUC score. 

This project was my first foray into deep learning, and as such, I determined that it would be more advantageous to focus on classifying one pathology (i.e., cardiomegaly) in particular, as opposed to the 14 labels that the competition focused on. As for project "goal," it was to see if I could build a classifier that could get as close to replicating the results that Stanford's ML team achieved in detecting cardiomegaly, which stood at 0.854. 

In the end, my model very nearly achieved this mark, with an AUC of approximately 0.83. Not bad for my first venture into deep learning!
