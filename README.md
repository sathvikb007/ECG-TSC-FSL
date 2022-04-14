# Similarity Learning based Few Shot Learning for ECG Time Series Classification

The repository contains the official code implementation of the following paper:

**Title**: "Similarity Learning based Few Shot Learning for ECG Time Series Classification"

**Authors**: Priyanka Gupta, Sathvik Bhaskarpandit, Manik Gupta

**ArXiv Link**: https://arxiv.org/abs/2202.00612

**Code Implementation**: https://github.com/sathvikb007/ECG-TSC-FSL

**Abstract**: 
Using deep learning models to classify time series data generated from the Internet of Things (IoT) devices requires a large amount of labeled data. However, due to constrained resources available in IoT devices, it is often difficult to accommodate training using large data sets. This paper proposes and demonstrates a Similarity Learning-based Few Shot Learning for ECG arrhythmia classification using Siamese Convolutional Neural Networks. Few shot learning resolves the data scarcity issue by identifying novel classes from very few labeled examples. Few Shot Learning relies first on pretraining the model on a related relatively large database, and then the learning is used for further adaptation towards few examples available per class. Our experiments evaluate the performance accuracy with respect to K (number of instances per class) for ECG time series data classification. The accuracy with 5- shot learning is 92.25% which marginally improves with further increase in K. We also compare the performance of our method against other well-established similarity learning techniques such as Dynamic Time Warping (DTW), Euclidean Distance (ED), and a deep learning model - Long Short Term Memory Fully Convolutional Network (LSTM-FCN) with the same amount of data and conclude that our method outperforms them for a limited dataset size. For K=5, the accuracies obtained are 57%, 54%, 33%, and 92% approximately for ED, DTW, LSTM-FCN, and SCNN, respectively.

## Instructions 
Here are the instructions to use the code base:

### Dependencies
Clone the repository, navigate to the src folder and run the following code to install the necessary dependencies :

```
pip install requirements.txt
```

### Data
Download the datasets to the 'data' folder. The data can be found here:
https://drive.google.com/drive/folders/1Up-D4JMj79co-Td5eeQzePgV9zeUzc45?usp=sharing

### Running the Models
To train and test models on the few-shot learning tasks, use the command line below for detailed guide:

```
python main.py --help
```

## Citation:
If you use any content of this repo for your work, please cite the following bib entry:
```
@inproceedings{gupta2021similarity,
  title={Similarity Learning based Few Shot Learning for ECG Time Series Classification},
  author={Gupta, Priyanka and Bhaskarpandit, Sathvik and Gupta, Manik},
  booktitle={2021 Digital Image Computing: Techniques and Applications (DICTA)},
  pages={1--8},
  organization={IEEE}
}
```


