# COVID Cough Predictor
>  Hackacthon bitsxlamarato submission: Telegram bot that classifies between cough and non-cough audio files and then COVID-cough and non-COVID-cough

The goal of this project is to develop a tool that can classify a cough audio message between COVID and non-COVID cough. This has been achieved by training a first predictor on cough and non-cough data and a second predictor on COVID and non-COVID cough data.

The main challenges  we ahve had to tackle are the difference in the length of the audio files, as well as the heavy unbalance between classes.

<p align="center">
  <img src='README Images/screenshot.png'/ width = 250>
</p>

## Preprocessing

To work with different length audio signals we obtain the spectrogram and reshape it to a standard size. This preserves the features of the sound files while allowing us to feed the networks with images of the same size:

<p align="center">
  <img src='README Images/spectrogram.JPG'/ width = 500>
</p>

To solve the unbalanced class problem we have performed data augmentation by changing the sampling frequency of the audio signals. We have also tunned the class weights in the loss function.

### Learning Models

The final model uses trained weights from the VGG16 network trained over ImageNet. The feature extraction layers have been frozen, and we have fine-tuned an MLP on top of them. This approach is much faster than learning a model from scratch, and aids in avoiding the over-fitting problem.

<p align="center">
  <img src='README Images/vgg_arch.png'/ width = 500>
</p>

## Installation

In order to correctly use the programs provided, it is necessary to have installed the following libraries:

* `numpy`
* `torch`
* `torchvision`
* `torchaudio`
* `tqdm`
* `matplotlib`
* `soundfile`
* `pandas`
* `shutil`
* `sklearn`


These can be installed using the pip command in the command line:

`pip install name_of_package`

## Architecture

The repository folder contains several files, one being this README. The rest of them are:

* The [`preprocess_pipeline.py`](./preprodcessing/preprocess_pipeline.py) inside the *preprocessing* folder, along with other auxiliary notebooks.
* [`CNN+LSTM_covid.ipynb`](./model_training/CNN+LSTM_covid.ipynb), [`CNN_scratch.ipynb`](./model_training/CNN_scratch.ipynb), [`CNN_transfer.ipynb`](./model_training/CNN_transfer.ipynb), [`CNN_transfer_covid.ipynb`](./model_training/CNN_transfer_covid.ipynb) inside the model *model_training* folder, notebooks that contain the training and validation of the different models.
* [`cough_detection_model_from_scratch.pt`](./models/cough_detection_model_from_scratch.pt), [`fine_tuned_transfer.pt`](./models/fine_tuned_transfer.pt), [`fine_tuned_transfer_augmented.pt`](./models/fine_tuned_transfer_augmented.pt), [`fine_tuned_transferv2.pt`](./models/fine_tuned_transferv2.pt) inside the *models* folder.
* [`Presentation`](./APOLO\ STYLE\ BBY.pptx) with slides showing an overview of the project.


*The datasets required have not been uploaded due to their size. The repository is for educational purpose only.*
*Note: The rest of the files in the master branch are auxiliary or license related*

## Team

This project was developed by:
| [![CarlOwOs](https://avatars3.githubusercontent.com/u/49389491?s=60&u=b239b67c3f064bf2dae05e08ae9965b7c7e34c36&v=4)](https://github.com/CarlOwOs) | [![megaelius](https://avatars2.githubusercontent.com/u/43412999?v=4&s=60)](https://github.com/megaelius) | [![alexmartiguiu](https://avatars2.githubusercontent.com/u/49391060?v=4&s=60)](https://github.com/alexmartiguiu) | [![turcharnau](https://avatars2.githubusercontent.com/u/70148725?v=4&s=60)](https://github.com/turcharnau) |
| --- | --- | --- | --- |
| [Carlos Hurtado](https://github.com/CarlOwOs) | [Elías Abad](https://github.com/megaelius) | [Alex Martí](https://github.com/alexmartiguiu) | [Arnau Turch](https://github.com/turcharnau) |


Students of Data Science and Engineering at [UPC](https://www.upc.edu/ca).

## License

[MIT License](./LICENSE)
