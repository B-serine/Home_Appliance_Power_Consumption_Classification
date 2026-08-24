# Home Appliance Power Consumption Classification

## Project Overview

This project classifies electrical power-consumption recordings according to the home appliance that produced each recording. The work is presented as a multiclass time-series classification study for the TSAC course.

The notebook compares feature-based machine-learning methods with time-series methods based on random convolutional kernels. The selected approach is used to train on the complete training data and create a prediction file for the competition test data.

## Notebook

The complete analysis is available in [TSAC_Group_Project.ipynb](TSAC_Group_Project.ipynb).

The notebook contains:

- Problem formulation and motivation
- Environment setup and package installation
- Data loading from a shared Google Drive folder
- Preparation of training and test arrays
- Class-name definitions and stratified cross-validation setup
- Visual inspection of representative appliance signals
- Statistical and structural feature extraction
- Baseline model training and evaluation
- Dynamic Time Warping nearest-neighbor experimentation
- A from-scratch ROCKET implementation
- Optimized ROCKET kernel-count tuning with `aeon`
- Comparison between ROCKET and MiniRocket
- Confusion matrices and accuracy visualizations
- Final model selection and retraining
- Test-set prediction and submission-file generation
- Discussion of results, limitations, and possible improvements

## Data

The notebook expects the following files in the shared project directory on Google Drive:

- `train.csv`, containing the training labels and power-consumption recordings
- `test.csv`, containing test identifiers and recordings

The data is read with `pandas`. The first training column is treated as the target label, while the remaining columns represent the ordered time-series observations. The first test column is retained as the prediction identifier.

The recordings are also reshaped into the channel-and-time format required by `aeon` and `tslearn`.

## Preprocessing and Feature Extraction

For the classical models, every power-consumption recording is transformed into a compact feature vector. The extracted descriptors include:

- Central-tendency statistics
- Dispersion statistics
- Minimum and maximum values
- Percentiles
- Signal range and energy
- Skewness and kurtosis
- Sign changes and differences between adjacent observations
- Peak and trough counts
- Peak-height statistics
- Segment-level means and standard deviations
- Proportions of observations above selected thresholds

This representation reduces the dimensionality of the raw recordings and gives the feature-based classifiers statistical and structural information about each signal.

## Models Compared

### Random Forest

A tuned Random Forest classifier is trained on the extracted features. Grid search is used to select the main tree and split parameters, followed by cross-validation, confusion-matrix analysis, and prediction on the test features.

### Support Vector Machine

An RBF-kernel Support Vector Machine is trained in a pipeline with feature standardization. Cross-validation and confusion-matrix analysis are used to assess its performance before fitting it on all available training features.

### Nearest Neighbors with Dynamic Time Warping

A time-series nearest-neighbor classifier compares recordings with Dynamic Time Warping rather than direct point-by-point distance. The recordings are resampled before this experiment so that the search remains computationally manageable. Candidate neighborhood sizes and warping constraints are evaluated through stratified cross-validation.

The full Dynamic Time Warping search is kept in the notebook as a reference workflow because it is computationally expensive.

### ROCKET

The project first implements ROCKET directly with NumPy. Random convolutional kernels are generated with varying lengths, biases, dilations, and padding choices. Each kernel produces activation-based features consisting of its maximum response and the proportion of positive responses.

The notebook then uses the optimized `aeon` ROCKET implementation. Several kernel configurations are evaluated with a Ridge classifier and stratified cross-validation. The configuration with the strongest validation behavior is retained for comparison with MiniRocket.

### MiniRocket

MiniRocket is evaluated as a more efficient ROCKET variant. Its transformed features are passed to a Ridge classifier, and its cross-validation behavior is compared with the selected ROCKET configuration.

## Evaluation

The notebook uses stratified cross-validation for the internal comparison. It also records competition submission results for the candidate models and visualizes the comparison between validation accuracy and competition performance.

Confusion matrices are produced for the classical models and the convolution-based models to show which appliance categories are most frequently confused.

## Final Prediction and Submission

The final stage chooses between ROCKET and MiniRocket using their validation results. The selected pipeline is refitted using all training data, predictions are generated for the test recordings, and the predicted class distribution is displayed.

The submission is saved as:

```text
submission_final.csv
```

The file contains the test identifiers and their predicted appliance labels.

## Installation

The notebook installs the main time-series dependencies with:

```bash
pip install aeon dtaidistance tslearn
```

It also uses common scientific Python and machine-learning packages, including NumPy, pandas, SciPy, Matplotlib, Seaborn, and scikit-learn.

## How to Run

Open [TSAC_Group_Project.ipynb](TSAC_Group_Project.ipynb) in Jupyter or Visual Studio Code.

Before running the data-loading cells:

- Place `train.csv` and `test.csv` in the expected shared Google Drive folder.
- Make sure Google Drive access is available when running in Google Colab.
- Install the required Python packages.
- Run the cells in their displayed order so that transformed data, fitted models, and predictions are available to later cells.

The final prediction cell writes the submission file into the configured project directory.

## Limitations

The dataset is small, so cross-validation estimates may vary noticeably between splits. The ROCKET representation is powerful but difficult to interpret at the level of individual signal patterns. The Dynamic Time Warping workflow is also expensive to run compared with the other approaches.

## Future Improvements

Possible extensions include collecting more labeled recordings, adding frequency-domain features such as Fourier coefficients, investigating richer signal transformations, and combining predictions from ROCKET with the feature-based models through an ensemble.

