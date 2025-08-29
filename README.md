# ML Challenge — Feature Extraction & Baseline Model

This repository contains two main scripts that take you from raw JSON measurement data to a Kaggle-style submission:

1. **Feature extraction** &rarr; builds train/test CSVs from JSON  
2. **ML example** &rarr; trains a DNN and writes predictions for the test set  

Additionally, **COSMOS_EDFA_Dataset** is under `dataset/`, which provides extra EDFA measurement data. These are optional to use, but available if you want to extend training and evaluation.  

---

## Environment

- Recommended Python version: **3.11**  

```bash
pip install -U pip
pip install pandas numpy matplotlib tensorflow prettytable scipy scikit-learn
```

> The feature extraction script imports `libs.edfa_feature_extraction_libs` and `libs.edfaBasicLib`.  

---

## Expected Folder Layout

```
.
├── code
│   ├── kaggle_feature_extraction_user.ipynb
│   ├── ML_example_kaggle.ipynb
│   └── libs
│       ├── edfaBasicLib.py
│       └── edfa_feature_extraction_libs.py
│
├── dataset
│   ├── COSMOS_EDFA_Dataset        # optional additional data
│   │   ├── booster/{15dB,18dB,21dB}/...
│   │   └── preamp/{15dB,18dB,21dB,24dB,27dB}/...
│   └── ML_challenge_user
│       ├── Test
│       │   ├── aging/*.json
│       │   ├── shb/*.json
│       │   └── unseen/*.json
│       └── Train
│           ├── aging/*.json
│           ├── shb/*.json
│           └── unseen/*.json
│
├── Features
│   ├── Test
│   │   ├── test_features.csv
│   │   ├── test_labels.csv
│   │   ├── aging/features/*.csv
│   │   ├── shb/features/*.csv
│   │   └── unseen/features/*.csv
│   └── Train
│       ├── train_features.csv
│       ├── train_labels.csv
│       ├── COSMOS_features.csv          # optional features
│       ├── COSMOS_labels.csv            # optional labels
│       ├── aging/{features,labels}/*.csv
│       ├── shb/{features,labels}/*.csv
│       └── unseen/{features,labels}/*.csv
│
├── figures
│
├── manual
│   └── Ranges_ProductGuide_ROADM.pdf
│
└── model
    └── ML_example_model.h5
```

---

## What Each Script Does

### `code/kaggle_feature_extraction_user.ipynb`

- Reads raw JSON files from:
  - `dataset/ML_challenge_user/Train/{shb,aging,unseen}`
  - `dataset/ML_challenge_user/Test/{shb,aging,unseen}`
  - (Optionally) `dataset/COSMOS_EDFA_Dataset/{booster,preamp}/*/*.json`
- Extracts features using `featureExtraction_ML` and utilities in `libs/`.
- Infers:
  - **EDFA type** (`preamp` or `booster`)
  - **channel type** (`random`, `fix`, `goalpost`, `extraRandom`, `extraLow`)
  - **EDFA name** &rarr; mapped to index via dictionary
- Outputs intermediate CSVs for each dataset.
- Splits each CSV into:
  - `*_features.csv` (input features)
  - `*_labels.csv` (target gain spectra, masked by WSS activity)
- Combines into Kaggle-style files:
  - `Features/Train/train_features.csv`
  - `Features/Train/train_labels.csv`
  - `Features/Test/test_features.csv` (adds `ID`, `Usage`, `Category`)
  - `Features/Train/COSMOS_features.csv` and `COSMOS_labels.csv` (optional)

### `code/ML_example_kaggle.ipynb`

- Loads:
  - `Features/Train/train_features.csv`
  - `Features/Train/train_labels.csv`
  - `Features/Test/test_features.csv`
  - (Optionally) `Features/Train/COSMOS_features.csv` and `COSMOS_labels.csv`
- Defines and trains a DNN:
  - Custom **L2 loss** computed only on loaded channels
  - Optimizer = 500 epochs, 20% validation split
- Saves trained model:
  - `model/ML_example_model.h5`
- Predicts on test set, inserts Kaggle `ID`.
- Outputs Kaggle-style submission:
  - `Features/Test/test_labels.csv`

---

## How to Run (VS Code)

1. Open the repo in VS Code.  
2. Open `code/kaggle_feature_extraction_user.ipynb`.  
3. Select the correct Python kernel.  
4. Run all cells. This generates CSVs under `Features/Train` and `Features/Test`.  
   - If you want to include COSMOS data, adjust the notebook accordingly.  
5. Open `code/ML_example_kaggle.ipynb`.  
6. Run all cells. Outputs:  
   - `model/ML_example_model.h5`  
   - `figures/*.png` (if plotting cells executed)  
   - `Features/Test/test_labels.csv` (predictions)  

---

## Outputs

After running both scripts you should have:

- `Features/Train/train_features.csv`  
- `Features/Train/train_labels.csv`  
- `Features/Test/test_features.csv`  
- `Features/Test/test_labels.csv` &larr; Kaggle-style predictions  
- `model/ML_example_model.h5`  
- `figures/*.png` (optional plots)  
- `Features/Train/COSMOS_features.csv`, `COSMOS_labels.csv` (optional, if COSMOS used)

