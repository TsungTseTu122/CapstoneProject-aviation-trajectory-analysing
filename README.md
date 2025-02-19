# aviation-trajectory-prediction
## Overview
This repository extends the pre-trained TrajAirNet model for general aviation trajectory prediction. The focus is on handling Out-of-Distribution (OOD) and Non-IID data using synthetic data augmentation and fine-tuning techniques. The dataset includes real-world flight trajectories, synthetic data, and OOD test data.

## Repository Structure
```
aviation-trajectory-prediction/
│   .gitignore
│   LICENSE
│   README.md
│   requirements.txt
│
├───Documents
│       47801873 Final Project.docx
│
├───sample_data
│   ├───OOD_test
│   ├───processed_dataset
│   ├───synthetic_data
│   ├───weather_data
│
├───src
│   │   check matches(OOD and indis).py
│   │   synthetic data pre-processing.py
│   │   test3.py
│   │   train3.py
│
├───model (Required Pre-trained Model Files)
│   │   trajairnet.py
│   │   utils.py
│   │   cvae_base.py
│   │   tcn_model.py
│   │   gat_model.py
│   │   gat_layers.py
```

## Required Files from the Original Repository
To train and test the model, download the following files from the original **TrajAirNet repository** and place them in the `src/` folder:

- `cvae_base.py`
- `gat_layers.py`
- `gat_model.py`
- `tcn_model.py`
- `trajairnet.py`
- `utils.py`

Download these files from the official repository: [TrajAirNet GitHub](https://github.com/castacks/trajairnet)  
After downloading, place them inside the `src/` directory.

## Setup Instructions

1. Clone the repository
```
git clone https://github.com/TsungTseTu122/aviation-trajectory-prediction.git
cd aviation-trajectory-prediction
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Preprocess Synthetic Data (Optional)
You can either use the preprocessed dataset or regenerate it using:
`python src/synthetic data pre-processing.py`
If you wish to preprocess the synthetic data yourself, use the provided Excel file `(df_traj_tr_data_export_20241012_1047541.xlsm)`.

4. Modify Paths in `train3.py` and `test3.py`

By default, `train3.py` assumes that datasets and the pre-trained model are stored in a specific directory:
```
sys.path.append(os.path.join(os.getcwd(), 'C:/Users/Michael/Desktop/Y2 S2/DATA7903-Data Science Capstone Project 2B/dataset and pre-trained model/pre-trained model'))
```
Change this path to match your directory structure where the pre-trained model files (trajairnet.py, utils.py...) are stored.

Similarly, update the dataset paths:
```
parser.add_argument('--dataset_folder', type=str, default='C:/Users/Michael/Desktop/Y2 S2/DATA7903-Data Science Capstone Project 2B/dataset and pre-trained model/pre-trained model/dataset')
```
Modify this to:
```
parser.add_argument('--dataset_folder', type=str, default='./sample_data')
```
5. Run Training
```
python src/train3.py --dataset_folder ./sample_data
```
6. Run Testing

Ensure paths in `test3.py` are also updated similarly. Then, run:
```
python src/test3.py
```

## Hyperparameter Adjustment

Users can freely modify the hyperparameters based on their own conditions and research objectives.

## Citation & Acknowledgment
- Original TrajAirNet Model: [GitHub Source](https://github.com/castacks/trajairnet)

- Preprocessing code for synthetic data was modified to address inconsistencies in the provided dataset.

## License

This project is licensed under the BSD 3-Clause License. See `LICENSE` for details.
