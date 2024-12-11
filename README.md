### Fine-Tuning:

The fine-tuning was deployed on a NVIDIA A-100 GPU on Supercloud.

To run the job:
```bash
python main.py
```

### Initial Data Processing:

All data processing code should be accessible in the data processing folder. Steps to run:

1. Download our data here: [insert link].
2. Upload this dataset to a Google Drive folder
3. Mount the Google Drive folder and change the file path to load the csv accordingly

The rest of the notebook is standalone and should be functional. 

### Genre Clustering and Date Classificiation:

Steps:
1. Upload the distilbert_models and data folders to Google Drive
2. Run the clustering_and_date_classification_FINAL.ipynb in Google Colab on 1 T4 GPU. 


