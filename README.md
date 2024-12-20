## Abstract

Music is an immensely popular form of art – one that includes a relatively unique corpus of
language in the form of lyrics. To examine the performance of modern NLP tools in evaluating lyrics, we set the task of analyzing genres and eras of music using encoder and classification models, both in isolation and comparatively. We find that a fine-tuned BERT model largely agrees with human comparative analysis of modern musical eras, but reaches novel conclusions when examining variation within a specific era of a specific genre, especially within rap. These results point towards a qualitative gap between the ability to parse standard English and the ability to parse lyrics specifically. Through data classification, the results also suggest the existence of specific distinctive eras of music, most primarily within the decades of the 1980s and the 2010s.

## Setup

Make sure you have access to Google Colab and T4 GPU. Supercloud is recommended for fine-tuning, but not necessary as the results of the fine-tuned jobs are stored in `distilbert_models` and the final dataset is stored in `data`


### Genre Clustering and Date Classificiation:

Steps:
1. Upload the `distilbert_models` and `data` folders to Google Drive and mount your Google Drive. Add your local path to this folder when mounting your Google Drive.
2. Run the `clustering_and_date_classification.ipynb` in Google Colab on 1 T4 GPU. 

### Fine-Tuning (Optional):

The fine-tuning was deployed on a NVIDIA A100 GPU on Supercloud.

To run the job, submitted via SLURM:

```
sbatch script.sh
```

A template script is available at `script.sh`.

Any of the params concerning tuning in `main.py` may be adjusted. We've included the parameters that contributed to the best runs and were used for downstream analysis. Feel free to adjust them as needed.

### Initial Data Processing (Optional):

All data processing code should be accessible in the `initial_data_processing` folder, so this is not necessary. However, here are steps to run:

1. Download our large dataset [[here](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information/code)].
2. Upload this dataset to a Google Drive folder
3. Mount the Google Drive folder and change the file path to load the csv accordingly
4. Run `initial_data_processing.ipynb` in Google Colab on 1 T4 GPU with your appropriate local file path to the csv. This file should result in and save a smaller and processed csv

The rest of the notebook is standalone and should be functional. 
