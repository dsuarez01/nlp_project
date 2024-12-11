## Abstract

Music is an immensely popular form of art –
one that includes a relatively unique corpus of
language in the form of lyrics. To examine the
performance of modern NLP tools in evaluat-
ing lyrics, we set the task of analyzing genres
and eras of music using encoder and classifi-
cation models, both in isolation and compara-
tively. We find that a fine-tuned BERT model
largely agrees with human comparative analy-
sis of modern musical eras, but reaches novel
conclusions when examining variation within
a specific era of a specific genre, especially
within rap. These results point towards a quali-
tative gap between the ability to parse standard
English and the ability to parse lyrics specifi-
cally. Through data classification, the results
also suggest the existence of specific distinc-
tive eras of music, most primarily within the
decades of the 1980s and the 2010s.

## Setup

Make sure you have access to Google Colab and T4 GPU. Supercloud is recommended for fine-tuning, but not necessary as the results of the fine-tuned jobs are stored in `distilbert_models`

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
1. Upload the `distilbert_models` and `data` folders to Google Drive
2. Run the clustering_and_date_classification_FINAL.ipynb in Google Colab on 1 T4 GPU. 


