# BERT-Consumer-Complaint-Classification
This project fine-tunes a BERT-based model for multi-class consumer complaint classification using the CFPB complaint dataset. Complaint narratives are encoded using BertModel, and the CLS token embedding is passed to a custom feed-forward network for classification. The model is trained with a 70:30 split and evaluated using accuracy and AUC.
