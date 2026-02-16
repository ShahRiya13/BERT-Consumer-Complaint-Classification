# BERT-Consumer-Complaint-Classification
**Overview**

This project fine-tunes a BERT-based neural network for multi-class classification of consumer complaints. The dataset contains approximately 600,000 records from the CFPB complaint database.

**Here:**

consumer_complaint_narrative as the input text

product as the target label

The task is formulated as a multi-class classification problem.

**Data Preprocessing**

Removed records where consumer_complaint_narrative is empty

Selected only the required columns:

consumer_complaint_narrative

product

Encoded product labels into numeric form

Split dataset into:

70% Training

30% Testing

**Model Architecture**

Instead of using BertForSequenceClassification, this implementation uses:

from transformers import BertTokenizer, BertModel

Forward Process

Tokenize complaint narratives using BertTokenizer

Pass tokens into BertModel

Extract the CLS token embedding

Feed CLS embedding into a custom Feed-Forward Network (FFN)

**Output class probabilities**

This allows full control over the classification head and embedding usage.

Custom Torch Dataset

A custom torch.utils.data.Dataset class was implemented to:

Handle tokenization

Return input IDs, attention masks, and labels

Support batching with DataLoader

Training Details

Loss Function: CrossEntropyLoss

**Optimizer: AdamW**

Train/Test Split: 70/30

Multi-class classification setup

**Evaluation Metrics**

The model is evaluated on the test set using:

**Accuracy**

AUC (Area Under the ROC Curve) using multi-class AUC computation

**Key Components Explained**

The 5-minute video covers:

Dataset filtering and preprocessing

Custom PyTorch Dataset implementation

BERT embedding extraction using CLS token

FFN classification head

Training loop and optimization

Accuracy and AUC evaluation

