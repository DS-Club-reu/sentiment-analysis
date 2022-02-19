import time
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
import numpy as np
import pandas as pd
import nltk
import pymorphy2
from nltk.corpus import stopwords
import re 
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
nltk.download("stopwords")
mytokenizer=AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")

class BertClassifier(nn.Module):
    """
    Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """

        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 312, 32, 2

        # Instantiate BERT model
        self.bert = AutoModel.from_pretrained("cointegrated/rubert-tiny")

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """

        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

class SentimentDL:
    """
        A class used to load, save model and predict sentiment.

        Methods
        ----------
    train(self, model)-> model class:
        Returns trained model
        
        test(self)-> str
        Returns str with predition 'Позитивная(positive)' or 'Негативная(negative)'
        
        save_model(self)-> None
        Save model to path with pth format.
        
        load_model(self)-> model class:
            Returns model that was uploaded from state_dict
            
    """

    def __init__(self, model=BertClassifier(freeze_bert=False), lematizer=pymorphy2.MorphAnalyzer(), 
                max_len=64,
                tokenizer=mytokenizer):

        "@param    model: a BertModel object"
        self.max_len = max_len
        self.model = model
        self.lematizer = lematizer
        self.tokenizer = tokenizer

        # Check device
        if torch.cuda.is_available():    
            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

        self.model = model.to(self.device)

    def preprocessing(self, data):
        """
        Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                    tokens should be attended to by the model.
        """
        
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in data:
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            text_preprocessed = self.text_preprocessing(sent)
            encoded_sent = self.tokenizer.encode_plus(
                text=text_preprocessed,  # Preprocess sentence
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                max_length=self.max_len,                  # Max length to truncate/pad
                pad_to_max_length=True,         # Pad sentence to max length
                #return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True      # Return attention mask
                )
            
            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        return input_ids, attention_masks
    
    def text_preprocessing(self, s):
        """
        - reduce to the subscript
        - delete "@name"
        - delete punctuation marks except "?"
        - delete other special characters
        - remove stop words
        - we bring all the words to the initial form
        """
        
        s = s.lower()
        s = re.sub(r'(@.*?)[\s]', ' ', s)
        s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
        s = re.sub(r'[^\w\s\?]', ' ', s)
        s = re.sub(r'([\;\:\|•«\n])', ' ', s)
        s = " ".join([word for word in s.split()
                    if word not in stopwords.words('russian')])
        s = " ".join([self.lematizer.parse(word)[0].normal_form for word in s.split()])
        s = re.sub(r'\s+', ' ', s).strip()
        
        return s

    def training_model(self, data):
        """
        @param    data: prepeared train dataloader
        @param    model: a BertModel object

        Returns a trained model class
        """

        # Settings of the model train
        loss_fn = nn.CrossEntropyLoss()
        train_dataloader = data 
        val_dataloader=None 
        epochs=4
        evaluation=False
        optimizer = AdamW(self.model.parameters(),
                        lr=5e-5,    # Default learning rate
                        eps=1e-8    # Default epsilon value
                        )

        # Total number of training steps
        total_steps = len(train_dataloader) * epochs

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0, # Default value
                                                    num_training_steps=total_steps)
        
        # Start training loop
        print("Start training...\n")
        for epoch_i in range(epochs):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-"*70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                batch_counts +=1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

                # Zero out any previously calculated gradients
                self.model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = self.model(b_input_ids, b_attn_mask)

                # Compute loss and accumulate the loss values
                loss = loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)
            print("-"*70)
        
        return self.model

    def predict(self, text_to_predict):
        """
        @param   text_to_predict: string that contain feedback to predict

        Returns a str with sentiment prediction
        """
    
        # Prepear data to load
        test_inputs, test_masks = self.preprocessing(pd.Series(text_to_predict))
        test_dataset = TensorDataset(test_inputs, test_masks)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)
        
        # Compute predicted probabilities on the test set
        probs = self.bert_predict(test_dataloader)

        # Get predictions from the probabilities
        threshold = 0.75
        preds = np.where(probs[:, 1] > threshold, 1, 0)
        return("Positive" if preds[0] == 1 else "Negative")

    def bert_predict(self, test_dataloader):
        """
        Perform a forward pass on the trained BERT model to predict probabilities
        on the test set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        
        self.model.eval()
        all_logits = []

        # For each batch in our test set...
        for batch in test_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask = tuple(obj.to(self.device) for obj in batch)[:2]
            # Compute logits
            with torch.no_grad():
                logits = self.model(b_input_ids, b_attn_mask)
            all_logits.append(logits)
        # Concatenate logits from each batch
        all_logits = torch.cat(all_logits, dim=0)
        # Apply softmax to calculate probabilities
        probs = F.softmax(all_logits, dim=1).cpu().numpy()

        return probs

    def save_model(self, path):
            """
            @param    path: str path where you want to save your model with extension .pth

            Save model to path
            """

            torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        """
        @param    path: str path from where you want to load your model

        Returns model class
        """

        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    