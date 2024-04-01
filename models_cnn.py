import json
import random
import numpy as np
import torch
import transformers
import torch.nn as nn
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models

class Model(nn.Module):
    def __init__(
        self,
        name: str,
        num_choices: int,
        device: str
    ):
        super().__init__()
        
        self.name = name
        self.imageEmbeddingSize = 2048
        self.num_choices = num_choices
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device)
        self.cnnlinear = nn.Linear(self.imageEmbeddingSize, 768)
        self.model.classifier = nn.Linear(768, 768)
        
        self.max_length = 512
        
        if "base" in name:
            self.hidden_size = 768
        elif "xx-large" in name:
            self.hidden_size = 1536
        elif "large" in name:
            self.hidden_size = 1024
        elif "tiny" in name:
            self.hidden_size = 128
        elif "small" in name:
            self.hidden_size = 768
        elif "aristo" in name:
            self.hidden_size = 768
            
        self.ce_loss_func = nn.CrossEntropyLoss()
        self.scorer = nn.Linear(self.hidden_size * 2, 2)
        
    def score_input(self, content):
        batch = self.tokenizer(
            content, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        out = self.model(
            batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device),
            output_hidden_states=True
        )

        output = torch.mean(out["logits"], dim = 1)

        return output

    def forward(self, batch):
        images, content, labels = batch
        #print("images = ",images[0].shape)
        images = np.vstack(images)
        images = torch.from_numpy(images)
        
        imageEmbedding = images.to(self.device)
        imageEmbeddingOut = self.cnnlinear(imageEmbedding)
        #print(" before = ", imageEmbedding.size())
        #imageEmbedding = img_label * imageEmbedding
        #print(" after = ", imageEmbedding.size())

        logits = self.score_input(content)    
        embeddings = torch.cat((imageEmbeddingOut, logits), dim=1)
        #print(" after cat = ", embeddings.size())
        output = self.scorer(embeddings)

        labels = torch.tensor(labels, dtype=torch.long).to(output.device)
        loss = self.ce_loss_func(output, labels)
        preds_cls = list(torch.argmax(output, 1).cpu().numpy())
        #print(output)
        positive_logits = output[:, 1]
        
        if self.num_choices!=-1:
            preds = torch.argmax(positive_logits.reshape(-1, self.num_choices), 1)
            preds = list(preds.cpu().numpy())
        else:
            preds = list(positive_logits.detach().cpu().numpy())
        
        #print(preds)
        #print(preds_cls)
        #print(positive_logits)
        return loss, preds, preds_cls