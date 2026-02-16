import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import re
import random
from collections import Counter
import os

# --- CONFIGURATION ---
class Config:
    TRAIN_FILE = 'data.txt'
    SEQ_LEN = 32          # Keep it short for fast learning
    EMBED_DIM = 64        # Small brain
    N_HEADS = 4
    N_LAYERS = 2
    BATCH_SIZE = 32
    EPOCHS = 500         # INCREASED: Small models need many loops
    LR = 1e-3
    MASK_PROB = 0.15

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. STRICT TOKENIZER (No Punctuation)
# ============================================================
class StrictTokenizer:
    def __init__(self):
        self.vocab = {}
        self.id_to_token = {}
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.unk_token = "[UNK]"
        
    def build_vocab(self, texts):
        all_tokens = []
        for text in texts:
            all_tokens.extend(self._tokenize(text))
        
        # Count and keep only words appearing > 1 time
        token_freq = Counter(all_tokens)
        self.vocab = {self.pad_token: 0, self.mask_token: 1, self.unk_token: 2}
        
        idx = 3
        for token, freq in token_freq.items():
            if freq > 1: # Filter rare words
                self.vocab[token] = idx
                idx += 1
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        print(f"Vocab Size: {len(self.vocab)} (Punctuation Removed)")
        
    def _tokenize(self, text):
        # 1. Handle [MASK] (case insensitive)
        text = re.sub(r'\[mask\]', '[MASK]', text, flags=re.IGNORECASE)
        # 2. Protect [MASK]
        text = text.replace('[MASK]', ' __MASK__ ')
        # 3. REMOVE ALL PUNCTUATION (Keep only letters, numbers, spaces)
        text = re.sub(r'[^a-zA-Z0-9\s\__MASK__]', '', text)
        # 4. Split and Lowercase
        tokens = []
        for w in text.split():
            if w == '__MASK__':
                tokens.append('[MASK]')
            else:
                tokens.append(w.lower())
        return tokens

    def encode(self, text):
        tokens = self._tokenize(text)
        return [self.vocab.get(t, self.vocab[self.unk_token]) for t in tokens]

# ============================================================
# 2. DATASET
# ============================================================
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # READ FILE
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found!")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # TOKENIZE EVERYTHING
        self.full_tokens = tokenizer.encode(text)
        
        # CREATE SLIDING WINDOWS
        self.samples = []
        for i in range(0, len(self.full_tokens) - seq_len, seq_len):
            self.samples.append(self.full_tokens[i:i+seq_len])
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Input sequence
        input_ids = torch.tensor(self.samples[idx])
        target_ids = input_ids.clone()
        
        # Masking Logic
        rand = torch.rand(input_ids.shape)
        # Mask 15% of tokens (but not PAD/UNK/Special)
        mask_mask = (rand < config.MASK_PROB) & (input_ids > 2)
        
        input_ids[mask_mask] = 1 # Set to [MASK] ID
        
        # Ignore loss for unmasked items
        target_ids[~mask_mask] = -100 
        
        return input_ids, target_ids

# ============================================================
# 3. MINI-BERT MODEL
# ============================================================
class MiniBERT(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers):
        super(MiniBERT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Embedding(config.SEQ_LEN, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            dim_feedforward=embed_dim*4, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        # Create position IDs [0, 1, 2...]
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoder(positions)
        x = self.transformer_encoder(x)
        return self.fc_out(x)

# ============================================================
# 4. PREDICT FUNCTION
# ============================================================
def predict(model, tokenizer, sentence):
    model.eval()
    # Prepare Input
    ids = tokenizer.encode(sentence)
    
    # Pad to correct length
    if len(ids) < config.SEQ_LEN:
        ids += [0] * (config.SEQ_LEN - len(ids))
    ids = ids[:config.SEQ_LEN]
    
    tensor_in = torch.tensor([ids]).to(device)
    
    # Find [MASK] index
    try:
        mask_idx = ids.index(1) # ID for [MASK] is 1
    except ValueError:
        print(f"No [MASK] found in: {sentence}")
        return

    with torch.no_grad():
        logits = model(tensor_in)
        predictions = logits[0, mask_idx].topk(5)
        
    print(f"\nInput: {sentence}")
    print("Predictions:")
    for score, idx in zip(predictions.values, predictions.indices):
        word = tokenizer.id_to_token.get(idx.item(), "[UNK]")
        print(f"  {word}: {score.item():.4f}")

# ============================================================
# 5. MAIN
# ============================================================
if __name__ == "__main__":
    # --- STEP 1: LOAD DATA ---
    print("Loading Data...")
    
    # SANITY CHECK: UNCOMMENT THIS IF IT FAILS AGAIN
    # with open('data.txt', 'w') as f:
    #     f.write(("Sherlock Holmes is a great detective. " * 100))
    
    with open(config.TRAIN_FILE, 'r', encoding='utf-8') as f:
        raw_text = [f.read()]

    tokenizer = StrictTokenizer()
    tokenizer.build_vocab(raw_text)
    
    dataset = TextDataset(config.TRAIN_FILE, tokenizer, config.SEQ_LEN)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # --- STEP 2: INIT MODEL ---
    model = MiniBERT(len(tokenizer.vocab), config.EMBED_DIM, config.N_HEADS, config.N_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # --- STEP 3: TRAIN ---
    print("Starting Training...")
    losses = []
    model.train()
    
    for epoch in range(config.EPOCHS):
        total_loss = 0
        for batch_in, batch_target in loader:
            batch_in, batch_target = batch_in.to(device), batch_target.to(device)
            
            optimizer.zero_grad()
            output = model(batch_in)
            
            # Reshape for Loss: (batch*seq, vocab_size) vs (batch*seq)
            loss = criterion(output.view(-1, len(tokenizer.vocab)), batch_target.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.EPOCHS} | Loss: {avg_loss:.4f}")

    # --- STEP 4: RESULTS ---
    plt.plot(losses)
    plt.savefig('loss.png')
    print("Saved loss.png")
    
    # Interactive Test
    predict(model, tokenizer, "Sherlock is a [MASK] detective")
    predict(model, tokenizer, "The cat sat on the [MASK]")