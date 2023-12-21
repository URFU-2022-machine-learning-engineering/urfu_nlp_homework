import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator


# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data using pandas
df = pd.read_parquet('data/Tweets.parquet')

# Tokenization and text preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the 'Text' column
df['Processed_Text'] = df['Text'].apply(preprocess_text)

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(df['Processed_Text'], df['Sentiment'], test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Define tokenizer
tokenizer = get_tokenizer('basic_english')

# Build vocabulary
def build_vocab(texts):
    # Tokenize texts and create an iterator
    tokenized_texts = [tokenizer(text) for text in texts]
    return build_vocab_from_iterator(tokenized_texts, specials=['<unk>', '<pad>'])

vocab = build_vocab(X_train)

# Set indices for unknown and padding tokens
vocab.set_default_index(vocab['<unk>'])

# Numericalize text function
def numericalize_text(text):
    return [vocab[token] for token in tokenizer(text)]

# Convert texts to sequences of indices
X_train_seq = [numericalize_text(text) for text in X_train]
X_test_seq = [numericalize_text(text) for text in X_test]

# Pad sequences
max_len = 100  # Adjust as needed
X_train_padded = pad_sequence([torch.tensor(x) for x in X_train_seq], padding_value=vocab['<pad>'], batch_first=True)
X_test_padded = pad_sequence([torch.tensor(x) for x in X_test_seq], padding_value=vocab['<pad>'], batch_first=True)



# Convert labels to tensors
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Custom Dataset
class TextDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

# Create datasets
train_dataset = TextDataset(X_train_padded, y_train_tensor)
test_dataset = TextDataset(X_test_padded, y_test_tensor)

# DataLoaders
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# Create validation dataset and DataLoader
X_val_seq = [numericalize_text(text) for text in X_val]
X_val_padded = pad_sequence([torch.tensor(x) for x in X_val_seq], padding_value=vocab['<pad>'], batch_first=True)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
val_dataset = TextDataset(X_val_padded, y_val_tensor)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


# LSTM Model with Embedding Layer
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab['<pad>'])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Apply dropout to the output of the last LSTM layer
        out = self.fc(out)
        return out


# Instantiate the model
vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 128
num_layers = 2
output_dim = len(df['Sentiment'].unique())
model = LSTMModel(vocab_size, embed_dim, hidden_dim, output_dim, num_layers).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop with validation
num_epochs = 3
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()
            _, predicted_labels = torch.max(predictions, 1)
            correct += (predicted_labels == y_batch).sum().item()
            total += y_batch.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total
    train_losses.append(avg_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Plotting the training and validation losses
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting the validation accuracy
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# plt.show()

# Evaluation on the test set
model.eval()
test_predictions = []
test_true_labels = []
test_probabilities = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        predictions = model(X_batch)
        probabilities = torch.softmax(predictions, dim=1)
        _, predicted_labels = torch.max(probabilities, 1)
        
        test_predictions.extend(predicted_labels.cpu().numpy())
        test_true_labels.extend(y_batch.cpu().numpy())
        test_probabilities.extend(probabilities.cpu().numpy())

# Convert to numpy arrays for scoring
test_predictions = np.array(test_predictions)
test_true_labels = np.array(test_true_labels)
test_probabilities = np.array(test_probabilities)

# Calculate ROC AUC and F1 Score
roc_auc = roc_auc_score(test_true_labels, test_probabilities, multi_class='ovr', average='macro')
f1 = f1_score(test_true_labels, test_predictions, average='macro')

print(f"Test ROC AUC: {roc_auc:.4f}, Test F1 Score: {f1:.4f}")
