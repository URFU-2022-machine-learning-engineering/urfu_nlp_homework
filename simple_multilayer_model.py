import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Processed_Text'], df['Sentiment'], test_size=0.2, random_state=42)

# Vectorize the text using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Convert text data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_vec.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_vec.toarray(), dtype=torch.float32)

# Convert labels to tensors
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Assuming you have a multi-label model, modify the output dimension accordingly
output_dim = len(df['Sentiment'].unique())

# Instantiate the model with dropout
class MultiLabelModelWithDropout(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super(MultiLabelModelWithDropout, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # No sigmoid activation here

# Instantiate the model with dropout
model = MultiLabelModelWithDropout(input_dim=X_train_tensor.shape[1], output_dim=output_dim, dropout_rate=0.5)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Experiment with different learning rates

# Training loop with dropout
num_epochs = 50
train_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    
    # Add L2 regularization
    l2_reg = 0.001
    for param in model.parameters():
        loss += l2_reg * torch.sum(param.pow(2))
    
    loss.backward()
    optimizer.step()
    
    # Append the training loss for visualization
    train_losses.append(loss.item())

    # Print and visualize the training loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Plot the training loss
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()

# Evaluation loop
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    probabilities = F.softmax(predictions, dim=1)
    _, predicted_labels = torch.max(probabilities, 1)

# Calculate metrics
roc_auc = roc_auc_score(y_test_tensor, probabilities, multi_class='ovr', average='macro')
f1 = f1_score(y_test_tensor, predicted_labels, average='macro')

print(f"ROC AUC: {roc_auc:.4f}, F1 Score: {f1:.4f}")