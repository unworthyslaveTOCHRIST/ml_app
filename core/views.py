from django.shortcuts import render

# Create your views here.

import os
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target.reshape(-1, 1)

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Global model variable
model = ANN(X.shape[1], 10, y.shape[1])

def index(request):
    image_dir = os.path.join(settings.BASE_DIR, 'mlp_app/static/iris_images')
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    return render(request, 'core/index.html', {'images': images})

def predict(request):
    slider_value = float(request.GET.get('slider_value', 0.5))
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):
        outputs = model(X)
        loss = criterion(outputs, torch.argmax(y, dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return JsonResponse({ 'message': 'Training done', 'loss': loss.item() })
