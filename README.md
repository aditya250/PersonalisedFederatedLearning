# PersonalisedFederatedLearning



## 🚀 Overview
This project implements **Personalized Federated Learning (PFL)** using the **Flower (FL) framework** and **PyTorch**. It evaluates how different datasets (CIFAR-10, Fashion-MNIST, and SVHN) can benefit from personalized models while using a federated learning setup.

🔹 **Key Features:**
- Uses **CNN-based models** for image classification.
- Implements **client-specific personalization** via fine-tuning.
- Supports **multiple datasets** in a federated learning setup.
- Utilizes **FedAvg and FedProx** strategies.
- Optimized for **GPU acceleration** (if available).

## 📦 Installation
To run this project, ensure you have **Python 3.8+** and install dependencies:

```bash
pip install -r requirements.txt
```

Or manually install key dependencies:
```bash
pip install flwr torch torchvision numpy
```

## 📂 Dataset Support
The project works with:
- **CIFAR-10** 🏞️
- **Fashion-MNIST** 👕
- **SVHN** 🔢

These datasets will be **automatically downloaded** when running the script.

## 🎯 Model Architecture
A CNN-based model is used, with dataset-specific adjustments:
```python
class CNN_Model(nn.Module):
    def __init__(self, dataset_name):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc_config = nn.Linear(64 * 8 * 8, 128)
        self.fc_op = self._get_fc_layers(dataset_name)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc_config(x))
        x = self.fc_op(x)
        return F.softmax(x, dim=1)
```

## ⚙️ Running the Simulation
To start federated learning with personalized models:
```bash
python Personalised_Fed_Learn.py
```

## 🔍 How It Works
1. **Clients Load Datasets** 🗂️
   - Each client loads one dataset: CIFAR-10, Fashion-MNIST, or SVHN.

2. **Federated Training Begins** 🔄
   - **FedAvg** is used for global model aggregation.
   - **FedProx** introduces regularization to maintain proximity to global models.

3. **Client Personalization** 🎯
   - Each client **fine-tunes the model** on its dataset.

4. **Evaluation** 📊
   - Clients report back accuracy and loss metrics.

## 📈 Results & Performance
After training, clients print accuracy results, for example:
```bash
Client 1: Accuracy = 87.4%
Client 2: Accuracy = 85.2%
Client 3: Accuracy = 89.1%
```

## 🤝 Contributing
Contributions are welcome! Feel free to fork, modify, and submit a PR.

## 📝 License
This project is licensed under the **MIT License**.

---
Happy coding! 🚀

