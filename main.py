import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from matplotlib import pyplot as plt
from PIL import Image

label_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.ImageFolder(root="cifar10_images", transform=transform)


# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class SimpleNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return F.softmax(x, dim=1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

model = SimpleNN()

if input("train model? (y/n)") == "y":
    trainer = pl.Trainer(max_epochs=20, accelerator="auto", devices="auto")
    trainer.fit(model, train_loader, test_loader)
else:
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

torch.save(model.state_dict(), "model.pth")
# read and sort test.png
while input("again? (y/n)") == "y":

    test_image = Image.open("test.png").convert("RGB")
    test_image = test_image.resize((32, 32), Image.BILINEAR)

    with torch.no_grad():
        test_image_tensor = transform(test_image).unsqueeze(0)
        output = model(test_image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        print(f"Predicted class: {predicted_class}")

    # Display the image
    plt.imshow(test_image)
    plt.axis('off')
    plt.title(f"Predicted class: {label_names[predicted_class]}")
    plt.show()

    # Save model
    
