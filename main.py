from torchvision.datasets import CIFAR10
from torchvision import transforms
from mycnn import SimpleCNN

#Optional: transform to tensor
transform = transforms.ToTensor()

dataset = CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

print(len(dataset))        # 50000
image, label = dataset[0]
print(image.shape, label)  # (3, 32, 32)
