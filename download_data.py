"""Download CIFAR-10 once (about 170 MB) into data/cifar10 using torchvision."""
import torchvision
torchvision.datasets.CIFAR10("data/cifar10", train=True, download=True)
torchvision.datasets.CIFAR10("data/cifar10", train=False, download=True)
print("CIFAR-10 ready in data/cifar10")
