import torch
import torchvision.transforms as transforms

# Device & class labels
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class_list = ["empty", "person", "vehicle"]
  

# Transforms for images
pic_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



