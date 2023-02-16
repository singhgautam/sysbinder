import glob

from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image


class GlobDataset(Dataset):
    def __init__(self, root, phase, img_size):
        self.root = root
        self.img_size = img_size
        self.total_imgs = sorted(glob.glob(root))

        if phase == 'train':
            self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * 0.7)]
        elif phase == 'val':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.7):int(len(self.total_imgs) * 0.85)]
        elif phase == 'test':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.85):]
        else:
            pass

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        tensor_image = self.transform(image)
        return tensor_image
