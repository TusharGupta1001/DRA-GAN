import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset  # Add Dataset here
from PIL import Image, UnidentifiedImageError
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.optim as optim
import torch.nn as nn


# Generator
class generator(nn.Module):
    def __init__(self, input_dim=62, output_dim=3, input_size=64):
        super(generator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (input_size // 4) * (input_size // 4)),
            nn.BatchNorm1d(128 * (input_size // 4) * (input_size // 4)),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_dim, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 16, 16)  # Reshape to [batch_size, 128, 16, 16]
        out = self.deconv(x)
        return out

# Discriminator
class discriminator(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, input_size=64):
        super(discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * (input_size // 4) * (input_size // 4), 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        out = self.fc(x)
        return out

# Generator definition remains unchanged

# Discriminator definition remains unchanged

# Function to load the CelebA dataset
# Custom CelebA Dataset
class CelebADataset(Dataset):
    def __init__(self, image_dir, transform=None,limit=50000):
        self.image_dir = image_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))][:limit]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])

        try:
            image = Image.open(img_path).convert('RGB')  # Convert image to RGB
        except UnidentifiedImageError:
            print(f"Skipping corrupted image: {img_path}")
            return self.__getitem__((idx + 1) % len(self.image_filenames))  # Skip to the next image

        if self.transform:
            image = self.transform(image)
        return image, 0  # Return a dummy label since CelebA doesn't have class labels


# Function to load the CelebA dataset using the custom dataset class
def load_celeba_data(image_dir, batch_size, image_size,limit=50000):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize image to [-1, 1] range
    ])

    # Use the custom CelebA dataset class
    dataset = CelebADataset(image_dir, transform=transform,limit=limit)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader



# DRAGAN class for training and generating images
class DRAGAN(object):
    def __init__(self, args):
        # Parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62
        self.lambda_ = 0.25

        # Load CelebA dataset
        image_dir = os.path.join(args.dataset)
        self.data_loader = load_celeba_data(image_dir, self.batch_size, self.input_size,limit=50000)

        # Initialize networks
        data = next(iter(self.data_loader))[0]  # Get a batch of real images to define input size
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        self.print_network(self.G)
        self.print_network(self.D)
        print('-----------------------------------------------')

        # Fixed noise for generating images
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("Total number of parameters: {}".format(num_params))

    def train(self):
        for epoch in range(self.epoch):
            for i, (real_images, _) in enumerate(self.data_loader):

                # Check if the batch size is less than the expected batch size
                if real_images.size(0) < self.batch_size:
                    continue  # Skip this iteration

                # Train Discriminator
                real_images = Variable(real_images)
                if self.gpu_mode:
                    real_images = real_images.cuda()

                z = Variable(torch.rand(real_images.size(0), self.z_dim))  # Use the actual batch size
                if self.gpu_mode:
                    z = z.cuda()

                fake_images = self.G(z)

                # Real and fake labels
                real_labels = Variable(torch.ones(real_images.size(0), 1))  # Use actual batch size
                fake_labels = Variable(torch.zeros(real_images.size(0), 1))  # Use actual batch size
                if self.gpu_mode:
                    real_labels = real_labels.cuda()
                    fake_labels = fake_labels.cuda()

                # Compute discriminator loss on real images
                D_real_loss = self.BCE_loss(self.D(real_images), real_labels)

                # Compute discriminator loss on fake images
                D_fake_loss = self.BCE_loss(self.D(fake_images), fake_labels)

                D_loss = D_real_loss + D_fake_loss

                # Backprop and optimize D
                self.D_optimizer.zero_grad()
                D_loss.backward()
                self.D_optimizer.step()

                # Train Generator
                z = Variable(torch.rand(real_images.size(0), self.z_dim))  # Use the actual batch size
                if self.gpu_mode:
                    z = z.cuda()

                fake_images = self.G(z)
                G_loss = self.BCE_loss(self.D(fake_images), real_labels)  # Use actual batch size

                # Backprop and optimize G
                self.G_optimizer.zero_grad()
                G_loss.backward()
                self.G_optimizer.step()

                if i % 100 == 0:
                    print(f'Epoch [{epoch}/{self.epoch}], Step [{i}/{len(self.data_loader)}], '
                          f'D Loss: {D_loss.item()}, G Loss: {G_loss.item()}')

            # Save generated images and model checkpoints
            self.save_results(epoch)

    def save_results(self, epoch):
        z = Variable(torch.rand(self.batch_size, self.z_dim))
        if self.gpu_mode:
            z = z.cuda()
        generated_images = self.G(z)
        save_image(generated_images.data, os.path.join(self.result_dir, f'generated_{epoch}.png'), normalize=True)


# Example usage
class Args:
    epoch = 25
    batch_size = 64
    save_dir = r'C:\Users\srija\Desktop\research paper\deepfake detection\saves'
    result_dir = r'C:\Users\srija\Desktop\research paper\deepfake detection\results'
    dataset = r'C:\Users\srija\Desktop\research paper\deepfake detection\data\img_align_celeba'
    log_dir = './logs'
    gpu_mode = False  # Set False if not using GPU
    gan_type = 'DRAGAN'
    input_size = 64
    lrG = 0.0002
    lrD = 0.0002
    beta1 = 0.5
    beta2 = 0.999


args = Args()
model = DRAGAN(args)
model.train()
