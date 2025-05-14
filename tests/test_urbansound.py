"""
Test the UrbanSoundDataset class.
"""
from ssl_bioacoustics.custom_datasets import UrbanSoundDataset
from torchvision.transforms import transforms
from PIL import Image


def save_image(rgb_tensor, name='output_rgb_image.png'):
    # rgb_tensor Shape: (C, H, W)
    rgb_uint8_tensor = (rgb_tensor * 255).byte().permute(1, 2, 0)  # (H, W, C)
    image = Image.fromarray(rgb_uint8_tensor.numpy(), mode='RGB')
    image.save(name)


def test_index_reproducibility():
    """
    Access the same index twice to see that the same sample is returned.
    Also test that different samples are returned for different indices.
    """
    size = 224
    s = 1  # strength of color jitter, 0.5 to 1.5 is mentioned to be typical in the GitHub repo

    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])

    # Instantiate the dataset
    dataset = UrbanSoundDataset(
        root_dir='/oscar/home/vsharm44/jobtmp/UrbanSound8k/images/archive/',
        fold=1,
        csv_file="/oscar/home/vsharm44/jobtmp/UrbanSound8k/images/archive/UrbanSound8K.csv",
        transform=transform
        )

    assert len(dataset) > 0

    image1, label1 = dataset[2]
    image2, label2 = dataset[2]

    save_image(image1, '/users/vsharm44/projects/ssl-bioacoustics/logs/figures/urbansound_image1.png')
    save_image(image2, '/users/vsharm44/projects/ssl-bioacoustics/logs/figures/urbansound_image2.png')
    assert label1 == label2

    image3, label3 = dataset[len(dataset)-5]

    assert label1 != label3
    save_image(image3, '/users/vsharm44/projects/ssl-bioacoustics/logs/figures/urbansound_image3.png')
