from operator import add
from PIL import ImageStat
import torch
from torchvision.transforms import functional
from datasets.dataset_retrieval import PlayingCardsDataset


# https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/13
class Stats(ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(map(add, self.h, other.h)))


def calculate_norms(dataset):
    dl = torch.utils.data.DataLoader(dataset, batch_size=30, num_workers=6)

    stats = None
    for image, _ in dl:
        for i in range(image.shape[0]):
            s = Stats(functional.to_pil_image(image[i]))
            if stats is None:
                stats = s
            else:
                stats += s

    print(f"mean: {stats.mean}\nstd: {stats.stddev}")
    # TODO: Write this to a file and implement functionality for reading from file


if __name__ == "__main__":
    calculate_norms(PlayingCardsDataset())

# mean: [198.535528698164, 186.41360832482462, 179.76456663467994]
# std: [78.065716597632, 85.66597062664934, 87.28034113077337]    