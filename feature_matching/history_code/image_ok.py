from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="imagenet/one", extra="imagenet/two")
    dataset.dump_extra()