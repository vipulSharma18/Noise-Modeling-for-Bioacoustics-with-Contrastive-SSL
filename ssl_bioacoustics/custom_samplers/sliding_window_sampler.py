import torch


class TransformSlideBySlide(torch.nn.Module):
    """
    When we do sliding window, we add a num slides leading dimension to the input data.
    Any subsequent transform needs to be applied This transform is used to apply a transform to each slide individually.
    """

    def __init__(self, slide_transforms):
        super(TransformSlideBySlide, self).__init__()
        self.slide_transforms = slide_transforms

    def __call__(self, x):
        slides = x.shape[0]
        outputs = []
        for slide_ind in range(slides):
            outputs.append(self.slide_transforms(x[slide_ind]))
        outputs = torch.stack(outputs, dim=0)
        return outputs
