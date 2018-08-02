import torch
import torch.nn as nn


class EndToEndEvalModel(nn.Module):
    """Used for evaluation with known object indices.

    Outputs a single pose estimate corresponding to the specified object index.
    """
    def __init__(self, segm_model, pose_model):
        super(EndToEndEvalModel, self).__init__()
        self.segm_model = segm_model
        self.resize = nn.AdaptiveMaxPool2d((240, 320))
        self.pose_model = pose_model

    def forward(self, x, object_index, object_id):
        x = self.segm_model(x)
        _, x = x.max(1, keepdim=True)
        x = x.eq(object_id.view(object_id.size(0), 1, 1, 1)).float()
        mask = self.resize(x)
        return self.pose_model(mask, object_index)


class EndToEndModel(nn.Module):
    """Inference with unknown object indices.

    Outputs pose estimates for all detected objects in input image.
    """
    def __init__(self, segm_model, pose_model, object_names, object_ids):
        super(EndToEndModel, self).__init__()
        self.segm_model = segm_model
        self.resize = nn.AdaptiveMaxPool2d((240, 320))
        self.pose_model = pose_model
        self.object_names = object_names
        self.object_ids = object_ids

    def forward(self, x):
        assert x.size(0) == 1
        x = self.segm_model(x)
        _, x = x.max(1, keepdim=True)

        object_names = []
        positions = []
        orientations = []
        for i, object_name in enumerate(self.object_names):
            mask = self.resize(x.eq(self.object_ids[i]).float())
            if mask.sum().item() < 20:
                continue

            object_index = torch.LongTensor([i])
            position, orientation = self.pose_model(mask, object_index)

            object_names.append(object_name)
            positions.append(position[0].cpu().numpy())
            orientations.append(orientation[0].cpu().numpy())

        return x[0].cpu().numpy().squeeze(0), object_names, positions, orientations
