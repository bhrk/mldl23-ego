from torch import nn
import torch
from math import ceil
from models import I3D


class Classifier(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        self.avg_modality = model_args.avg_modality
        self.num_classes = model_args.num_classes
        self.num_clips = model_args.num_clips

        self.AvgPool = nn.AdaptiveAvgPool2d((1,1024))

        self.TRN=RelationModuleMultiScale(1024, 1024, self.num_clips)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_classes),
            nn.ReLU())

    def forward(self, x):

        if self.avg_modality == 'Pooling':
            x = self.AvgPool(x)
            x = x.view(-1, 1024)

        if self.avg_modality == 'TRN':
            x = self.TRN(x)
            x = x.sum(1)

        x = self.fc1(x)
        x= self.fc2(x)

        return x, {}

class RelationModuleMultiScale(nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, num_bottleneck, num_frames):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        # self.num_class = num_class
        self.num_frames = num_frames
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        )

            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_scale_1 = input[:, self.relations_scales[0][0] , :]
        act_scale_1 = act_scale_1.view(act_scale_1.size(0), self.scales[0] * self.img_feature_dim)
        act_scale_1 = self.fc_fusion_scales[0](act_scale_1)
        act_scale_1 = act_scale_1.unsqueeze(1) # add one dimension for the later concatenation
        act_all = act_scale_1.clone()

        for scaleID in range(1, len(self.scales)):
            act_relation_all = torch.zeros_like(act_scale_1)
            # iterate over the scales
            num_total_relations = len(self.relations_scales[scaleID])
            num_select_relations = self.subsample_scales[scaleID]
            idx_relations_evensample = [int(ceil(i * num_total_relations / num_select_relations)) for i in range(num_select_relations)]

            #for idx in idx_relations_randomsample:
            for idx in idx_relations_evensample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_relation = act_relation.unsqueeze(1)  # add one dimension for the later concatenation
                act_relation_all += act_relation

            act_all = torch.cat((act_all, act_relation_all), 1)
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))
