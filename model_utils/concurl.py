import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision import models


class Unsup(nn.Module):
    def __init__(
        self, backbone, n_clusters, out_dim=256,
        normalize=True,
        arch='resnet18',
        hidden_mlp=2048, rand_transforms=None,
        use_no_grad=False,
        use_sobel=False,
        include_rgb=True,
        transform_generator=None,
    ):
        """Class definition for the model. 

        Args:
            backbone (object): Here, backbone is an NCE class object
            n_clusters (int): Number of clusters, used to initialize the number of prototypes
            out_dim (int, optional): dimension of the embeddings. Defaults to 256.
            normalize (bool, optional): Normalize the embeddings before passing through prototypes . Defaults to True.
            arch (str, optional): resnet architecture used inside backbone. Helps to set input dim of the projection head. Defaults to 'resnet18'.
            hidden_mlp (int, optional): Dimension of the hidden layers of projection head (MLP). Defaults to 2048.
            rand_transforms ([type], optional): list of random transformations to apply to embeddings. Defaults to None.
            use_no_grad (bool, optional): track gradients or not for normalizing prototypes. Defaults to False.
            use_sobel (bool, optional): Apply sobel transformation. Defaults to False.
            include_rgb (bool, optional): Use rgb or grayscale image. Defaults to True.
            transform_generator (object, optional): helper function to generate random transformations. Defaults to None.

        Raises:
            NotImplementedError: when architecture is not a resnet model
        """

        super(Unsup, self).__init__()

        if backbone is None:
            self.backbone = models.__dict__['resnet18'](pretrained=False)
        else:
            self.backbone = backbone

        self.use_sobel = use_sobel
        self.include_rgb = include_rgb

        if use_sobel:
            grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0,0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1,0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

        self.n_clusters = n_clusters

        self.normalize = normalize
        self.projection_head = []
        self.transform_generator = transform_generator
        if rand_transforms is None:
            self.rand_transforms = []
        else:
            self.rand_transforms = rand_transforms

        self.n_transforms = len(self.rand_transforms)
        self.use_no_grad = use_no_grad

        if arch in ['resnet18', 'resnet34']:
            proj_head_indim = 512
        elif arch in ['resnet50','resnet101']:
            proj_head_indim = 2048
        else:
            raise NotImplementedError

        self.projection_head = nn.Sequential(
            nn.Linear(proj_head_indim, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, out_dim),
        )
        
        self.prototypes = nn.Linear(out_dim, n_clusters, bias=False)

    def update_transforms(self):
        self.rand_transforms = self.transform_generator.getTransformations(reInit=True)

    def update_moving_average(self):
        if self.use_byol:
            self.backbone.update_moving_average()
        

    def forward_rand_transform(self, vec1, transf):
        """gets output of backbone, and computes random projection on the 
        corresponding layers and prototypes. 
        Returns the normalized output of the transformed layer and prototypes


        """

        temp_rand_one = torch.mm(vec1, transf.T)

        temp_rand_one = nn.functional.normalize(temp_rand_one, dim=1, p=2)

        if self.use_no_grad:
            with torch.no_grad():
                temp_proto = torch.mm(self.prototypes.weight.data.clone(), transf.T)
                temp_proto = nn.functional.normalize(temp_proto, dim=1, p=2)

        else:
            temp_proto = torch.mm(self.prototypes.weight.data.clone(), transf.T)
            temp_proto = nn.functional.normalize(temp_proto, dim=1, p=2)


        out_rand_one = torch.mm(temp_rand_one, temp_proto.T)

        return {'cTz': out_rand_one}


    def forward(self, batch):
        """Computes a forward pass for the concurl object. 
            Note that the input batch is of a single view. forward is called once for each view

            Outputs backbone_out, output, rand_out
            backbone_out is output of backbone (NCE)
            output is a dict with keys ['clust_head', 'cTz']. output['cTz'] are the outcodes
            rand_out is a list of dicts. rand_out[i] is the output using of each random transformation, each of which is a dict as above.

        Args:
            batch ([type]): Batch of images 

        Returns:
            list: output of backbone, dict with outputs after passing embeddings through prototypes, and list of dicts outputs similar outputs of each random transformation
        """
        # get backbone output
        if self.sobel is None:
            backbone_feat, backbone_out = self.backbone.forward(batch, return_penultimate=True)
        else:
            temp_batch = self.sobel(batch)
            if self.include_rgb:
                backbone_feat, backbone_out = self.backbone(torch.cat([batch, temp_batch], dim=1), return_penultimate=True)
            else:
                backbone_feat, backbone_out = self.backbone.forward(temp_batch, return_penultimate=True)

        # computes inner product of prototypes with the representations
        output = {}
        
        temp_1 = self.projection_head(backbone_feat)

        rand_out = []
        for ind in range(self.n_transforms):
            rand_out.append(
                self.forward_rand_transform(
                    temp_1, self.rand_transforms[ind]
                )
            )
        output['clust_head'] = temp_1.clone().detach()

        if self.normalize:
            temp_1 = nn.functional.normalize(temp_1, dim=1, p=2)

        output['cTz'] = self.prototypes(temp_1)

        return backbone_out, output, rand_out


