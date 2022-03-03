import spconv
from spconv.modules import SparseModule
import torch
from torch import nn

class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)
        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class UnetUnfold(SparseModule):

    def __init__(self, nPlanes, norm_fn, block_reps, block):
        super().__init__()
        
        self.block0 = self._make_layers(nPlanes[0], nPlanes[0], block_reps, norm_fn, indice_key=0)
        
        self.conv1 = spconv.SparseSequential(
            norm_fn(nPlanes[0]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(1))
        )
        self.block1 = self._make_layers(nPlanes[1], nPlanes[1], block_reps, norm_fn, indice_key=1)

        self.conv2 = spconv.SparseSequential(
            norm_fn(nPlanes[1]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[1], nPlanes[2], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(2))
        )
        self.block2 = self._make_layers(nPlanes[2], nPlanes[2], block_reps, norm_fn, indice_key=2)

        self.conv3 = spconv.SparseSequential(
            norm_fn(nPlanes[2]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[2], nPlanes[3], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(3))
        )
        self.block3 = self._make_layers(nPlanes[3], nPlanes[3], block_reps, norm_fn, indice_key=3)


        self.conv4 = spconv.SparseSequential(
            norm_fn(nPlanes[3]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[3], nPlanes[4], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(4))
        )
        self.block4 = self._make_layers(nPlanes[4], nPlanes[4], block_reps, norm_fn, indice_key=4)

        self.conv5 = spconv.SparseSequential(
            norm_fn(nPlanes[4]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[4], nPlanes[5], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(5))
        )
        self.block5 = self._make_layers(nPlanes[5], nPlanes[5], block_reps, norm_fn, indice_key=5)

        self.conv6 = spconv.SparseSequential(
            norm_fn(nPlanes[5]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[5], nPlanes[6], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(6))
        )
        self.block6 = self._make_layers(nPlanes[6], nPlanes[6], block_reps, norm_fn, indice_key=6)

        self.deconv6 = spconv.SparseSequential(
            norm_fn(nPlanes[6]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[6], nPlanes[5], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(6))
        )

        self.deblock5 = self._make_layers(nPlanes[5] * 2, nPlanes[5], block_reps, norm_fn, indice_key=5)
        self.deconv5 = spconv.SparseSequential(
            norm_fn(nPlanes[5]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[5], nPlanes[4], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(5))
        )

        self.deblock4 = self._make_layers(nPlanes[4] * 2, nPlanes[4], block_reps, norm_fn, indice_key=4)
        self.deconv4 = spconv.SparseSequential(
            norm_fn(nPlanes[4]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[4], nPlanes[3], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(4))
        )

        self.deblock3 = self._make_layers(nPlanes[3] * 2, nPlanes[3], block_reps, norm_fn, indice_key=3)
        self.deconv3 = spconv.SparseSequential(
            norm_fn(nPlanes[3]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[3], nPlanes[2], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(3))
        )

        self.deblock2 = self._make_layers(nPlanes[2] * 2, nPlanes[2], block_reps, norm_fn, indice_key=2)
        self.deconv2 = spconv.SparseSequential(
            norm_fn(nPlanes[2]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[2], nPlanes[1], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(2))
        )

        self.deblock1 = self._make_layers(nPlanes[1] * 2, nPlanes[1], block_reps, norm_fn, indice_key=1)
        self.deconv1 = spconv.SparseSequential(
            norm_fn(nPlanes[1]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(1))
        )

        self.deblock0 = self._make_layers(nPlanes[0] * 2, nPlanes[0], block_reps, norm_fn, indice_key=0)

        self.fc = nn.Linear(nPlanes[-1], nPlanes[-1])
        self.conv = spconv.SparseSequential(
            norm_fn(nPlanes[-1] * 2),
            nn.ReLU(), 
            spconv.SubMConv3d(nPlanes[-1] * 2, nPlanes[-1], 1, bias=True))

    def _make_layers(self, inplanes, planes, block_reps, norm_fn, indice_key=0):
        blocks = [ResidualBlock(inplanes, planes, norm_fn, indice_key='bb_subm{}'.format(indice_key))]
        for i in range(block_reps - 1):
            blocks.append(ResidualBlock(planes, planes, norm_fn, indice_key='bb_subm{}'.format(indice_key)))
        return spconv.SparseSequential(*blocks)

    def pool(self, x):
        batch_size = x.batch_size
        N = x.features.shape[0]
        C = x.features.shape[1]
        x_pool = x.features.new_zeros(batch_size, C)
        for i in range(batch_size):
            inds = x.indices[:, 0] == i
            x_pool[i] = x.features[inds].mean(dim=0)
        x_pool = self.fc(x_pool)
        indices = x.indices[:, 0].long()
        x_pool_expand = x_pool[indices]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        x = self.conv(x)
        return x

    def forward(self, x):
        out0 = self.block0(x)
        
        out1 = self.conv1(out0)
        out1 = self.block1(out1)

        out2 = self.conv2(out1)
        out2 = self.block2(out2)

        out3 = self.conv3(out2)
        out3 = self.block3(out3)

        out4 = self.conv4(out3)
        out4 = self.block4(out4)

        out5 = self.conv5(out4)
        out5 = self.block5(out5)

        out6 = self.conv6(out5)
        out6 = self.block6(out6)
        out6 = self.pool(out6)

        d_out5 = self.deconv6(out6)
        d_out5.features = torch.cat((d_out5.features, out5.features), dim=1)
        d_out5 = self.deblock5(d_out5)

        d_out4 = self.deconv5(d_out5)
        d_out4.features = torch.cat((d_out4.features, out4.features), dim=1)
        d_out4 = self.deblock4(d_out4)

        d_out3 = self.deconv4(d_out4)
        d_out3.features = torch.cat((d_out3.features, out3.features), dim=1)
        d_out3 = self.deblock3(d_out3)

        d_out2 = self.deconv3(d_out3)
        d_out2.features = torch.cat((d_out2.features, out2.features), dim=1)
        d_out2 = self.deblock2(d_out2)

        d_out1 = self.deconv2(d_out2)
        d_out1.features = torch.cat((d_out1.features, out1.features), dim=1)
        d_out1 = self.deblock1(d_out1)

        d_out0 = self.deconv1(d_out1)
        d_out0.features = torch.cat((d_out0.features, out0.features), dim=1)
        d_out0 = self.deblock0(d_out0)

        return d_out0
