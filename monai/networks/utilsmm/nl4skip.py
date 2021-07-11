import torch
from torch import nn
import torch.nn.functional as F
print_tensor = lambda n, x: print(n, type(x), x.shape, x.min(), x.max())

class NonLocalGeneric(nn.Module):
    """
    a generic class for non-local operations
    can easily be used for self-attention or cross-attention

    theta, phi and g are three operations and can accept two types of inputs,
    one for query and one for key/reference
    in self-attention fashion, the query and reference input are both from the same input feature maps
    in cross-reference fashion, two inputs can originate from difference sources. 
    in the skip-connection module of Unet, the lower level features from encoder and the high level ones from decoder
    can assume as either inputs. 
    """

    def __init__(self, in_channels, skip_in_channels = None, key_channels=None, 
                       dimension=2, sub_sample=False, bn_layer=True, psp_size = None):

        # theta, phi and g are three operations and can accept two types of inputs,
        # one for query and one for key/reference
        # in self-attention fashion, the query and reference input are both from the same input feature maps
        # in cross-reference fashion, two inputs can originate from difference sources. 
        # in the skip-connection module of Unet, the lower level features from encoder and the high level ones from decoder
        # can assume as either inputs. 
        super().__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.skip_in_channels = skip_in_channels
        self.key_channels = key_channels
        self.psp_size = psp_size

        if self.key_channels is None:
            self.key_channels = in_channels // 2 # 1024 to 512 to reduce computation
            if self.key_channels == 0:
                self.key_channels = 1

        ops = OpsDim(dimension)
        # in_channels = [head_channels] + list(decoder_channels[:-1]) # 2048 256 128 64 32
        # skip_channels = list(encoder_channels[1:]) + [0] # 1024, 512, 256, 64, 0
        self.value_channels = self.skip_in_channels if skip_in_channels is not None else self.in_channels
        self.value_half_chn = self.value_channels//2

        print('NLblock: ip, skip chn %s %s' %(self.in_channels, self.value_channels))
        self.g4value = ops.ConvND1k1s(self.value_channels, self.value_half_chn) # embedding, features to aggregate

        if bn_layer: # transformation before fusion with original input
            self.W = ops.ConvBN1k1s(self.value_half_chn, self.in_channels)
        else:
            self.W =  ops.ConvND1k1s(self.value_half_chn, self.in_channels)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        if psp_size:
            self.theta4query = ops.ConvBN1k1s(self.in_channels, self.key_channels)
            self.phi4key = ops.ConvBN1k1s(self.in_channels, self.key_channels)

        else:
            self.theta4query = ops.ConvND1k1s(self.in_channels, self.key_channels)
            self.phi4key = ops.ConvND1k1s(self.in_channels, self.key_channels)                       

        # n = [print(type(a), a) for a in self.theta4query]

        if psp_size:
            self.psp = PSPModule(psp_size)  # B C N
        if sub_sample:
            self.g4value = nn.Sequential(self.g4value, ops.max_pool_layer)
            self.phi4key = nn.Sequential(self.phi4key, ops.max_pool_layer) # g and phi need to match for non-channel dimension

    def forward(self, x, skip = None):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)
        # N = t*h*w

        value_feature = skip if skip is not None else x

        # B, N, Cm, no downsample 
        query = self.theta4query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)

        # assert  query x key should come from different sources
        
        if self.psp_size:
            key = self.psp(self.phi4key(x))
            value = self.psp(self.g4value(value_feature))
        else:
            key = self.phi4key(x).view(batch_size, self.key_channels, -1)   
            value = self.g4value(value_feature).view(batch_size, self.value_half_chn, -1) # B, Cm, Ng  
        value = value.permute(0, 2, 1) # B, Ng, Cm, possible downsample

        # possible downsample
        # B, Cm, Np
        sim_map = torch.matmul(query, key) # BNCm, BCmNp  
        # sim_map = (self.key_channels ** -.5) * sim_map
        sim_map_div_C = F.softmax(sim_map, dim=-1) #BNNp, 

        y = torch.matmul(sim_map_div_C, value) # BNNp X BNgCm = BNCm, Np == Ng. 
        y = y.permute(0, 2, 1).contiguous() 
        y = y.view(batch_size, self.value_half_chn, *x.size()[2:])
        
        context = self.W(y)
        # z = context + x # 可以修改成concat

        return context


class NL4Skip_CR(nn.Module):
    def __init__(self, in_channels, skip_in_channels = None, key_channels=None, 
                       dimension=2, sub_sample=False, bn_layer=True, psp_size = None):

        # theta, phi and g are three operations and can accept two types of inputs,
        # one for query and one for key/reference
        # in self-attention fashion, the query and reference input are both from the same input feature maps
        # in cross-reference fashion, two inputs can originate from difference sources. 
        # in the skip-connection module of Unet, the lower level features from encoder and the high level ones from decoder
        # can assume as either inputs. 
        super().__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.skip_in_channels = skip_in_channels
        self.key_channels = key_channels
        self.psp_size = psp_size

        if self.key_channels is None:
            self.key_channels = in_channels // 2 # 1024 to 512 to reduce computation
            if self.key_channels == 0:
                self.key_channels = 1

        ops = OpsDim(dimension)
        # in_channels = [head_channels] + list(decoder_channels[:-1]) # 2048 256 128 64 32
        # skip_channels = list(encoder_channels[1:]) + [0] # 1024, 512, 256, 64, 0
        self.value_channels = self.skip_in_channels if skip_in_channels is not None else self.in_channels
        self.value_half_chn = self.value_channels//2

        print('NLblock: ip, skip chn %s %s' %(self.in_channels, self.value_channels))
        self.g4value = ops.ConvND1k1s(self.value_channels, self.value_half_chn) # embedding, features to aggregate

        if bn_layer: # transformation before fusion with original input
            self.W = ops.ConvBN1k1s(self.value_half_chn, self.in_channels)
        else:
            self.W =  ops.ConvND1k1s(self.value_half_chn, self.in_channels)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        if psp_size:
            self.theta4query = ops.ConvBN1k1s(self.in_channels, self.key_channels)
            self.phi4key = ops.ConvBN1k1s(self.in_channels, self.key_channels)
        else:
            self.theta4query = ops.ConvND1k1s(self.in_channels, self.key_channels)
            self.phi4key = ops.ConvND1k1s(self.skip_in_channels, self.key_channels)                       

        # n = [print(type(a), a) for a in self.theta4query]
        if psp_size:
            self.psp = PSPModule(psp_size)  # B C N
        if sub_sample:
            self.g4value = nn.Sequential(self.g4value, ops.max_pool_layer)
            self.phi4key = nn.Sequential(self.phi4key, ops.max_pool_layer) # g and phi need to match for non-channel dimension

    def forward(self, x, skip = None):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)
        # N = t*h*w

        value_feature = skip if skip is not None else x

        # B, N, Cm, no downsample 
        query = self.theta4query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)

        # assert 
        if self.psp_size:
            key = self.psp(self.phi4key(value_feature))
            value = self.psp(self.g4value(value_feature))
        else:
            key = self.phi4key(value_feature).view(batch_size, self.key_channels, -1)   
            value = self.g4value(value_feature).view(batch_size, self.value_half_chn, -1) # B, Cm, Ng  
        value = value.permute(0, 2, 1) # B, Ng, Cm, possible downsample

        # possible downsample
        # B, Cm, Np
        sim_map = torch.matmul(query, key) # BNCm, BCmNp  
        # sim_map = (self.key_channels ** -.5) * sim_map
        sim_map_div_C = F.softmax(sim_map, dim=-1) #BNNp, 

        y = torch.matmul(sim_map_div_C, value) # BNNp X BNgCm = BNCm, Np == Ng. 
        y = y.permute(0, 2, 1).contiguous() 
        y = y.view(batch_size, self.value_half_chn, *x.size()[2:])
        
        context = self.W(y)
        # z = context + x # 可以修改成concat

        return context

class OpsDim(object):

    def __init__(self, dimension =2 ):
        self.dimension = dimension

    @property
    def conv_nd(self):
        if self.dimension == 3:
            return nn.Conv3d
        elif self.dimension ==2:
            return nn.Conv2d
        else:
            return nn.Conv1d

    @property
    def max_pool_layer(self):
        if self.dimension == 3:
            return nn.MaxPool3d(kernel_size=(1, 2, 2))
        elif self.dimension ==2:
            return nn.MaxPool2d(kernel_size=(2, 2))
        else:
            return nn.MaxPool1d(kernel_size=(2))

    @property
    def bn(self):
        if self.dimension == 3:
            return nn.BatchNorm3d
        elif self.dimension ==2:
            return nn.BatchNorm2d
        else:
            return nn.BatchNorm1d
    

    def ConvBN1k1s(self, in_channels, key_channels):
        f = nn.Sequential(self.conv_nd(in_channels=in_channels, out_channels=key_channels,
                          kernel_size=1, stride=1, padding=0),
                        self.bn(key_channels), nn.ReLU(inplace=True)
                        )# 2nd component of the relation matrix


        nn.init.constant_(f[1].weight, 0)
        nn.init.constant_(f[1].bias, 0)
        
        return f
    
    def ConvND1k1s(self, in_channels, key_channels):
        return self.conv_nd(in_channels=in_channels, out_channels=key_channels,
                kernel_size=1, stride=1, padding=0)



class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, *_ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center

