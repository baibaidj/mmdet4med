import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, normal_init
from mmcv.cnn import ConvModule

print_tensor = lambda n, x: print(n, type(x), x.shape, x.min(), x.max())


class AsyNonLocal2D(nn.Module):
    """Non-local module.

    See https://arxiv.org/abs/1711.07971 for details of non-local 2D
    See https://arxiv.org/pdf/1908.07678.pdf for details of asymetric non-local 
    

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        use_scale (bool): Whether to scale pairwise_weight by 1/inter_channels.
        conv_cfg (dict): The config dict for convolution layers.
            (only applicable to conv_out)
        norm_cfg (dict): The config dict for normalization layers.
            (only applicable to conv_out)
        mode (str): Options are `embedded_gaussian` and `dot_product`.

        self.g is used for the reference input
        self.theta is used for the querry input
        self.phi is also used for the reference input
        cross_attention is conducted by: dot_product(self.theta, self.phi) * self.g
    """

    def __init__(self,
                 in_channels,
                 refer_in_channels,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian'):
        super(AsyNonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.refer_in_channels = refer_in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']

        # g, theta, phi are actually `nn.Conv2d`. Here we use ConvModule for
        # potential usage.
        self.g = ConvModule(
            self.refer_in_channels,
            self.inter_channels,
            kernel_size=1,
            act_cfg=None)
        self.theta = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            act_cfg=None)
        self.phi = ConvModule(
            self.refer_in_channels,
            self.inter_channels,
            kernel_size=1,
            act_cfg=None)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.init_weights()

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.g, self.theta, self.phi]:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
        else:
            normal_init(self.conv_out.conv, std=std)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, querry, reference):
        ##forward by: dot_product(self.theta(q), self.phi(ref)) * self.g(ref)
        rn, _, rh, rw = querry.shape
        qn, _, qh, qw = reference.shape

        # g_x: [N, DxH'xW', C] for reference 
        # reference in N C DH' W'; g(reference) in N C' DH' W';
        g_x = self.g(reference).view(rn, self.inter_channels, -1) # gx in N C' DH'W'
        g_x = g_x.permute(0, 2, 1) #gx in N DH'W' C'

        # theta_x: [N, HxW, C] for querry
        theta_x = self.theta(querry).view(qn, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # phi_x: [N, C, DH'xW'] for reference
        phi_x = self.phi(reference).view(rn, self.inter_channels, -1) # phi_x in (N C' DH'W')

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, DH'xW']
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).reshape(rn, self.inter_channels, rh, rw)

        output = querry + self.conv_out(y)

        return output




class NonLocal4Point(nn.Module):
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

    def __init__(self, query_in_channels, key_in_channels = None, interm_channels=None, 
                      out_channels = None,  query_dim = 2, key_dim = 1, 
                      sub_sample=False, bn_layer=True, psp_size = None,
                      verbose = False,
                      ):

        # theta, phi and g are three operations and can accept two types of inputs,
        # one for query and one for key/reference
        # in self-attention fashion, the query and reference input are both from the same input feature maps
        # in cross-reference fashion, two inputs can originate from difference sources. 
        # in the skip-connection module of Unet, the lower level features from encoder and the high level ones from decoder
        # can assume as either inputs. 
        super().__init__()

        assert query_dim in [1, 2, 3] and key_dim in [1, 2, 3]

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.sub_sample = sub_sample

        self.query_in_channels = query_in_channels
        self.key_in_channels = key_in_channels
        self.interm_channels = self.query_in_channels // 2 if interm_channels is None else interm_channels
        self.out_channels = self.query_in_channels if out_channels is None else out_channels
        self.psp_size = psp_size
        self.verbose = verbose

        query_ops = OpsDim(self.query_dim)
        key_ops = OpsDim(self.key_dim)
        # in_channels = [head_channels] + list(decoder_channels[:-1]) # 2048 256 128 64 32
        # skip_channels = list(encoder_channels[1:]) + [0] # 1024, 512, 256, 64, 0

        print('NLblock: ip, skip chn %s %s' %(self.query_in_channels, self.key_in_channels))
        self.g4value = key_ops.ConvND1k1s(self.key_in_channels, self.interm_channels) # embedding, features to aggregate

        if bn_layer: # transformation before fusion with original input
            self.W = query_ops.ConvBN1k1s(self.interm_channels, self.out_channels)
        else:
            self.W =  query_ops.ConvND1k1s(self.interm_channels, self.out_channels)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        if psp_size:
            self.theta4query = query_ops.ConvBN1k1s(self.query_in_channels, self.interm_channels)
            self.phi4key = key_ops.ConvBN1k1s(self.key_in_channels, self.interm_channels)

        else:
            self.theta4query = query_ops.ConvND1k1s(self.query_in_channels, self.interm_channels)
            self.phi4key = key_ops.ConvND1k1s(self.key_in_channels, self.interm_channels)                       

        # n = [print(type(a), a) for a in self.theta4query]

        if psp_size:
            self.psp = PSPModule(psp_size, dimension= self.key_dim)  # B C N
        if sub_sample:
            self.g4value = nn.Sequential(self.g4value, key_ops.max_pool_layer)
            self.phi4key = nn.Sequential(self.phi4key, key_ops.max_pool_layer) # g and phi need to match for non-channel dimension

    def forward(self, query_feats, key_feats):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        if self.verbose: print('\t\tNon-local block')
        if self.verbose: print_tensor('\t\tquery', query_feats); print_tensor('\t\tkey', key_feats)
        batch_size = query_feats.size(0)
        # N = t*h*w

        value_feature = key_feats #if key_feats is not None else query_feats
        # B, C, N -> B, N, Cm, no downsample 
        query = self.theta4query(query_feats).view(batch_size, self.interm_channels, -1)
        query = query.permute(0, 2, 1)
        if self.verbose: print_tensor('\t\tembed query', query)
        # assert 
        if self.psp_size:
            key = self.psp(self.phi4key(key_feats))
            value = self.psp(self.g4value(value_feature))
        else:
            key = self.phi4key(key_feats).view(batch_size, self.interm_channels, -1)   
            value = self.g4value(value_feature).view(batch_size, self.interm_channels, -1) # B, Cm, Ng  
        value = value.permute(0, 2, 1) # B, Ng, Cm, possible downsample
        if self.verbose: print_tensor('\t\tembed key', key); print_tensor('\t\tembed value', value)
        # possible downsample
        # B, Cm, Np
        sim_map = torch.matmul(query, key) # BNCm, BCmNp  
        # sim_map = (self.key_channels ** -.5) * sim_map
        sim_map_div_C = F.softmax(sim_map, dim=-1) #BNNp, 
        if self.verbose: print_tensor('\t\taffinity map', sim_map); print_tensor('\t\tnorm affinity map', sim_map_div_C)
        y = torch.matmul(sim_map_div_C, value) # BNNp X BNgCm = BNCm, Np == Ng. 
        y = y.permute(0, 2, 1).contiguous() 
        y = y.view(batch_size, self.interm_channels, *query_feats.size()[2:])
        
        context = self.W(y)
        # print_tensor('context', context)q
        # z = context + x # 可以修改成concat

        return context



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
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center



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

        normal_init(f[0])
        nn.init.constant_(f[1].weight, 0)
        nn.init.constant_(f[1].bias, 0)
        
        return f
    
    def ConvND1k1s(self, in_channels, key_channels):
        f = self.conv_nd(in_channels=in_channels, out_channels=key_channels,
                kernel_size=1, stride=1, padding=0)
        normal_init(f)
        return f