import torch
from . import networks as N
from .cbam import CBAM
# import networks as N
# from cbam import CBAM
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .AWISP_utils import DWT, IWT
from .AWISP_modules import shortcutblock, GCIWTResUp, GCWTResDown, GCRDB, ContextBlock2d, SE_net, PSPModule, last_upsample

# from AWISP_utils import DWT, IWT
# from AWISP_modules import shortcutblock, GCIWTResUp, GCWTResDown, GCRDB, ContextBlock2d, SE_net, PSPModule, last_upsample
import functools

import torch.nn as nn


def color_block(in_filters, out_filters, normalization=False):
	conv = nn.Conv2d(in_filters, out_filters, 1, stride=1, padding=0)
	pooling = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=True)
	act = nn.LeakyReLU(0.2)
	layers = [conv, pooling, act]
	if normalization:
		layers.append(nn.InstanceNorm2d(out_filters, affine=True))
	return layers


class Color_Condition(nn.Module):
	def __init__(self, in_channels=4, cond_c=32):
		super(Color_Condition, self).__init__()
		cond_nf = 32
		self.cond_first = nn.Sequential(nn.Conv2d(in_channels, cond_nf, 3, 1, 1), nn.LeakyReLU(0.1, True),
	  							nn.Conv2d(cond_nf, cond_nf, 2, 2), nn.LeakyReLU(0.1, True), 
								nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True), 
								nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True))

		# global modulation
		self.global_modulation = nn.Sequential(
			# *color_block(32, 32, normalization=True),
			*color_block(cond_nf, cond_nf, normalization=True),
			*color_block(cond_nf, cond_nf*2, normalization=True),
			*color_block(cond_nf*2, cond_nf*4, normalization=True),
			*color_block(cond_nf*4, cond_nf*4),
			nn.Dropout(p=0.5),
			nn.Conv2d(cond_nf*4, cond_c, 1, stride=1, padding=0),
			nn.AdaptiveAvgPool2d(1),
		)
		# local modulation
		self.CondNet1 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 1), 
								nn.LeakyReLU(0.1, True), 
								nn.Conv2d(cond_nf, cond_c, 1)) # H W
		self.CondNet2 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), 
								nn.LeakyReLU(0.1, True), 
								nn.Conv2d(cond_nf, cond_c, 1)) # H/2 W/2
		self.CondNet3 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), 
								nn.LeakyReLU(0.1, True), 
								nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), 
								nn.LeakyReLU(0.1, True), 
								nn.Conv2d(cond_nf, cond_c, 1)) # H/4 W/4
		self.CondNet4 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), 
								nn.LeakyReLU(0.1, True), 
								nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), 
								nn.LeakyReLU(0.1, True), 
								nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), 
								nn.LeakyReLU(0.1, True), 
								nn.Conv2d(cond_nf, cond_c, 1)) # H/8 W/8
  
	def forward(self, x):
	 
		x = self.cond_first(x)
		gfm_vector = self.global_modulation(x)
		lfm_feature1 = self.CondNet1(x)
		lfm_feature2 = self.CondNet2(x)
		lfm_feature3 = self.CondNet3(x)
		lfm_feature4 = self.CondNet4(x)
	 
		return gfm_vector, [lfm_feature1, lfm_feature2, lfm_feature3, lfm_feature4]

def pad_to_multiple_of_16(x):
	_, _, h, w = x.size()

	# 计算每个维度上需要添加的padding大小
	pad_h = (16 - h % 16) % 16
	pad_w = (16 - w % 16) % 16

	# 为了实现均匀的padding，分别在左右和上下均匀分配padding
	# pad_top = pad_h // 2
	# pad_bottom = pad_h - pad_top
	# pad_left = pad_w // 2
	# pad_right = pad_w - pad_left
	pad_top = 0
	pad_bottom = pad_h
	pad_left = 0 
	pad_right = pad_w

	# 应用padding
	padding = (pad_left, pad_right, pad_top, pad_bottom)
	x_padded = F.pad(x, padding, "constant", 0)

	return x_padded, (h, w)


def remove_padding(x_padded, original_size):
	"""
	去除图像的填充部分。

	:param x_padded: 填充后的图像张量。
	:param original_size: 原始图像的尺寸（高度，宽度）。
	:return: 去除填充后的图像张量。
	"""
	orig_h, orig_w = original_size
	padded_h, padded_w = x_padded.size()[-2:]

	# 计算目标尺寸（原始尺寸的两倍）
	target_h = orig_h * 2
	target_w = orig_w * 2

	# 确保目标尺寸不大于填充后的尺寸
	target_h = min(target_h, padded_h)
	target_w = min(target_w, padded_w)

	# 裁剪掉填充部分
	return x_padded[:,:, :target_h, :target_w]

class PreCoord(nn.Module):
	def __init__(self, pre_train=True):
		super(PreCoord, self).__init__()

		self.ch_1 = 64

		self.down = N.seq(
			N.conv(4, self.ch_1, 3, stride=2, padding=0, mode='CR'),
			N.conv(self.ch_1, self.ch_1, kernel_size=3, stride=2, padding=0, mode='CR'),
			N.conv(self.ch_1, self.ch_1, kernel_size=3, stride=2, padding=0, mode='CR'),
			N.conv(self.ch_1, self.ch_1, kernel_size=3, stride=2, padding=0, mode='CR'),
		)

		self.fc = N.seq(
			nn.Linear(self.ch_1*13*13, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, 2)
		)
		
		if pre_train:
			self.load_state_dict(torch.load('./ckpt/coord.pth')['state_dict'])

	def forward(self, raw):
		N, C, H, W = raw.size()
		input = raw
		if H != 224 or W != 224:
			input = F.interpolate(input, size=[224, 224], mode='bilinear', align_corners=True)
		
		down = self.down(input)
		down = down.view(N, self.ch_1*13*13)
		out = self.fc(down)
		
		return out


class ConditionNet(nn.Module):
	def __init__(self, nf=64, classifier='color_condition', cond_c=3):
		super(ConditionNet, self).__init__()

		if classifier=='color_condition':
			self.classifier = Color_Condition(out_c=cond_c)

		self.GFM_nf = 64

		self.cond_scale_first = nn.Linear(cond_c, nf)
		self.cond_scale_HR = nn.Linear(cond_c, nf)
		self.cond_scale_last = nn.Linear(cond_c, 3)

		self.cond_shift_first = nn.Linear(cond_c, nf)
		self.cond_shift_HR = nn.Linear(cond_c, nf)
		self.cond_shift_last = nn.Linear(cond_c, 3)

		self.conv_first = nn.Conv2d(3, nf, 1, 1)
		self.HRconv = nn.Conv2d(nf, nf, 1, 1)
		self.conv_last = nn.Conv2d(nf, 3, 1, 1)
		self.act = nn.ReLU(inplace=True)

	def forward(self, x):
		content = x[0]
		condition = x[1]
		fea = self.classifier(condition).squeeze(2).squeeze(2)

		scale_first = self.cond_scale_first(fea)
		shift_first = self.cond_shift_first(fea)

		scale_HR = self.cond_scale_HR(fea)
		shift_HR = self.cond_shift_HR(fea)

		scale_last = self.cond_scale_last(fea)
		shift_last = self.cond_shift_last(fea)

		out = self.conv_first(content)
		out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
		out = self.act(out)

		out = self.HRconv(out)
		out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
		out = self.act(out)

		out = self.conv_last(out)
		out = out * scale_last.view(-1, 3, 1, 1) + shift_last.view(-1, 3, 1, 1) + out

		return out


class CB(nn.Module):
	def __init__(self, in_filters, out_filters,normalization=False):
		super(CB, self).__init__()
		self.conv=nn.Conv2d(in_filters, out_filters, 1, stride=1, padding=0)
		self.pooling = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=True)
		self.act = nn.LeakyReLU(0.2)
		self.normalization=normalization
		if normalization:
			self.norm=nn.InstanceNorm2d(out_filters, affine=True)
	def forward(self,x):
		x=self.conv(x)
		x=self.pooling(x)
		x=self.act(x)
		if self.normalization:
			x=self.norm(x)
		return x

import torch.nn.init as init
def initialize_weights(net_l, scale=1):
	if not isinstance(net_l, list):
		net_l = [net_l]
	for net in net_l:
		for m in net.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight, a=0, mode='fan_in')
				m.weight.data *= scale  # for residual block
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				init.kaiming_normal_(m.weight, a=0, mode='fan_in')
				m.weight.data *= scale
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				init.constant_(m.weight, 1)
				init.constant_(m.bias.data, 0.0)


class Color_ConditionUnet(nn.Module):
	def __init__(self, in_channels=3, out_c=6):
		super(Color_ConditionUnet, self).__init__()
		self.downblocks=nn.ModuleList()
		self.downblocks.append(CB(3, 16, normalization=True)) # /2
		self.downblocks.append(CB(16, 32, normalization=True)) # /4
		self.downblocks.append(CB(32, 64, normalization=True)) # /8
		self.downblocks.append(CB(64, 128, normalization=True)) # /16
		self.downblocks.append(CB(128, 128, normalization=False))# /32
		self.global_vector=nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Conv2d(128, out_c, 1, stride=1, padding=0),
			nn.AdaptiveAvgPool2d(1))
		self.drop=nn.Dropout(p=0.5)

		self.conv2 = nn.Sequential(
			nn.Conv2d(128,out_c,1),
			nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=True),
			nn.LeakyReLU(0.2)        
			)

	def forward(self,o):
		x=o
		# downfeats=[]
		for fe in self.downblocks:
			x=fe(x)
			# downfeats.append(x)
		vector=self.global_vector(x)
		feat1=F.upsample(x,(o.shape[2],o.shape[3])) 
		feat1=self.drop(feat1)
		feat1=self.conv2(feat1)   # 128
		
		# feat4 = self.upsample1(self.conv1(feat16))+downfeats[-4]  # 32
		# feat4=self.drop(feat4)
		# feat1 = self.conv2(self.upsample2(feat4))
		if False:
			x=x.cpu()
			torch.cuda.empty_cache()
		return vector.squeeze(2).squeeze(2),feat1

class SFTLayer(nn.Module):
	def __init__(self, cond_c=32, out_nc=64, nf=32):
		super(SFTLayer, self).__init__()
		self.SFT_scale_conv0 = nn.Conv2d(cond_c, nf, 1)
		self.SFT_scale_conv1 = nn.Conv2d(nf, out_nc, 1)
		self.SFT_shift_conv0 = nn.Conv2d(cond_c, nf, 1)
		self.SFT_shift_conv1 = nn.Conv2d(nf, out_nc, 1)

	def forward(self, x):
		# x[0]: fea; x[1]: cond
		scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
		shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
		return x[0] * (scale + 1) + shift


class GFMLayer(nn.Module):
	def __init__(self, cond_c=32, out_nc=64, nf=32):
		super(GFMLayer, self).__init__()
		self.GFM_scale_conv0 = nn.Linear(cond_c, nf)
		self.GFM_scale_conv1 = nn.Linear(nf, out_nc)
		self.GFM_shift_conv0 = nn.Linear(cond_c, nf)
		self.GFM_shift_conv1 = nn.Linear(nf, out_nc)
		self.out_nc = out_nc
	def forward(self, x):
		
		scale = self.GFM_scale_conv1(F.leaky_relu(self.GFM_scale_conv0(x[1]), 0.1, inplace=True))
		shift = self.GFM_shift_conv1(F.leaky_relu(self.GFM_shift_conv0(x[1]), 0.1, inplace=True))
		out = x[0] * scale.view(-1, self.out_nc, 1, 1) + shift.view(-1, self.out_nc, 1, 1) + x[0]
		return out




class ResBlock_with_modulation(nn.Module):
	def __init__(self, nf=64, cond_c=32):
		super(ResBlock_with_modulation, self).__init__()
		self.gfm = GFMLayer(cond_c=cond_c, out_nc=nf, nf=nf)
		self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
		self.sft = SFTLayer(cond_c=cond_c, out_nc=nf, nf=nf)
		self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

		# initialization
		initialize_weights([self.conv1, self.conv2], 0.1)

	def forward(self, x):
		# x[0]: fea; x[1]: vector; x[2]: fea
		fea = self.gfm((x[0], x[1]))
		fea = F.relu(self.conv1(fea), inplace=True)
		fea = self.sft((fea, x[2]))
		fea = self.conv2(fea)
		return (x[0] + fea, x[1], x[2])

class Color_Condition_GFM(nn.Module):
	def __init__(self, in_channels=4, out_c=32):
		super(Color_Condition_GFM, self).__init__()

		self.model = nn.Sequential(
			*color_block(in_channels, 16, normalization=True),
			*color_block(16, 32, normalization=True),
			*color_block(32, 64, normalization=True),
			*color_block(64, 128, normalization=True),
			*color_block(128, 128),
			nn.Dropout(p=0.5),
			nn.Conv2d(128, out_c, 1, stride=1, padding=0),
			nn.AdaptiveAvgPool2d(1),
		)

	def forward(self, img_input):
		return self.model(img_input)

class Lens_Shading_Correction(nn.Module):
	def __init__(self, in_channels=2, out_c=32, nf=32):
		super(Lens_Shading_Correction, self).__init__()

		self.model = nn.Sequential(
			nn.Conv2d(in_channels, nf, 1, 1),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.Conv2d(nf, nf, 1, 1),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.Conv2d(nf, nf, 1, 1),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.Conv2d(nf, out_c, 1, 1),
		)

	def forward(self, img_input):
		return self.model(img_input)


class HyCondModConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, act='relu'):
		super().__init__()

		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
		if act == 'lrelu':
			self.act = nn.LeakyReLU(0.2, inplace=True)
		elif act == 'prelu':
			self.act = nn.PReLU()
		else:
			self.act = nn.ReLU(inplace=True)
		

	def forward(self, x):
		return self.act(self.conv(x))

class HyCondModEncBlock(nn.Module):
	"""
		input: (N, in_channels, H, W)
		output: (N, out_channels, H / 2, W / 2)
	"""
	def __init__(self, in_channels, out_channels, downscale_method='stride'):
		super().__init__()

		if downscale_method == 'stride':
			self.down = HyCondModConvBlock(in_channels, out_channels, stride=2)
		elif downscale_method == 'pool':
			self.down = nn.Sequential(
				nn.MaxPool2d(2),
				HyCondModConvBlock(in_channels, out_channels)
			)
		else:
			raise NotImplementedError

		self.conv = HyCondModConvBlock(out_channels, out_channels)

	def forward(self, x):
		return self.conv(self.down(x))

class HyCondModDecBlock(nn.Module):
	"""
		input: (N, in_channels, H, W)
		output: (N, out_channels, 2 * H, 2 * W)
	"""
	def __init__(self, in_channels, out_channels, upscale_method='bilinear'):
		super().__init__()

		if upscale_method == 'bilinear':
			self.up = nn.Sequential(
				nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
				HyCondModConvBlock(in_channels, out_channels)
			)
		elif upscale_method == 'transpose':
			self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
		else:
			raise NotImplementedError

		self.conv = HyCondModConvBlock(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		# input is CHW
		# diffY = x2.size()[2] - x1.size()[2]
		# diffX = x2.size()[3] - x1.size()[3]

		# x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
		# 				diffY // 2, diffY - diffY // 2])
		# # if you have padding issues, see
		# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
		# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
		x = torch.cat([x2, x1], dim=1)

		return self.conv(x)

class HyCondModGlobalConditionBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.cond = nn.Sequential(
			HyCondModConvBlock(in_channels, out_channels, kernel_size=1, padding=0),
			nn.AdaptiveAvgPool2d(1)
		)

	def forward(self, x):
		return self.cond(x)

class LFMConditionModule(nn.Module):
	def __init__(self, in_channels, out_channels, init_mid_channels=16,
				 down_method='stride', up_method='bilinear', GFM_channels=32):
		super().__init__()

		self.in_conv = HyCondModConvBlock(in_channels, init_mid_channels)                           # in_channels -> 16
		self.enc_1 = HyCondModEncBlock(init_mid_channels, init_mid_channels * 2, down_method)       # 16 -> 32  1/2
		self.enc_2 = HyCondModEncBlock(init_mid_channels*2, init_mid_channels * 4, down_method)   # 32 -> 64  1/4
		self.enc_3 = HyCondModEncBlock(init_mid_channels*4, init_mid_channels * 8, down_method)   # 64 -> 128  1/8
		# self.global_cond = HyCondModGlobalConditionBlock(init_mid_channels * 8, global_cond_channels)  # 128 -> 64
		self.dec_1 = HyCondModDecBlock(init_mid_channels*8, init_mid_channels*4, up_method)     # 128 -> 64  1/4
		# self.gfm1 = GFMLayer(cond_c=GFM_channels, out_nc=init_mid_channels*4, nf=init_mid_channels*4)
		self.dec_2 = HyCondModDecBlock(init_mid_channels*4, init_mid_channels*2, up_method)     # 64 -> 32  1/2
		# self.gfm2 = GFMLayer(cond_c=GFM_channels, out_nc=init_mid_channels*2, nf=init_mid_channels*4)
		self.dec_3 = HyCondModDecBlock(init_mid_channels*2, init_mid_channels, up_method)         # 32 -> 16  1
		# self.gfm3 = GFMLayer(cond_c=GFM_channels, out_nc=init_mid_channels, nf=init_mid_channels*4)
		self.out_conv = HyCondModConvBlock(init_mid_channels, out_channels)                         # 16 -> out_channels

	def forward(self, x):
		x_1 = self.in_conv(x)       # 16
		x_2 = self.enc_1(x_1)       # 32
		x_3 = self.enc_2(x_2)       # 64
		x_4 = self.enc_3(x_3)       # 128
		y = self.dec_1(x_4, x_3)    # 64
		# y = self.gfm1((y, global_vector))
		y = self.dec_2(y, x_2)      # 32
		# y = self.gfm2((y, global_vector))
		y = self.dec_3(y, x_1)      # 16
		# y = self.gfm3((y, global_vector))
		y = self.out_conv(y)        # out_channels

		return y


class Color_Condition_GFM_LFM(nn.Module):
	def __init__(self, in_channels=4, GFM_out_c=32, LFM_out_c=32):
		super(Color_Condition_GFM_LFM, self).__init__()
		self.downblocks=nn.ModuleList()
		self.downblocks.append(CB(in_channels, 16, normalization=True)) # /2
		self.downblocks.append(CB(16, 32, normalization=True)) # /4
		self.downblocks.append(CB(32, 64, normalization=True)) # /8
		self.downblocks.append(CB(64, 128, normalization=True)) # /16
		self.downblocks.append(CB(128, 256, normalization=True))# /32
		self.downblocks.append(CB(256, 384, normalization=False))# /64
		self.global_vector=nn.Sequential(
			nn.Dropout(p=0.8),
			nn.Conv2d(384, GFM_out_c, 1, stride=1, padding=0),
			nn.AdaptiveAvgPool2d(1))

  
		self.cond_first = nn.Sequential(nn.Conv2d(in_channels, LFM_out_c, 3, 1, 1), nn.LeakyReLU(0.1, True),
						# nn.Conv2d(cond_nf, cond_nf, 2, 2), nn.LeakyReLU(0.1, True), 
						nn.Conv2d(LFM_out_c, LFM_out_c, 1), nn.LeakyReLU(0.1, True), 
						nn.Conv2d(LFM_out_c, LFM_out_c, 1), nn.LeakyReLU(0.1, True))
		# self.LFM_module = LFMConditionModule(in_channels=in_channels, out_channels=LFM_out_c, GFM_channels=GFM_out_c)
		self.cond_first = nn.Sequential(nn.Conv2d(in_channels, LFM_out_c, 3, 1, 1))

		
	def forward(self, global_raw, local_patch):
		# global_raw: 1,4,4000,6000
		# local_patch: 1,4,256,256
		x=global_raw
		for fe in self.downblocks:
			x=fe(x)
		vector=self.global_vector(x)
		# lfm_fea是原始local_patch分辨率
		lfm_fea = self.cond_first(local_patch)
		return vector, lfm_fea


class Res_GFM(nn.Module):
	def __init__(self, in_nc=32, chan=32, cond_c=32, out_nc=32, nf=64):
	#  out_nc需要跟chan一致
		super(Res_GFM, self).__init__()
		self.conv0 = nn.Conv2d(in_nc, chan, 3, 1, 1)
		self.conv1 = nn.Conv2d(chan, chan, 3, 1, 1)
		self.GFM_scale_conv0 = nn.Linear(cond_c, nf)
		self.GFM_scale_conv1 = nn.Linear(nf, chan)
		self.GFM_shift_conv0 = nn.Linear(cond_c, nf)
		self.GFM_shift_conv1 = nn.Linear(nf, chan)
		self.out_nc = chan
		self.act = nn.LeakyReLU(inplace=True)
	def forward(self, x):
		# x[0]: input_img
		# x[1]: cond_img
		
		fea = self.conv0(x[0])
		scale = self.GFM_scale_conv1(F.leaky_relu(self.GFM_scale_conv0(x[1]), 0.1, inplace=True))
		shift = self.GFM_shift_conv1(F.leaky_relu(self.GFM_shift_conv0(x[1]), 0.1, inplace=True))
		fea = fea * scale.view(-1, self.out_nc, 1, 1) + shift.view(-1, self.out_nc, 1, 1) + fea
		fea = self.act(fea)
		fea = self.conv1(fea) + x[0]
		return fea, x[1]


class SpatialFeatureTransform(nn.Module):
	def __init__(self, cond_channels, n_features, ada_method='vanilla', residual=True):
		super().__init__()
		if ada_method == 'vanilla':
			self.cond_scale = nn.Sequential(
				nn.Conv2d(cond_channels, n_features, 3, stride=1, padding=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(n_features, n_features, 3, stride=1, padding=1)
			)
			self.cond_shift = nn.Sequential(
				nn.Conv2d(cond_channels, n_features, 3, stride=1, padding=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(n_features, n_features, 3, stride=1, padding=1)
			)
		elif ada_method == 'cbam':
			self.cond_scale = nn.Sequential(
				nn.Conv2d(cond_channels, n_features, 1),
				nn.ReLU(inplace=True),
				CBAM(n_features)
			)
			self.cond_shift = nn.Sequential(
				nn.Conv2d(cond_channels, n_features, 1),
				nn.ReLU(inplace=True),
				CBAM(n_features)
			)

		self.residual = residual

	def forward(self, x):
		
		scale = self.cond_scale(x[1])  # (N, n_features, H, W)
		shift = self.cond_shift(x[1])  # (N, n_features, H, W)
		out = x[0] * scale + shift

		if self.residual:
			return out + x[0]
		else:
			return out

class Res_GFM_LFM(nn.Module):
	def __init__(self, cond_c=32, out_nc=32, nf=64):
	#  out_nc需要跟chan一致
		super(Res_GFM_LFM, self).__init__()
		self.gfm = GFMLayer(cond_c=cond_c, out_nc=out_nc, nf=nf)
		self.conv1 = nn.Conv2d(out_nc, out_nc, 3, 1, 1)
		self.lfm = SFTLayer(cond_c=cond_c, out_nc=out_nc, nf=out_nc)
		# self.lfm = SpatialFeatureTransform(cond_channels=cond_c, n_features=out_nc, ada_method='cbam')
		self.conv2 = nn.Conv2d(out_nc, out_nc, 3, 1, 1)
		
  

	def forward(self, x):
		# x[0]: fea; x[1]: vector; x[2]: fea
		fea = self.gfm((x[0], x[1]))
		fea = F.leaky_relu(self.conv1(fea), negative_slope=0.1, inplace=True)
		fea = self.lfm((fea, x[2]))
		fea = self.conv2(fea)

		return (x[0] + fea, x[1], x[2])



class ISPNet_gfm(nn.Module):
	def __init__(self):
		super(ISPNet_gfm, self).__init__()
		cond_c=32	
		self.classifier = Color_Condition_GFM(in_channels=4, out_c=cond_c)
		chan = 48
		
		self.intro = nn.Conv2d(4, chan, 3, 1, 1)
		self.GFM_layer1 = Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=64)
		self.GFM_layer2 = Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=64)
		self.GFM_layer3 = Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=64)
		self.GFM_layer4 = Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=64)
		self.GFM_layer5 = Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=64)
		self.GFM_layer6 = Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=64)
		# self.ending = nn.Conv2d(chan, 3, 3, 1, 1)
		self.ending = N.seq(
			nn.Conv2d(chan, chan*4, 3, 1, 1),
			nn.PixelShuffle(upscale_factor=2),
			nn.Conv2d(chan, 3, 3, 1, 1)
		)  # shape: (N, 3, H*2, W*2)   

		self.GFM_last = GFMLayer(cond_c=32, out_nc=3, nf=64)
	def forward(self, x):
		fea = self.intro(x[0])
		gfm_vector = self.classifier(x[1])
		gfm_vector = gfm_vector.squeeze(2).squeeze(2)
		fea = self.GFM_layer1((fea, gfm_vector))
		fea = self.GFM_layer2((fea, gfm_vector))
		fea = self.GFM_layer3((fea, gfm_vector))
		fea = self.GFM_layer4((fea, gfm_vector))
		fea = self.GFM_layer5((fea, gfm_vector))
		fea = self.GFM_layer6((fea, gfm_vector))
		fea = self.ending(fea)
		out = self.GFM_last((fea, gfm_vector))
		return out



class ISPNet_modulation(nn.Module):
	def __init__(self):
		super(ISPNet_modulation, self).__init__()

		cond_c=32	
		self.classifier = Color_Condition(cond_c=cond_c)
		self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

		chan = 32
		n_blocks = 2
		modulation_blocks = 2
		self.intro = N.seq(
			nn.Conv2d(4, chan, 3, 1, 1)
		)  # shape: (N, 32, H, W)
  
		# down1
		self.encoder_modulation1 = N.seq(
	  		*[ResBlock_with_modulation(nf=chan) for _ in range(modulation_blocks)]
		)
		self.encoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down1 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 64, H//2, W//2)
		chan = chan * 2

		# down2	
		self.encoder_modulation2 = N.seq(
	  		*[ResBlock_with_modulation(nf=chan) for _ in range(modulation_blocks)]
		)
		self.encoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down2 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 128, H//4, W//4)
		chan = chan * 2
  
		# down3
		self.encoder_modulation3 = N.seq(
	  		*[ResBlock_with_modulation(nf=chan) for _ in range(modulation_blocks)]
		)
		self.encoder3 = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down3 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 256, H//8, W//8)
		chan = chan * 2		

		# middle
		self.middle_modulation = N.seq(
	  		*[ResBlock_with_modulation(nf=chan) for _ in range(modulation_blocks)]
		)
		self.middle = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
		)  # shape: (N, 256, H//8, W//8)

		# up3
		self.up3 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder_modulation3 = N.seq(
	  		*[ResBlock_with_modulation(chan) for _ in range(modulation_blocks)]
		)
		self.decoder3 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 128, H//8, W//8)

		# up2
		self.up2 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder_modulation2 = N.seq(
	  		*[ResBlock_with_modulation(chan) for _ in range(modulation_blocks)]
		)
		self.decoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 64, H//8, W//8)

		# up1
		self.up1 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan // 2
		self.decoder_modulation1 = N.seq(
	  		*[ResBlock_with_modulation(chan) for _ in range(modulation_blocks)]
		)
		self.decoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 32, H//8, W//8)
  
		self.tail = N.seq(
			N.conv(chan, chan*4, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(chan, 3, mode='C')
		)  # shape: (N, 3, H*2, W*2)   
	def forward(self, x):
		#  x[0]: input_img
		#  x[1]: input_cond
		fea_intro = self.intro(x[0])
		gfm_vector, lfm_feas = self.classifier(x[1])
		gfm_vector = gfm_vector.squeeze(2).squeeze(2)


		fea, _, _ = self.encoder_modulation1((fea_intro, gfm_vector, lfm_feas[0]))
		fea = self.encoder1(fea)
		d1 = self.down1(fea)

		fea, _, _ = self.encoder_modulation2((d1, gfm_vector, lfm_feas[1]))
		fea = self.encoder2(fea)
		d2 = self.down2(fea)

		fea, _, _ = self.encoder_modulation3((d2, gfm_vector, lfm_feas[2]))
		fea = self.encoder3(fea)
		d3 = self.down3(fea)

		middle, _, _ = self.middle_modulation((d3, gfm_vector, lfm_feas[3]))
		middle = self.middle(middle) + d3

		u3 = self.decoder3(self.up3(middle))
		u3, _, _ = self.decoder_modulation3((u3, gfm_vector, lfm_feas[2]))
		u3 = u3 + d2
  
		u2 = self.decoder2(self.up2(u3))
		u2, _, _ = self.decoder_modulation2((u2, gfm_vector, lfm_feas[1]))
		u2 = u2 + d1

		u1 = self.decoder1(self.up1(u2))
		u1, _, _ = self.decoder_modulation1((u1, gfm_vector, lfm_feas[0]))
		u1 = u1 + fea_intro

		
		out = self.tail(u1)
		return out


class ISPUNet_GFM_crop(nn.Module):
	def __init__(self):
		super(ISPUNet_GFM_crop, self).__init__()

		cond_c=64
		self.classifier = Color_Condition_GFM(in_channels=4, out_c=cond_c)
		# self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

		chan = 64
		n_blocks = 2
		modulation_blocks = 1
		self.intro = N.seq(
			nn.Conv2d(4, chan, 3, 1, 1)
		)  # shape: (N, 32, H, W)
  
		# down1
		self.encoder_modulation1 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)

		self.encoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down1 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 64, H//2, W//2)
		chan = chan * 2

		# down2	
		self.encoder_modulation2 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.encoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down2 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 128, H//4, W//4)
		chan = chan * 2
  
		# down3
		self.encoder_modulation3 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.encoder3 = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down3 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 256, H//8, W//8)
		chan = chan * 2		

		# middle
		self.middle_modulation = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.middle = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks*2),
			nn.Conv2d(chan, chan, 3, 1, 1),
		)  # shape: (N, 256, H//8, W//8)

		# up3
		self.up3 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder_modulation3 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.decoder3 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 128, H//8, W//8)

		# up2
		self.up2 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder_modulation2 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.decoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 64, H//8, W//8)

		# up1
		self.up1 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan // 2
		self.decoder_modulation1 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.decoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 32, H//8, W//8)
  
		self.tail = N.seq(
			N.conv(chan, chan*4, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(chan, 3, mode='C')
		)  # shape: (N, 3, H*2, W*2)   
	def forward(self, x):
		#  x[0]: input_img
		#  x[1]: input_cond
		#  x[2]: input_coord
		#  以自身为condition，global(crop)
		fea_intro = self.intro(x[0])
		gfm_vector = self.classifier(x[0])
		gfm_vector = gfm_vector.squeeze(2).squeeze(2)


		fea, _ = self.encoder_modulation1((fea_intro, gfm_vector))
		fea = self.encoder1(fea)
		d1 = self.down1(fea)

		fea, _ = self.encoder_modulation2((d1, gfm_vector))
		fea = self.encoder2(fea)
		d2 = self.down2(fea)

		fea, _ = self.encoder_modulation3((d2, gfm_vector))
		fea = self.encoder3(fea)
		d3 = self.down3(fea)

		middle, _ = self.middle_modulation((d3, gfm_vector))
		middle = self.middle(middle) + d3

		u3 = self.decoder3(self.up3(middle))
		u3, _ = self.decoder_modulation3((u3, gfm_vector))
		u3 = u3 + d2
  
		u2 = self.decoder2(self.up2(u3))
		u2, _ = self.decoder_modulation2((u2, gfm_vector))
		u2 = u2 + d1

		u1 = self.decoder1(self.up1(u2))
		u1, _ = self.decoder_modulation1((u1, gfm_vector))
		u1 = u1 + fea_intro

		
		out = self.tail(u1)
		return out


class ISPUNet_GFM(nn.Module):
	def __init__(self):
		super(ISPUNet_GFM, self).__init__()

		cond_c=32
		self.classifier = Color_Condition_GFM(in_channels=4, out_c=cond_c)
		# self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

		chan = 32
		n_blocks = 2
		modulation_blocks = 2
		self.intro = N.seq(
			nn.Conv2d(4, chan, 3, 1, 1)
		)  # shape: (N, 32, H, W)
  
		# down1
		self.encoder_modulation1 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)

		self.encoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down1 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 64, H//2, W//2)
		chan = chan * 2

		# down2	
		self.encoder_modulation2 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.encoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down2 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 128, H//4, W//4)
		chan = chan * 2
  
		# down3
		self.encoder_modulation3 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.encoder3 = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down3 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 256, H//8, W//8)
		chan = chan * 2		

		# middle
		self.middle_modulation = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.middle = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks*2),
			nn.Conv2d(chan, chan, 3, 1, 1),
		)  # shape: (N, 256, H//8, W//8)

		# up3
		self.up3 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder_modulation3 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.decoder3 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 128, H//8, W//8)

		# up2
		self.up2 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder_modulation2 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.decoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 64, H//8, W//8)

		# up1
		self.up1 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan // 2
		self.decoder_modulation1 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.decoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 32, H//8, W//8)
  
		self.tail = N.seq(
			N.conv(chan, chan*4, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(chan, 3, mode='C')
		)  # shape: (N, 3, H*2, W*2)   
	def forward(self, x):
		#  x[0]: input_img
		#  x[1]: input_cond
		fea_intro = self.intro(x[0])
		gfm_vector = self.classifier(x[1])
		gfm_vector = gfm_vector.squeeze(2).squeeze(2)


		fea, _ = self.encoder_modulation1((fea_intro, gfm_vector))
		fea = self.encoder1(fea)
		d1 = self.down1(fea)

		fea, _ = self.encoder_modulation2((d1, gfm_vector))
		fea = self.encoder2(fea)
		d2 = self.down2(fea)

		fea, _ = self.encoder_modulation3((d2, gfm_vector))
		fea = self.encoder3(fea)
		d3 = self.down3(fea)

		middle, _ = self.middle_modulation((d3, gfm_vector))
		middle = self.middle(middle) + d3

		u3 = self.decoder3(self.up3(middle))
		u3, _ = self.decoder_modulation3((u3, gfm_vector))
		u3 = u3 + d2
  
		u2 = self.decoder2(self.up2(u3))
		u2, _ = self.decoder_modulation2((u2, gfm_vector))
		u2 = u2 + d1

		u1 = self.decoder1(self.up1(u2))
		u1, _ = self.decoder_modulation1((u1, gfm_vector))
		u1 = u1 + fea_intro

		
		out = self.tail(u1)
		return out


class ISPUNet_LSC(nn.Module):
	def __init__(self):
		super(ISPUNet_LSC, self).__init__()
		

		chan = 32
		n_blocks = 2
		self.intro = N.seq(
			nn.Conv2d(4, chan, 3, 1, 1)
		)  # shape: (N, 32, H, W)
		self.lsc = Lens_Shading_Correction(in_channels=2, out_c=chan, nf=chan)
		# down1

		self.encoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down1 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 64, H//2, W//2)
		chan = chan * 2

		# down2	
		self.encoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down2 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 128, H//4, W//4)
		chan = chan * 2
  
		# down3
		self.encoder3 = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down3 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 256, H//8, W//8)
		chan = chan * 2		

		# middle
		self.middle = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks*2),
			nn.Conv2d(chan, chan, 3, 1, 1),
		)  # shape: (N, 256, H//8, W//8)

		# up3
		self.up3 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder3 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 128, H//8, W//8)

		# up2
		self.up2 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 64, H//8, W//8)

		# up1
		self.up1 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan // 2
		self.decoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 32, H//8, W//8)
  
		self.tail = N.seq(
			N.conv(chan, chan*4, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(chan, 3, mode='C')
		)  # shape: (N, 3, H*2, W*2)   
	def forward(self, x):
		#  x[0]: input_img
		#  x[1]: input_cond
		#  x[2]: coord_img
		
		fea_intro = self.intro(x[0])
		lsc_fea = self.lsc(x[2])

		fea_intro = fea_intro*(lsc_fea+1)

		fea = self.encoder1(fea_intro)
		d1 = self.down1(fea)
		fea = self.encoder2(d1)
		d2 = self.down2(fea)
		fea = self.encoder3(d2)
		d3 = self.down3(fea)

		middle = self.middle(d3) + d3

		u3 = self.decoder3(self.up3(middle))
		u3 = u3 + d2
		u2 = self.decoder2(self.up2(u3))
		u2 = u2 + d1
		u1 = self.decoder1(self.up1(u2))
		u1 = u1 + fea_intro
		
		out = self.tail(u1)
		return out


class ISPUNet_GFM_LSC(nn.Module):
	def __init__(self, cond_c=32, chan=32, m_blocks=2):
		super(ISPUNet_GFM_LSC, self).__init__()

		cond_c=cond_c
		self.classifier = Color_Condition_GFM(in_channels=4, out_c=cond_c)
		

		chan = chan
		n_blocks = 2
		modulation_blocks = m_blocks
		self.intro = N.seq(
			nn.Conv2d(4, chan, 3, 1, 1)
		)  # shape: (N, 32, H, W)
		self.lsc = Lens_Shading_Correction(in_channels=2, out_c=chan, nf=chan)
		# down1
		self.encoder_modulation1 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)

		self.encoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down1 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 64, H//2, W//2)
		chan = chan * 2

		# down2	
		self.encoder_modulation2 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.encoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down2 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 128, H//4, W//4)
		chan = chan * 2
  
		# down3
		self.encoder_modulation3 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.encoder3 = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down3 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 256, H//8, W//8)
		chan = chan * 2		

		# middle
		self.middle_modulation = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.middle = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks*2),
			nn.Conv2d(chan, chan, 3, 1, 1),
		)  # shape: (N, 256, H//8, W//8)

		# up3
		self.up3 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder_modulation3 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.decoder3 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 128, H//8, W//8)

		# up2
		self.up2 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder_modulation2 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.decoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 64, H//8, W//8)

		# up1
		self.up1 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan // 2
		self.decoder_modulation1 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.decoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 32, H//8, W//8)
  
		self.tail = N.seq(
			N.conv(chan, chan*4, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(chan, 3, mode='C')
		)  # shape: (N, 3, H*2, W*2)   
	def forward(self, x):
		#  x[0]: input_img
		#  x[1]: input_cond
		#  x[2]: coord_img
		# print(x[0].shape, x[1].shape, x[2].shape)
		fea_intro = self.intro(x[0])
		gfm_vector = self.classifier(x[1])
		lsc_fea = self.lsc(x[2])
		gfm_vector = gfm_vector.squeeze(2).squeeze(2)

		fea_intro = fea_intro*(lsc_fea+1)

		fea, _ = self.encoder_modulation1((fea_intro, gfm_vector))
		fea = self.encoder1(fea)
		d1 = self.down1(fea)

		fea, _ = self.encoder_modulation2((d1, gfm_vector))
		fea = self.encoder2(fea)
		d2 = self.down2(fea)

		fea, _ = self.encoder_modulation3((d2, gfm_vector))
		fea = self.encoder3(fea)
		d3 = self.down3(fea)

		middle, _ = self.middle_modulation((d3, gfm_vector))
		middle = self.middle(middle) + d3

		u3 = self.decoder3(self.up3(middle))
		u3, _ = self.decoder_modulation3((u3, gfm_vector))
		u3 = u3 + d2
  
		u2 = self.decoder2(self.up2(u3))
		u2, _ = self.decoder_modulation2((u2, gfm_vector))
		u2 = u2 + d1

		u1 = self.decoder1(self.up1(u2))
		u1, _ = self.decoder_modulation1((u1, gfm_vector))
		u1 = u1 + fea_intro

		
		out = self.tail(u1)
		return out


class ISPUNet_GFM_LSC1(nn.Module):
	# 直接将位置编码concat到RGGB图像里去
	def __init__(self):
		super(ISPUNet_GFM_LSC1, self).__init__()

		cond_c=32
		self.classifier = Color_Condition_GFM(in_channels=4, out_c=cond_c)
		

		chan = 32
		n_blocks = 2
		modulation_blocks = 2
		self.intro = N.seq(
			nn.Conv2d(6, chan, 3, 1, 1)
		)  # shape: (N, 32, H, W)

		# down1
		self.encoder_modulation1 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)

		self.encoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down1 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 64, H//2, W//2)
		chan = chan * 2

		# down2	
		self.encoder_modulation2 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.encoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down2 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 128, H//4, W//4)
		chan = chan * 2
  
		# down3
		self.encoder_modulation3 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.encoder3 = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down3 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 256, H//8, W//8)
		chan = chan * 2		

		# middle
		self.middle_modulation = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.middle = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks*2),
			nn.Conv2d(chan, chan, 3, 1, 1),
		)  # shape: (N, 256, H//8, W//8)

		# up3
		self.up3 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder_modulation3 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.decoder3 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 128, H//8, W//8)

		# up2
		self.up2 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder_modulation2 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.decoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 64, H//8, W//8)

		# up1
		self.up1 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan // 2
		self.decoder_modulation1 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.decoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 32, H//8, W//8)
  
		self.tail = N.seq(
			N.conv(chan, chan*4, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(chan, 3, mode='C')
		)  # shape: (N, 3, H*2, W*2)   
	def forward(self, x):
		#  x[0]: input_img
		#  x[1]: input_cond
		#  x[2]: coord_img
		
		fea_intro = self.intro(torch.cat([x[0], x[2]], dim=1))
		gfm_vector = self.classifier(x[1])
		gfm_vector = gfm_vector.squeeze(2).squeeze(2)


		fea, _ = self.encoder_modulation1((fea_intro, gfm_vector))
		fea = self.encoder1(fea)
		d1 = self.down1(fea)

		fea, _ = self.encoder_modulation2((d1, gfm_vector))
		fea = self.encoder2(fea)
		d2 = self.down2(fea)

		fea, _ = self.encoder_modulation3((d2, gfm_vector))
		fea = self.encoder3(fea)
		d3 = self.down3(fea)

		middle, _ = self.middle_modulation((d3, gfm_vector))
		middle = self.middle(middle) + d3

		u3 = self.decoder3(self.up3(middle))
		u3, _ = self.decoder_modulation3((u3, gfm_vector))
		u3 = u3 + d2
  
		u2 = self.decoder2(self.up2(u3))
		u2, _ = self.decoder_modulation2((u2, gfm_vector))
		u2 = u2 + d1

		u1 = self.decoder1(self.up1(u2))
		u1, _ = self.decoder_modulation1((u1, gfm_vector))
		u1 = u1 + fea_intro

		
		out = self.tail(u1)
		return out


class ISPUNet_GFM_LFM(nn.Module):
	def __init__(self, cond_c=32, n_blocks=2, modulation_blocks=1, chan=32):
		super(ISPUNet_GFM_LFM, self).__init__()

		self.cond_c=cond_c
		# self.classifier = Color_Condition_GFM(in_channels=4, out_c=cond_c)
		self.classifier = Color_Condition_GFM_LFM(in_channels=4, GFM_out_c=cond_c, LFM_out_c=cond_c)

		self.chan = chan
		self.n_blocks = n_blocks
		self.modulation_blocks = modulation_blocks
		self.intro = N.seq(
			nn.Conv2d(4, chan, 3, 1, 1)
		)  # shape: (N, 32, H, W)
  
		# down1
		self.encoder_modulation1 = N.seq(
			*[Res_GFM_LFM(cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)

		self.encoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down1 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 64, H//2, W//2)
		chan = chan * 2

		# down2	
		self.encoder_modulation2 = N.seq(
	  		*[Res_GFM_LFM(cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.encoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down2 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 128, H//4, W//4)
		chan = chan * 2
  
		# down3
		self.encoder_modulation3 = N.seq(
	  		*[Res_GFM_LFM(cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.encoder3 = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down3 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 256, H//8, W//8)
		chan = chan * 2		

		# middle
		self.middle_modulation = N.seq(
	  		*[Res_GFM_LFM(cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.middle = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks*2),
			nn.Conv2d(chan, chan, 3, 1, 1),
		)  # shape: (N, 256, H//8, W//8)

		# up3
		self.up3 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder_modulation3 = N.seq(
	  		*[Res_GFM_LFM(cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.decoder3 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 128, H//8, W//8)

		# up2
		self.up2 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder_modulation2 = N.seq(
	  		*[Res_GFM_LFM(cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.decoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 64, H//8, W//8)

		# up1
		self.up1 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan // 2
		self.decoder_modulation1 = N.seq(
	  		*[Res_GFM_LFM(cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.decoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 32, H//8, W//8)
  
		self.tail = N.seq(
			N.conv(chan, chan*4, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(chan, 3, mode='C')
		)  # shape: (N, 3, H*2, W*2)   

		# local modulation
		self.CondNet1 = nn.Sequential(nn.Conv2d(cond_c, cond_c, 1), 
								nn.LeakyReLU(0.1, True), 
								nn.Conv2d(cond_c, cond_c, 1)) # H W
		self.CondNet2 = nn.Sequential(nn.Conv2d(cond_c, cond_c, 2, 2), 
								nn.LeakyReLU(0.1, True), 
								nn.Conv2d(cond_c, cond_c, 1)) # H/2 W/2
		self.CondNet3 = nn.Sequential(nn.Conv2d(cond_c, cond_c, 2, 2), 
								nn.LeakyReLU(0.1, True), 
								nn.Conv2d(cond_c, cond_c, 2, 2), 
								nn.LeakyReLU(0.1, True), 
								nn.Conv2d(cond_c, cond_c, 1)) # H/4 W/4
		self.CondNet4 = nn.Sequential(nn.Conv2d(cond_c, cond_c, 2, 2), 
								nn.LeakyReLU(0.1, True), 
								nn.Conv2d(cond_c, cond_c, 2, 2), 
								nn.LeakyReLU(0.1, True), 
								nn.Conv2d(cond_c, cond_c, 2, 2), 
								nn.LeakyReLU(0.1, True), 
								nn.Conv2d(cond_c, cond_c, 1)) # H/8 W/8


	def forward(self, x):
		#  x[0]: input_img
		#  x[1]: input_cond
		fea_intro = self.intro(x[0])
		gfm_vector, lfm_fea = self.classifier(x[1], x[0])
		gfm_vector = gfm_vector.squeeze(2).squeeze(2)
		lfm_fea_1 = self.CondNet1(lfm_fea)
		lfm_fea_2 = self.CondNet2(lfm_fea)
		lfm_fea_4 = self.CondNet3(lfm_fea)
		lfm_fea_8 = self.CondNet4(lfm_fea)

		fea, _, _ = self.encoder_modulation1((fea_intro, gfm_vector, lfm_fea_1))
		fea = self.encoder1(fea)
		d1 = self.down1(fea)

		fea, _, _ = self.encoder_modulation2((d1, gfm_vector, lfm_fea_2))
		fea = self.encoder2(fea)
		d2 = self.down2(fea)

		fea, _, _ = self.encoder_modulation3((d2, gfm_vector, lfm_fea_4))
		fea = self.encoder3(fea)
		d3 = self.down3(fea)

		middle, _, _ = self.middle_modulation((d3, gfm_vector, lfm_fea_8))
		middle = self.middle(middle) + d3

		u3 = self.decoder3(self.up3(middle))
		u3, _, _ = self.decoder_modulation3((u3, gfm_vector, lfm_fea_4))
		u3 = u3 + d2
  
		u2 = self.decoder2(self.up2(u3))
		u2, _, _ = self.decoder_modulation2((u2, gfm_vector, lfm_fea_2))
		u2 = u2 + d1

		u1 = self.decoder1(self.up1(u2))
		u1, _, _ = self.decoder_modulation1((u1, gfm_vector, lfm_fea_1))
		u1 = u1 + fea_intro

		
		out = self.tail(u1)
		return out


class LiteISPNet_LSC(nn.Module):
	def __init__(self):
		super(LiteISPNet_LSC, self).__init__()
		ch_1 = 48
		ch_2 = 128
		ch_3 = 128
		n_blocks = 4

		# modulation_blocks = 1
		self.head = N.seq(
			N.conv(4, ch_1, mode='C')
		)  # shape: (N, ch_1, H, W)
		self.lsc = Lens_Shading_Correction(in_channels=2, out_c=ch_1, nf=ch_1)
		# down1

		self.down1 = N.seq(
			N.conv(ch_1, ch_1, mode='C'),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1, mode='C'),
			N.DWTForward(ch_1)
		)  # shape: (N, ch_1*4, H/2, W/2)

		self.down2 = N.seq(
			N.conv(ch_1*4, ch_1, mode='C'),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.DWTForward(ch_1)
		)  # shape: (N, ch_1*4, H/4, W/4)

		self.down3 = N.seq(
			N.conv(ch_1*4, ch_2, mode='C'),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.DWTForward(ch_2)
		)  # shape: (N, ch_2*4, H/8, W/8)

		self.middle = N.seq(
			N.conv(ch_2*4, ch_3, mode='C'),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.conv(ch_3, ch_2*4, mode='C')
		)  # shape: (N, ch_2*4, H/8, W/8)

		self.up3 = N.seq(
			N.DWTInverse(ch_2*4),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.conv(ch_2, ch_1*4, mode='C')
		)  # shape: (N, ch_1*4, H/4, W/4)

		self.up2 = N.seq(
			N.DWTInverse(ch_1*4),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1*4, mode='C')
		)  # shape: (N, ch_1*4, H/2, W/2)

		self.up1 = N.seq(
			N.DWTInverse(ch_1*4),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1, mode='C')
		)  # shape: (N, ch_1, H, W)

		self.tail = N.seq(
			N.conv(ch_1, ch_1*4, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(ch_1, 3, mode='C')
		)  # shape: (N, 3, H*2, W*2)   

	def forward(self, x):
		#  x[0]: input_img
		#  x[1]: input_cond
		#  x[2]: coord_img
		# input = raw
		# 为什么要先做pow
		# input = torch.pow(raw, 1/2.2)

		# 先做pad
		# raw, original_size = pad_to_multiple_of_16(raw)
		h = self.head(x[0])
		lsc_fea = self.lsc(x[2])
		h = h*(lsc_fea+1)

		# h, _ = self.encoder_modulation1((h, gfm_vector))		
		d1 = self.down1(h)
  
		# d2, _ = self.encoder_modulation2((d1, gfm_vector))
		d2 = self.down2(d1)
		# d3, _ = self.encoder_modulation3((d2, gfm_vector))
		d3 = self.down3(d2)
  
		# d4, _ = self.encoder_modulation4((d3, gfm_vector))
		m = self.middle(d3) + d3
		u3 = self.up3(m) + d2
		u2 = self.up2(u3) + d1
		u1 = self.up1(u2) + h
		out = self.tail(u1)
		# out = remove_padding(out, original_size)

		return out



class LiteISPNet_GFM(nn.Module):
	def __init__(self):
		super(LiteISPNet_GFM, self).__init__()
		ch_1 = 64
		ch_2 = 128
		ch_3 = 128
		n_blocks = 4
		cond_c=64
		self.classifier = Color_Condition_GFM(in_channels=4, out_c=cond_c)
		modulation_blocks = 1
		self.head = N.seq(
			N.conv(4, ch_1, mode='C')
		)  # shape: (N, ch_1, H, W)
		# self.lsc = Lens_Shading_Correction(in_channels=2, out_c=ch_1, nf=ch_1)
		# down1
		self.encoder_modulation1 = N.seq(
	  		*[Res_GFM(in_nc=ch_1, chan=ch_1, cond_c=cond_c, out_nc=ch_1, nf=ch_1) for _ in range(modulation_blocks)]
		)

		self.down1 = N.seq(
			N.conv(ch_1, ch_1, mode='C'),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1, mode='C'),
			N.DWTForward(ch_1)
		)  # shape: (N, ch_1*4, H/2, W/2)

		self.encoder_modulation2 = N.seq(
	  		*[Res_GFM(in_nc=ch_1*4, chan=ch_1*4, cond_c=cond_c, out_nc=ch_1*4, nf=ch_1) for _ in range(modulation_blocks)]
		)
		self.down2 = N.seq(
			N.conv(ch_1*4, ch_1, mode='C'),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.DWTForward(ch_1)
		)  # shape: (N, ch_1*4, H/4, W/4)

		self.encoder_modulation3 = N.seq(
	  		*[Res_GFM(in_nc=ch_1*4, chan=ch_1*4, cond_c=cond_c, out_nc=ch_1*4, nf=ch_1) for _ in range(modulation_blocks)]
		)
		self.down3 = N.seq(
			N.conv(ch_1*4, ch_2, mode='C'),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.DWTForward(ch_2)
		)  # shape: (N, ch_2*4, H/8, W/8)

		self.encoder_modulation4 = N.seq(
	  		*[Res_GFM(in_nc=ch_2*4, chan=ch_2*4, cond_c=cond_c, out_nc=ch_2*4, nf=ch_2) for _ in range(modulation_blocks)]
		)
		self.middle = N.seq(
			N.conv(ch_2*4, ch_3, mode='C'),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.conv(ch_3, ch_2*4, mode='C')
		)  # shape: (N, ch_2*4, H/8, W/8)

		self.up3 = N.seq(
			N.DWTInverse(ch_2*4),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.conv(ch_2, ch_1*4, mode='C')
		)  # shape: (N, ch_1*4, H/4, W/4)

		self.up2 = N.seq(
			N.DWTInverse(ch_1*4),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1*4, mode='C')
		)  # shape: (N, ch_1*4, H/2, W/2)

		self.up1 = N.seq(
			N.DWTInverse(ch_1*4),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1, mode='C')
		)  # shape: (N, ch_1, H, W)

		self.tail = N.seq(
			N.conv(ch_1, ch_1*4, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(ch_1, 3, mode='C')
		)  # shape: (N, 3, H*2, W*2)   

	def forward(self, x):
		#  x[0]: input_img
		#  x[1]: input_cond
		#  x[2]: coord_img
		# input = raw
		# 为什么要先做pow
		# input = torch.pow(raw, 1/2.2)

		# 先做pad
		# raw, original_size = pad_to_multiple_of_16(raw)
		h = self.head(x[0])
		# lsc_fea = self.lsc(x[2])
		# h = h*(lsc_fea+1)
  
		gfm_vector = self.classifier(x[1])
		gfm_vector = gfm_vector.squeeze(2).squeeze(2)

		h, _ = self.encoder_modulation1((h, gfm_vector))		
		d1 = self.down1(h)
  
		d2, _ = self.encoder_modulation2((d1, gfm_vector))
		d2 = self.down2(d2)
		d3, _ = self.encoder_modulation3((d2, gfm_vector))
		d3 = self.down3(d3)
  
		d4, _ = self.encoder_modulation4((d3, gfm_vector))
		m = self.middle(d4) + d3
		u3 = self.up3(m) + d2
		u2 = self.up2(u3) + d1
		u1 = self.up1(u2) + h
		out = self.tail(u1)
		# out = remove_padding(out, original_size)

		return out



class LiteISPNet_GFM_LSC(nn.Module):
	def __init__(self):
		super(LiteISPNet_GFM_LSC, self).__init__()
		ch_1 = 48
		ch_2 = 128
		ch_3 = 128
		n_blocks = 4
		cond_c=32
		self.classifier = Color_Condition_GFM(in_channels=4, out_c=cond_c)
		modulation_blocks = 1
		self.head = N.seq(
			N.conv(4, ch_1, mode='C')
		)  # shape: (N, ch_1, H, W)
		self.lsc = Lens_Shading_Correction(in_channels=2, out_c=ch_1, nf=ch_1)
		# down1
		self.encoder_modulation1 = N.seq(
	  		*[Res_GFM(in_nc=ch_1, chan=ch_1, cond_c=cond_c, out_nc=ch_1, nf=ch_1) for _ in range(modulation_blocks)]
		)

		self.down1 = N.seq(
			N.conv(ch_1, ch_1, mode='C'),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1, mode='C'),
			N.DWTForward(ch_1)
		)  # shape: (N, ch_1*4, H/2, W/2)

		self.encoder_modulation2 = N.seq(
	  		*[Res_GFM(in_nc=ch_1*4, chan=ch_1*4, cond_c=cond_c, out_nc=ch_1*4, nf=ch_1) for _ in range(modulation_blocks)]
		)
		self.down2 = N.seq(
			N.conv(ch_1*4, ch_1, mode='C'),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.DWTForward(ch_1)
		)  # shape: (N, ch_1*4, H/4, W/4)

		self.encoder_modulation3 = N.seq(
	  		*[Res_GFM(in_nc=ch_1*4, chan=ch_1*4, cond_c=cond_c, out_nc=ch_1*4, nf=ch_1) for _ in range(modulation_blocks)]
		)
		self.down3 = N.seq(
			N.conv(ch_1*4, ch_2, mode='C'),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.DWTForward(ch_2)
		)  # shape: (N, ch_2*4, H/8, W/8)

		self.encoder_modulation4 = N.seq(
	  		*[Res_GFM(in_nc=ch_2*4, chan=ch_2*4, cond_c=cond_c, out_nc=ch_2*4, nf=ch_2) for _ in range(modulation_blocks)]
		)
		self.middle = N.seq(
			N.conv(ch_2*4, ch_3, mode='C'),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.conv(ch_3, ch_2*4, mode='C')
		)  # shape: (N, ch_2*4, H/8, W/8)

		self.up3 = N.seq(
			N.DWTInverse(ch_2*4),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.conv(ch_2, ch_1*4, mode='C')
		)  # shape: (N, ch_1*4, H/4, W/4)

		self.up2 = N.seq(
			N.DWTInverse(ch_1*4),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1*4, mode='C')
		)  # shape: (N, ch_1*4, H/2, W/2)

		self.up1 = N.seq(
			N.DWTInverse(ch_1*4),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1, mode='C')
		)  # shape: (N, ch_1, H, W)

		self.tail = N.seq(
			N.conv(ch_1, ch_1*4, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(ch_1, 3, mode='C')
		)  # shape: (N, 3, H*2, W*2)   

	def forward(self, x):
		#  x[0]: input_img
		#  x[1]: input_cond
		#  x[2]: coord_img
		# input = raw
		# 为什么要先做pow
		# input = torch.pow(raw, 1/2.2)

		# 先做pad
		# raw, original_size = pad_to_multiple_of_16(raw)
		h = self.head(x[0])
		lsc_fea = self.lsc(x[2])
		h = h*(lsc_fea+1)
  
		gfm_vector = self.classifier(x[1])
		gfm_vector = gfm_vector.squeeze(2).squeeze(2)

		h, _ = self.encoder_modulation1((h, gfm_vector))		
		d1 = self.down1(h)
  
		d2, _ = self.encoder_modulation2((d1, gfm_vector))
		d2 = self.down2(d2)
		d3, _ = self.encoder_modulation3((d2, gfm_vector))
		d3 = self.down3(d3)
  
		d4, _ = self.encoder_modulation4((d3, gfm_vector))
		m = self.middle(d4) + d3
		u3 = self.up3(m) + d2
		u2 = self.up2(u3) + d1
		u1 = self.up1(u2) + h
		out = self.tail(u1)
		# out = remove_padding(out, original_size)

		return out


class ResUNet(nn.Module):
	def __init__(self):
		super(ResUNet, self).__init__()
		

		chan = 32
		n_blocks = 2
		self.intro = N.seq(
			nn.Conv2d(4, chan, 3, 1, 1)
		)  # shape: (N, 32, H, W)
		# down1

		self.encoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down1 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 64, H//2, W//2)
		chan = chan * 2

		# down2	
		self.encoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down2 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 128, H//4, W//4)
		chan = chan * 2
  
		# down3
		self.encoder3 = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down3 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 256, H//8, W//8)
		chan = chan * 2		

		# middle
		self.middle = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks*2),
			nn.Conv2d(chan, chan, 3, 1, 1),
		)  # shape: (N, 256, H//8, W//8)

		# up3
		self.up3 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder3 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 128, H//8, W//8)

		# up2
		self.up2 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 64, H//8, W//8)

		# up1
		self.up1 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan // 2
		self.decoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 32, H//8, W//8)
  
		self.tail = N.seq(
			N.conv(chan, chan*4, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(chan, 3, mode='C')
		)  # shape: (N, 3, H*2, W*2)   
	def forward(self, x):
		#  x[0]: input_img
		#  x[1]: input_cond
		#  x[2]: coord_img
		
		fea_intro = self.intro(x[0])

		fea = self.encoder1(fea_intro)
		d1 = self.down1(fea)
		fea = self.encoder2(d1)
		d2 = self.down2(fea)
		fea = self.encoder3(d2)
		d3 = self.down3(fea)

		middle = self.middle(d3) + d3

		u3 = self.decoder3(self.up3(middle))
		u3 = u3 + d2
		u2 = self.decoder2(self.up2(u3))
		u2 = u2 + d1
		u1 = self.decoder1(self.up1(u2))
		u1 = u1 + fea_intro
		
		out = self.tail(u1)
		return out


class MWISP(nn.Module):
    def __init__(self):
        super(MWISP, self).__init__()
        c1 = 64
        c2 = 128
        c3 = 128
        n_b = 20
        self.head = N.DWTForward_()

        self.down1 = N.seq(
            nn.Conv2d(4 * 4, c1, 3, 1, 1),
            nn.PReLU(),
            N.RCAGroup(in_channels=c1, out_channels=c1, nb=n_b)
        )

        self.down2 = N.seq(
            N.DWTForward_(),
            nn.Conv2d(c1 * 4, c2, 3, 1, 1),
            nn.PReLU(),
              N.RCAGroup(in_channels=c2, out_channels=c2, nb=n_b)
        )

        self.down3 = N.seq(
            N.DWTForward_(),
            nn.Conv2d(c2 * 4, c3, 3, 1, 1),
            nn.PReLU()
        )

        self.middle = N.seq(
            N.RCAGroup(in_channels=c3, out_channels=c3, nb=n_b),
            N.RCAGroup(in_channels=c3, out_channels=c3, nb=n_b)
        )
        
        self.up1 = N.seq(
            nn.Conv2d(c3, c2 * 4, 3, 1, 1),
            nn.PReLU(),
            N.DWTInverse_()
        )

        self.up2 = N.seq(
            N.RCAGroup(in_channels=c2, out_channels=c2, nb=n_b),
            nn.Conv2d(c2, c1 * 4, 3, 1, 1),
            nn.PReLU(),
            N.DWTInverse_()
        )

        self.up3 = N.seq(
            N.RCAGroup(in_channels=c1, out_channels=c1, nb=n_b),
            nn.Conv2d(c1, 16, 3, 1, 1)
        )

        self.tail = N.seq(
            N.DWTInverse_(),
            nn.Conv2d(4, 12, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2)
        )

    def forward(self, x, c=None):
        c0 = x[0]
        c1 = self.head(c0)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        m = self.middle(c4)
        c5 = self.up1(m) + c3
        c6 = self.up2(c5) + c2
        c7 = self.up3(c6) + c1
        out = self.tail(c7)

        return out


class AWNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, block=[2, 2, 2, 4, 4]):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        # layer1
        _layer_1_dw = []
        for i in range(block[0]):
            _layer_1_dw.append(GCRDB(64, ContextBlock2d))
        _layer_1_dw.append(GCWTResDown(64, ContextBlock2d, norm_layer=None))
        self.layer1 = nn.Sequential(*_layer_1_dw)

        # layer 2
        _layer_2_dw = []
        for i in range(block[1]):
            _layer_2_dw.append(GCRDB(128, ContextBlock2d))
        _layer_2_dw.append(GCWTResDown(128, ContextBlock2d, norm_layer=None))
        self.layer2 = nn.Sequential(*_layer_2_dw)

        # layer 3
        _layer_3_dw = []
        for i in range(block[2]):
            _layer_3_dw.append(GCRDB(256, ContextBlock2d))
        _layer_3_dw.append(GCWTResDown(256, ContextBlock2d, norm_layer=None))
        self.layer3 = nn.Sequential(*_layer_3_dw)

        # layer 4
        _layer_4_dw = []
        for i in range(block[3]):
            _layer_4_dw.append(GCRDB(512, ContextBlock2d))
        _layer_4_dw.append(GCWTResDown(512, ContextBlock2d, norm_layer=None))
        self.layer4 = nn.Sequential(*_layer_4_dw)

        # layer 5
        _layer_5_dw = []
        for i in range(block[4]):
            _layer_5_dw.append(GCRDB(1024, ContextBlock2d))
        self.layer5 = nn.Sequential(*_layer_5_dw)

        # upsample4
        self.layer4_up = GCIWTResUp(2048, ContextBlock2d)

        # upsample3
        self.layer3_up = GCIWTResUp(1024, ContextBlock2d)

        # upsample2
        self.layer2_up = GCIWTResUp(512, ContextBlock2d)

        # upsample1
        self.layer1_up = GCIWTResUp(256, ContextBlock2d)

        self.sc_x1 = shortcutblock(64, 64)
        self.sc_x2 = shortcutblock(128, 128)
        self.sc_x3 = shortcutblock(256, 256)
        self.sc_x4 = shortcutblock(512, 512)

        self.scale_5 = nn.Conv2d(1024, out_channels, kernel_size=3, padding=1)
        self.scale_4 = nn.Conv2d(512, out_channels, kernel_size=3, padding=1)
        self.scale_3 = nn.Conv2d(256, out_channels, kernel_size=3, padding=1)
        self.scale_2 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        self.scale_1 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        self.se1 = SE_net(64, 64)
        self.se2 = SE_net(128, 128)
        self.se3 = SE_net(256, 256)
        self.se4 = SE_net(512, 512)
        self.se5 = SE_net(1024, 1024)

        self.last = last_upsample()

    def forward(self, x):
		#  x[0]: input_img
		#  x[1]: input_cond
		#  x[2]: coord_img
        x1 = self.conv1(x[0])

        x2, x2_dwt = self.layer1(self.se1(x1))
        x3, x3_dwt = self.layer2(self.se2(x2))
        x4, x4_dwt = self.layer3(self.se3(x3))
        x5, x5_dwt = self.layer4(self.se4(x4))
        x5_latent = self.layer5(self.se5(x5))

        # x5_out = self.scale_5(x5_latent)
        # x5_out = F.sigmoid(x5_out)
        x4_up = self.layer4_up(x5_latent, x5_dwt) + self.sc_x4(x4)
        # x4_out = self.scale_4(x4_up)
        # x4_out = F.sigmoid(x4_out)
        x3_up = self.layer3_up(x4_up, x4_dwt) + self.sc_x3(x3)
        # x3_out = self.scale_3(x3_up)
        # x3_out = F.sigmoid(x3_out)
        x2_up = self.layer2_up(x3_up, x3_dwt) + self.sc_x2(x2)
        # x2_out = self.scale_2(x2_up)
        # x2_out = F.sigmoid(x2_out)
        x1_up = self.layer1_up(x2_up, x2_dwt) + self.sc_x1(x1)
        # x1_out = self.scale_1(x1_up)
        # x1_out = F.sigmoid(x1_out)
        out = self.last(x1_up)
        return out
        # return (out, x1_out, x2_out, x3_out, x4_out, x5_out), x5_latent


class LiteISPNet(nn.Module):
	def __init__(self):
		super(LiteISPNet, self).__init__()
		ch_1 = 64
		ch_2 = 128
		ch_3 = 128
		n_blocks = 4
		modulation_blocks = 1
		self.head = N.seq(
			N.conv(4, ch_1, mode='C')
		)  # shape: (N, ch_1, H, W)
		# down1

		self.down1 = N.seq(
			N.conv(ch_1, ch_1, mode='C'),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1, mode='C'),
			N.DWTForward(ch_1)
		)  # shape: (N, ch_1*4, H/2, W/2)

		self.down2 = N.seq(
			N.conv(ch_1*4, ch_1, mode='C'),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.DWTForward(ch_1)
		)  # shape: (N, ch_1*4, H/4, W/4)

		self.down3 = N.seq(
			N.conv(ch_1*4, ch_2, mode='C'),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.DWTForward(ch_2)
		)  # shape: (N, ch_2*4, H/8, W/8)

		self.middle = N.seq(
			N.conv(ch_2*4, ch_3, mode='C'),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.conv(ch_3, ch_2*4, mode='C')
		)  # shape: (N, ch_2*4, H/8, W/8)

		self.up3 = N.seq(
			N.DWTInverse(ch_2*4),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.conv(ch_2, ch_1*4, mode='C')
		)  # shape: (N, ch_1*4, H/4, W/4)

		self.up2 = N.seq(
			N.DWTInverse(ch_1*4),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1*4, mode='C')
		)  # shape: (N, ch_1*4, H/2, W/2)

		self.up1 = N.seq(
			N.DWTInverse(ch_1*4),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1, mode='C')
		)  # shape: (N, ch_1, H, W)

		self.tail = N.seq(
			N.conv(ch_1, ch_1*4, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(ch_1, 3, mode='C')
		)  # shape: (N, 3, H*2, W*2)   

	def forward(self, x):
		# input = raw
		# 为什么要先做pow
		# input = torch.pow(raw, 1/2.2)

		# 先做pad
		# raw, original_size = pad_to_multiple_of_16(raw)
		h = self.head(x[0])
		# gfm_vector = self.classifier(x[0])
		# gfm_vector = gfm_vector.squeeze(2).squeeze(2)

		# h, _ = self.encoder_modulation1((h, gfm_vector))		
		d1 = self.down1(h)
  
		# d2, _ = self.encoder_modulation2((d1, gfm_vector))
		d2 = self.down2(d1)
		# d3, _ = self.encoder_modulation3((d2, gfm_vector))
		d3 = self.down3(d2)
  
		# d4, _ = self.encoder_modulation4((d3, gfm_vector))
		m = self.middle(d3) + d3
		u3 = self.up3(m) + d2
		u2 = self.up2(u3) + d1
		u1 = self.up1(u2) + h
		out = self.tail(u1)
		# out = remove_padding(out, original_size)

		return out

class LiteISPNet_GFMresize(nn.Module):
	def __init__(self):
		super(LiteISPNet_GFMresize, self).__init__()
		ch_1 = 64
		ch_2 = 128
		ch_3 = 128
		n_blocks = 4
		cond_c=32
		self.classifier = Color_Condition_GFM(in_channels=4, out_c=cond_c)
		modulation_blocks = 1
		self.head = N.seq(
			N.conv(4, ch_1, mode='C')
		)  # shape: (N, ch_1, H, W)
		# down1
		self.encoder_modulation1 = N.seq(
	  		*[Res_GFM(in_nc=ch_1, chan=ch_1, cond_c=cond_c, out_nc=ch_1, nf=ch_1*2) for _ in range(modulation_blocks)]
		)

		self.down1 = N.seq(
			N.conv(ch_1, ch_1, mode='C'),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1, mode='C'),
			N.DWTForward(ch_1)
		)  # shape: (N, ch_1*4, H/2, W/2)

		self.encoder_modulation2 = N.seq(
	  		*[Res_GFM(in_nc=ch_1*4, chan=ch_1*4, cond_c=cond_c, out_nc=ch_1*4, nf=ch_1*4) for _ in range(modulation_blocks)]
		)
		self.down2 = N.seq(
			N.conv(ch_1*4, ch_1, mode='C'),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.DWTForward(ch_1)
		)  # shape: (N, ch_1*4, H/4, W/4)

		self.encoder_modulation3 = N.seq(
	  		*[Res_GFM(in_nc=ch_1*4, chan=ch_1*4, cond_c=cond_c, out_nc=ch_1*4, nf=ch_1*4) for _ in range(modulation_blocks)]
		)
		self.down3 = N.seq(
			N.conv(ch_1*4, ch_2, mode='C'),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.DWTForward(ch_2)
		)  # shape: (N, ch_2*4, H/8, W/8)

		self.encoder_modulation4 = N.seq(
	  		*[Res_GFM(in_nc=ch_2*4, chan=ch_2*4, cond_c=cond_c, out_nc=ch_2*4, nf=ch_2*4) for _ in range(modulation_blocks)]
		)
		self.middle = N.seq(
			N.conv(ch_2*4, ch_3, mode='C'),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.conv(ch_3, ch_2*4, mode='C')
		)  # shape: (N, ch_2*4, H/8, W/8)

		self.up3 = N.seq(
			N.DWTInverse(ch_2*4),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.conv(ch_2, ch_1*4, mode='C')
		)  # shape: (N, ch_1*4, H/4, W/4)

		self.up2 = N.seq(
			N.DWTInverse(ch_1*4),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1*4, mode='C')
		)  # shape: (N, ch_1*4, H/2, W/2)

		self.up1 = N.seq(
			N.DWTInverse(ch_1*4),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1, mode='C')
		)  # shape: (N, ch_1, H, W)

		self.tail = N.seq(
			N.conv(ch_1, ch_1*4, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(ch_1, 3, mode='C')
		)  # shape: (N, 3, H*2, W*2)   

	def forward(self, x):
		# input = raw
		# 为什么要先做pow
		# input = torch.pow(raw, 1/2.2)

		# 先做pad
		# raw, original_size = pad_to_multiple_of_16(raw)
		h = self.head(x[0])
		gfm_vector = self.classifier(x[0])
		gfm_vector = gfm_vector.squeeze(2).squeeze(2)


		# fea, _ = self.encoder_modulation1((fea_intro, gfm_vector))
		h, _ = self.encoder_modulation1((h, gfm_vector))		
		d1 = self.down1(h)
  
		d2, _ = self.encoder_modulation2((d1, gfm_vector))
		d2 = self.down2(d2)
		d3, _ = self.encoder_modulation3((d2, gfm_vector))
		d3 = self.down3(d3)
  
		d4, _ = self.encoder_modulation4((d3, gfm_vector))
		m = self.middle(d4) + d3
		u3 = self.up3(m) + d2
		u2 = self.up2(u3) + d1
		u1 = self.up1(u2) + h
		out = self.tail(u1)
		# out = remove_padding(out, original_size)

		return out

class ISPUNet_GFM_LSC_noskip(nn.Module):
	def __init__(self, cond_c=32, lsc_c=32):
		super(ISPUNet_GFM_LSC_noskip, self).__init__()

		cond_c=cond_c
		self.classifier = Color_Condition_GFM(in_channels=4, out_c=cond_c)
		

		chan = 32
		n_blocks = 2
		modulation_blocks = 1
		self.intro = N.seq(
			nn.Conv2d(4, chan, 3, 1, 1)
		)  # shape: (N, 32, H, W)
		self.lsc = Lens_Shading_Correction(in_channels=2, out_c=chan, nf=lsc_c)
		# down1
		self.encoder_modulation1 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)

		self.encoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down1 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 64, H//2, W//2)
		chan = chan * 2

		# down2	
		self.encoder_modulation2 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.encoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down2 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 128, H//4, W//4)
		chan = chan * 2
  
		# down3
		self.encoder_modulation3 = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.encoder3 = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			nn.Conv2d(chan, chan, 3, 1, 1),
			nn.LeakyReLU(negative_slope=1e-1, inplace=True),
		)
		self.down3 = nn.Conv2d(chan, chan*2, 2, 2)  # shape: (N, 256, H//8, W//8)
		chan = chan * 2		

		# middle
		self.middle_modulation = N.seq(
	  		*[Res_GFM(in_nc=chan, chan=chan, cond_c=cond_c, out_nc=chan, nf=chan*2) for _ in range(modulation_blocks)]
		)
		self.middle = N.seq(
			nn.Conv2d(chan, chan, 3, 1, 1),
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks*2),
			nn.Conv2d(chan, chan, 3, 1, 1),
		)  # shape: (N, 256, H//8, W//8)

		# up3
		self.up3 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder3 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 128, H//8, W//8)

		# up2
		self.up2 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan//2
		self.decoder2 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 64, H//8, W//8)

		# up1
		self.up1 = N.seq(
			nn.Conv2d(chan, chan*2, 1, bias=False),
   			nn.PixelShuffle(2),
		)
		chan = chan // 2
		self.decoder1 = N.seq(
			N.RCAGroup(in_channels=chan, out_channels=chan, nb=n_blocks),
			N.conv(chan, chan, mode='C')
		)  # shape: (N, 32, H//8, W//8)
  
		self.tail = N.seq(
			N.conv(chan, chan*4, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(chan, 3, mode='C')
		)  # shape: (N, 3, H*2, W*2)   
	def forward(self, x):
		#  x[0]: input_img
		#  x[1]: input_cond
		#  x[2]: coord_img
		
		fea_intro = self.intro(x[0])
		gfm_vector = self.classifier(x[1])
		lsc_fea = self.lsc(x[2])
		gfm_vector = gfm_vector.squeeze(2).squeeze(2)
		fea_intro = fea_intro*(lsc_fea+1)

		fea, _ = self.encoder_modulation1((fea_intro, gfm_vector))
		fea = self.encoder1(fea)
		fea = self.down1(fea)
		fea, _ = self.encoder_modulation2((fea, gfm_vector))
		fea = self.encoder2(fea)
		fea = self.down2(fea)
		fea, _ = self.encoder_modulation3((fea, gfm_vector))
		fea = self.encoder3(fea)
		fea = self.down3(fea)

		fea, _ = self.middle_modulation((fea, gfm_vector))
		fea = self.middle(fea)
		fea = self.decoder3(self.up3(fea))
		fea = self.decoder2(self.up2(fea))
		fea = self.decoder1(self.up1(fea))

		
		out = self.tail(fea)
		return out



import thop
from thop import profile 
from thop import clever_format
if __name__=='__main__':
	DEVICE = 'cpu'

	# net = LiteISPNet().to(DEVICE)
	net = LiteISPNet_GFM_LSC().to(DEVICE)
	# net = ISPUNet_GFM_LSC_noskip().to(DEVICE)
	# net = ISPNet_modulation()
	# net = ISPNet_gfm()
	# net = ISPUNet_GFM()
	# net = ISPUNet_GFM_LFM().to('cuda:0')
	# net = ISPUNet_GFM_LSC().to('cuda:0')
	data = torch.randn(1,4,256,256).to(DEVICE)
	data_coord = torch.randn(1,2,256,256).to(DEVICE)
	cond = torch.randn(1,4,256,256).to(DEVICE)
	inputs = [data, cond, data_coord]
	with torch.no_grad():
		flops, params = profile(net, inputs=(inputs, ))
	flops, params = clever_format([flops, params], "%.3f")
	print(flops)
	print(params)
	# out = net((data, cond, data_coord))
	# out = net(data)
	# print(out.shape)


