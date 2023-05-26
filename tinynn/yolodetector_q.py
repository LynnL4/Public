
import torch
import torch.nn
import torch.functional
import torch.nn.functional
import torch.quantization
import torch.nn.quantized


class QYOLODetector(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("tensor_0", torch.tensor([[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]], [[116, 90], [156, 198], [373, 326]]], dtype=torch.int64), persistent=False)
        self.fake_quant_0 = torch.quantization.QuantStub()
        self.backbone_stem_conv = torch.nn.Conv2d(3, 16, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=False)
        self.backbone_stem_bn = torch.nn.BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stem_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage1_0_conv = torch.nn.Conv2d(16, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.backbone_stage1_0_bn = torch.nn.BatchNorm2d(24, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage1_0_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage1_1_short_conv_conv = torch.nn.Conv2d(24, 12, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage1_1_short_conv_bn = torch.nn.BatchNorm2d(12, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage1_1_short_conv_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage1_1_main_conv_conv = torch.nn.Conv2d(24, 12, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage1_1_main_conv_bn = torch.nn.BatchNorm2d(12, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage1_1_main_conv_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage1_1_blocks_0_conv1_conv = torch.nn.Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage1_1_blocks_0_conv1_bn = torch.nn.BatchNorm2d(12, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage1_1_blocks_0_conv1_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage1_1_blocks_0_conv2_conv = torch.nn.Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone_stage1_1_blocks_0_conv2_bn = torch.nn.BatchNorm2d(12, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage1_1_blocks_0_conv2_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage1_1_final_conv_conv = torch.nn.Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage1_1_final_conv_bn = torch.nn.BatchNorm2d(24, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage1_1_final_conv_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage2_0_conv = torch.nn.Conv2d(24, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.backbone_stage2_0_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage2_0_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage2_1_short_conv_conv = torch.nn.Conv2d(40, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage2_1_short_conv_bn = torch.nn.BatchNorm2d(20, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage2_1_short_conv_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage2_1_main_conv_conv = torch.nn.Conv2d(40, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage2_1_main_conv_bn = torch.nn.BatchNorm2d(20, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage2_1_main_conv_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage2_1_blocks_0_conv1_conv = torch.nn.Conv2d(20, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage2_1_blocks_0_conv1_bn = torch.nn.BatchNorm2d(20, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage2_1_blocks_0_conv1_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage2_1_blocks_0_conv2_conv = torch.nn.Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone_stage2_1_blocks_0_conv2_bn = torch.nn.BatchNorm2d(20, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage2_1_blocks_0_conv2_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage2_1_blocks_1_conv1_conv = torch.nn.Conv2d(20, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage2_1_blocks_1_conv1_bn = torch.nn.BatchNorm2d(20, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage2_1_blocks_1_conv1_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage2_1_blocks_1_conv2_conv = torch.nn.Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone_stage2_1_blocks_1_conv2_bn = torch.nn.BatchNorm2d(20, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage2_1_blocks_1_conv2_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage2_1_final_conv_conv = torch.nn.Conv2d(40, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage2_1_final_conv_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage2_1_final_conv_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage3_0_conv = torch.nn.Conv2d(40, 80, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.backbone_stage3_0_bn = torch.nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage3_0_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage3_1_short_conv_conv = torch.nn.Conv2d(80, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage3_1_short_conv_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage3_1_short_conv_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage3_1_main_conv_conv = torch.nn.Conv2d(80, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage3_1_main_conv_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage3_1_main_conv_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage3_1_blocks_0_conv1_conv = torch.nn.Conv2d(40, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage3_1_blocks_0_conv1_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage3_1_blocks_0_conv1_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage3_1_blocks_0_conv2_conv = torch.nn.Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone_stage3_1_blocks_0_conv2_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage3_1_blocks_0_conv2_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage3_1_blocks_1_conv1_conv = torch.nn.Conv2d(40, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage3_1_blocks_1_conv1_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage3_1_blocks_1_conv1_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage3_1_blocks_1_conv2_conv = torch.nn.Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone_stage3_1_blocks_1_conv2_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage3_1_blocks_1_conv2_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage3_1_blocks_2_conv1_conv = torch.nn.Conv2d(40, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage3_1_blocks_2_conv1_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage3_1_blocks_2_conv1_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage3_1_blocks_2_conv2_conv = torch.nn.Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone_stage3_1_blocks_2_conv2_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage3_1_blocks_2_conv2_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage3_1_final_conv_conv = torch.nn.Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage3_1_final_conv_bn = torch.nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage3_1_final_conv_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage4_0_conv = torch.nn.Conv2d(80, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.backbone_stage4_0_bn = torch.nn.BatchNorm2d(160, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage4_0_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage4_1_short_conv_conv = torch.nn.Conv2d(160, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage4_1_short_conv_bn = torch.nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage4_1_short_conv_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage4_1_main_conv_conv = torch.nn.Conv2d(160, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage4_1_main_conv_bn = torch.nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage4_1_main_conv_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage4_1_blocks_0_conv1_conv = torch.nn.Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage4_1_blocks_0_conv1_bn = torch.nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage4_1_blocks_0_conv1_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage4_1_blocks_0_conv2_conv = torch.nn.Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone_stage4_1_blocks_0_conv2_bn = torch.nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage4_1_blocks_0_conv2_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage4_1_final_conv_conv = torch.nn.Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage4_1_final_conv_bn = torch.nn.BatchNorm2d(160, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage4_1_final_conv_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage4_2_conv1_conv = torch.nn.Conv2d(160, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage4_2_conv1_bn = torch.nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage4_2_conv1_activate = torch.nn.ReLU(inplace=True)
        self.backbone_stage4_2_poolings = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.backbone_stage4_2_conv2_conv = torch.nn.Conv2d(320, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone_stage4_2_conv2_bn = torch.nn.BatchNorm2d(160, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.backbone_stage4_2_conv2_activate = torch.nn.ReLU(inplace=True)
        self.neck_reduce_layers_0 = torch.nn.Identity()
        self.neck_reduce_layers_1 = torch.nn.Identity()
        self.neck_reduce_layers_2_conv = torch.nn.Conv2d(160, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_reduce_layers_2_bn = torch.nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_reduce_layers_2_activate = torch.nn.ReLU(inplace=True)
        self.neck_upsample_layers_0 = torch.nn.Upsample(scale_factor=2.0)
        self.neck_top_down_layers_0_0_short_conv_conv = torch.nn.Conv2d(160, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_top_down_layers_0_0_short_conv_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_top_down_layers_0_0_short_conv_activate = torch.nn.ReLU(inplace=True)
        self.neck_top_down_layers_0_0_main_conv_conv = torch.nn.Conv2d(160, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_top_down_layers_0_0_main_conv_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_top_down_layers_0_0_main_conv_activate = torch.nn.ReLU(inplace=True)
        self.neck_top_down_layers_0_0_blocks_0_conv1_conv = torch.nn.Conv2d(40, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_top_down_layers_0_0_blocks_0_conv1_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_top_down_layers_0_0_blocks_0_conv1_activate = torch.nn.ReLU(inplace=True)
        self.neck_top_down_layers_0_0_blocks_0_conv2_conv = torch.nn.Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.neck_top_down_layers_0_0_blocks_0_conv2_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_top_down_layers_0_0_blocks_0_conv2_activate = torch.nn.ReLU(inplace=True)
        self.neck_top_down_layers_0_0_final_conv_conv = torch.nn.Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_top_down_layers_0_0_final_conv_bn = torch.nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_top_down_layers_0_0_final_conv_activate = torch.nn.ReLU(inplace=True)
        self.neck_top_down_layers_0_1_conv = torch.nn.Conv2d(80, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_top_down_layers_0_1_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_top_down_layers_0_1_activate = torch.nn.ReLU(inplace=True)
        self.neck_upsample_layers_1 = torch.nn.Upsample(scale_factor=2.0)
        self.neck_top_down_layers_1_short_conv_conv = torch.nn.Conv2d(80, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_top_down_layers_1_short_conv_bn = torch.nn.BatchNorm2d(20, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_top_down_layers_1_short_conv_activate = torch.nn.ReLU(inplace=True)
        self.neck_top_down_layers_1_main_conv_conv = torch.nn.Conv2d(80, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_top_down_layers_1_main_conv_bn = torch.nn.BatchNorm2d(20, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_top_down_layers_1_main_conv_activate = torch.nn.ReLU(inplace=True)
        self.neck_top_down_layers_1_blocks_0_conv1_conv = torch.nn.Conv2d(20, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_top_down_layers_1_blocks_0_conv1_bn = torch.nn.BatchNorm2d(20, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_top_down_layers_1_blocks_0_conv1_activate = torch.nn.ReLU(inplace=True)
        self.neck_top_down_layers_1_blocks_0_conv2_conv = torch.nn.Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.neck_top_down_layers_1_blocks_0_conv2_bn = torch.nn.BatchNorm2d(20, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_top_down_layers_1_blocks_0_conv2_activate = torch.nn.ReLU(inplace=True)
        self.neck_top_down_layers_1_final_conv_conv = torch.nn.Conv2d(40, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_top_down_layers_1_final_conv_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_top_down_layers_1_final_conv_activate = torch.nn.ReLU(inplace=True)
        self.neck_downsample_layers_0_conv = torch.nn.Conv2d(40, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.neck_downsample_layers_0_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_downsample_layers_0_activate = torch.nn.ReLU(inplace=True)
        self.neck_bottom_up_layers_0_short_conv_conv = torch.nn.Conv2d(80, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_bottom_up_layers_0_short_conv_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_bottom_up_layers_0_short_conv_activate = torch.nn.ReLU(inplace=True)
        self.neck_bottom_up_layers_0_main_conv_conv = torch.nn.Conv2d(80, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_bottom_up_layers_0_main_conv_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_bottom_up_layers_0_main_conv_activate = torch.nn.ReLU(inplace=True)
        self.neck_bottom_up_layers_0_blocks_0_conv1_conv = torch.nn.Conv2d(40, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_bottom_up_layers_0_blocks_0_conv1_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_bottom_up_layers_0_blocks_0_conv1_activate = torch.nn.ReLU(inplace=True)
        self.neck_bottom_up_layers_0_blocks_0_conv2_conv = torch.nn.Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.neck_bottom_up_layers_0_blocks_0_conv2_bn = torch.nn.BatchNorm2d(40, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_bottom_up_layers_0_blocks_0_conv2_activate = torch.nn.ReLU(inplace=True)
        self.neck_bottom_up_layers_0_final_conv_conv = torch.nn.Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_bottom_up_layers_0_final_conv_bn = torch.nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_bottom_up_layers_0_final_conv_activate = torch.nn.ReLU(inplace=True)
        self.neck_downsample_layers_1_conv = torch.nn.Conv2d(80, 80, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.neck_downsample_layers_1_bn = torch.nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_downsample_layers_1_activate = torch.nn.ReLU(inplace=True)
        self.neck_bottom_up_layers_1_short_conv_conv = torch.nn.Conv2d(160, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_bottom_up_layers_1_short_conv_bn = torch.nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_bottom_up_layers_1_short_conv_activate = torch.nn.ReLU(inplace=True)
        self.neck_bottom_up_layers_1_main_conv_conv = torch.nn.Conv2d(160, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_bottom_up_layers_1_main_conv_bn = torch.nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_bottom_up_layers_1_main_conv_activate = torch.nn.ReLU(inplace=True)
        self.neck_bottom_up_layers_1_blocks_0_conv1_conv = torch.nn.Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_bottom_up_layers_1_blocks_0_conv1_bn = torch.nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_bottom_up_layers_1_blocks_0_conv1_activate = torch.nn.ReLU(inplace=True)
        self.neck_bottom_up_layers_1_blocks_0_conv2_conv = torch.nn.Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.neck_bottom_up_layers_1_blocks_0_conv2_bn = torch.nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_bottom_up_layers_1_blocks_0_conv2_activate = torch.nn.ReLU(inplace=True)
        self.neck_bottom_up_layers_1_final_conv_conv = torch.nn.Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.neck_bottom_up_layers_1_final_conv_bn = torch.nn.BatchNorm2d(160, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.neck_bottom_up_layers_1_final_conv_activate = torch.nn.ReLU(inplace=True)
        self.neck_out_layers_0 = torch.nn.Identity()
        self.neck_out_layers_1 = torch.nn.Identity()
        self.neck_out_layers_2 = torch.nn.Identity()
        self.bbox_head_head_module_convs_pred_0 = torch.nn.Conv2d(40, 48, kernel_size=(1, 1), stride=(1, 1))
        self.bbox_head_head_module_convs_pred_1 = torch.nn.Conv2d(80, 48, kernel_size=(1, 1), stride=(1, 1))
        self.bbox_head_head_module_convs_pred_2 = torch.nn.Conv2d(160, 48, kernel_size=(1, 1), stride=(1, 1))
        self.fake_quant_11 = torch.quantization.QuantStub()
        self.fake_quant_12 = torch.quantization.QuantStub()
        self.rewritten_sigmoid_0 = torch.nn.Sigmoid()
        self.fake_quant_4 = torch.quantization.QuantStub()
        self.fake_dequant_inner_0_0_0 = torch.quantization.DeQuantStub()
        self.fake_quant_inner_2_0_0 = torch.quantization.QuantStub()
        self.fake_quant_13 = torch.quantization.QuantStub()
        self.fake_quant_14 = torch.quantization.QuantStub()
        self.rewritten_sigmoid_1 = torch.nn.Sigmoid()
        self.fake_quant_7 = torch.quantization.QuantStub()
        self.fake_dequant_inner_1_0_0 = torch.quantization.DeQuantStub()
        self.fake_quant_inner_1_0_0 = torch.quantization.QuantStub()
        self.fake_quant_15 = torch.quantization.QuantStub()
        self.fake_quant_16 = torch.quantization.QuantStub()
        self.rewritten_sigmoid_2 = torch.nn.Sigmoid()
        self.fake_quant_10 = torch.quantization.QuantStub()
        self.fake_dequant_inner_2_0_0 = torch.quantization.DeQuantStub()
        self.fake_quant_inner_0_0_0 = torch.quantization.QuantStub()
        self.fake_dequant_0 = torch.quantization.DeQuantStub()
        self.float_functional_simple_0 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_1 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_2 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_3 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_4 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_5 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_6 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_7 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_8 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_9 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_10 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_11 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_12 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_13 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_14 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_15 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_16 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_17 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_18 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_19 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_20 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_21 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_22 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_23 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_24 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_25 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_26 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_27 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_28 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_29 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_30 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_31 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_32 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_33 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_34 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_35 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_36 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_37 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_38 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_39 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_40 = torch.ao.nn.quantized.FloatFunctional()
        self.float_functional_simple_41 = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, input_0_f):
        fake_quant_0 = self.fake_quant_0(input_0_f)
        input_0_f = None
        backbone_stem_conv = self.backbone_stem_conv(fake_quant_0)
        fake_quant_0 = None
        backbone_stem_bn = self.backbone_stem_bn(backbone_stem_conv)
        backbone_stem_conv = None
        backbone_stem_activate = self.backbone_stem_activate(backbone_stem_bn)
        backbone_stem_bn = None
        backbone_stage1_0_conv = self.backbone_stage1_0_conv(backbone_stem_activate)
        backbone_stem_activate = None
        backbone_stage1_0_bn = self.backbone_stage1_0_bn(backbone_stage1_0_conv)
        backbone_stage1_0_conv = None
        backbone_stage1_0_activate = self.backbone_stage1_0_activate(backbone_stage1_0_bn)
        backbone_stage1_0_bn = None
        backbone_stage1_1_short_conv_conv = self.backbone_stage1_1_short_conv_conv(backbone_stage1_0_activate)
        backbone_stage1_1_short_conv_bn = self.backbone_stage1_1_short_conv_bn(backbone_stage1_1_short_conv_conv)
        backbone_stage1_1_short_conv_conv = None
        backbone_stage1_1_short_conv_activate = self.backbone_stage1_1_short_conv_activate(backbone_stage1_1_short_conv_bn)
        backbone_stage1_1_short_conv_bn = None
        backbone_stage1_1_main_conv_conv = self.backbone_stage1_1_main_conv_conv(backbone_stage1_0_activate)
        backbone_stage1_0_activate = None
        backbone_stage1_1_main_conv_bn = self.backbone_stage1_1_main_conv_bn(backbone_stage1_1_main_conv_conv)
        backbone_stage1_1_main_conv_conv = None
        backbone_stage1_1_main_conv_activate = self.backbone_stage1_1_main_conv_activate(backbone_stage1_1_main_conv_bn)
        backbone_stage1_1_main_conv_bn = None
        backbone_stage1_1_blocks_0_conv1_conv = self.backbone_stage1_1_blocks_0_conv1_conv(backbone_stage1_1_main_conv_activate)
        backbone_stage1_1_blocks_0_conv1_bn = self.backbone_stage1_1_blocks_0_conv1_bn(backbone_stage1_1_blocks_0_conv1_conv)
        backbone_stage1_1_blocks_0_conv1_conv = None
        backbone_stage1_1_blocks_0_conv1_activate = self.backbone_stage1_1_blocks_0_conv1_activate(backbone_stage1_1_blocks_0_conv1_bn)
        backbone_stage1_1_blocks_0_conv1_bn = None
        backbone_stage1_1_blocks_0_conv2_conv = self.backbone_stage1_1_blocks_0_conv2_conv(backbone_stage1_1_blocks_0_conv1_activate)
        backbone_stage1_1_blocks_0_conv1_activate = None
        backbone_stage1_1_blocks_0_conv2_bn = self.backbone_stage1_1_blocks_0_conv2_bn(backbone_stage1_1_blocks_0_conv2_conv)
        backbone_stage1_1_blocks_0_conv2_conv = None
        backbone_stage1_1_blocks_0_conv2_activate = self.backbone_stage1_1_blocks_0_conv2_activate(backbone_stage1_1_blocks_0_conv2_bn)
        backbone_stage1_1_blocks_0_conv2_bn = None
        add_0_f = self.float_functional_simple_0.add(backbone_stage1_1_blocks_0_conv2_activate, backbone_stage1_1_main_conv_activate)
        backbone_stage1_1_blocks_0_conv2_activate = None
        backbone_stage1_1_main_conv_activate = None
        cat_0_f = self.float_functional_simple_1.cat([add_0_f, backbone_stage1_1_short_conv_activate], dim=1)
        add_0_f = None
        backbone_stage1_1_short_conv_activate = None
        backbone_stage1_1_final_conv_conv = self.backbone_stage1_1_final_conv_conv(cat_0_f)
        cat_0_f = None
        backbone_stage1_1_final_conv_bn = self.backbone_stage1_1_final_conv_bn(backbone_stage1_1_final_conv_conv)
        backbone_stage1_1_final_conv_conv = None
        backbone_stage1_1_final_conv_activate = self.backbone_stage1_1_final_conv_activate(backbone_stage1_1_final_conv_bn)
        backbone_stage1_1_final_conv_bn = None
        backbone_stage2_0_conv = self.backbone_stage2_0_conv(backbone_stage1_1_final_conv_activate)
        backbone_stage1_1_final_conv_activate = None
        backbone_stage2_0_bn = self.backbone_stage2_0_bn(backbone_stage2_0_conv)
        backbone_stage2_0_conv = None
        backbone_stage2_0_activate = self.backbone_stage2_0_activate(backbone_stage2_0_bn)
        backbone_stage2_0_bn = None
        backbone_stage2_1_short_conv_conv = self.backbone_stage2_1_short_conv_conv(backbone_stage2_0_activate)
        backbone_stage2_1_short_conv_bn = self.backbone_stage2_1_short_conv_bn(backbone_stage2_1_short_conv_conv)
        backbone_stage2_1_short_conv_conv = None
        backbone_stage2_1_short_conv_activate = self.backbone_stage2_1_short_conv_activate(backbone_stage2_1_short_conv_bn)
        backbone_stage2_1_short_conv_bn = None
        backbone_stage2_1_main_conv_conv = self.backbone_stage2_1_main_conv_conv(backbone_stage2_0_activate)
        backbone_stage2_0_activate = None
        backbone_stage2_1_main_conv_bn = self.backbone_stage2_1_main_conv_bn(backbone_stage2_1_main_conv_conv)
        backbone_stage2_1_main_conv_conv = None
        backbone_stage2_1_main_conv_activate = self.backbone_stage2_1_main_conv_activate(backbone_stage2_1_main_conv_bn)
        backbone_stage2_1_main_conv_bn = None
        backbone_stage2_1_blocks_0_conv1_conv = self.backbone_stage2_1_blocks_0_conv1_conv(backbone_stage2_1_main_conv_activate)
        backbone_stage2_1_blocks_0_conv1_bn = self.backbone_stage2_1_blocks_0_conv1_bn(backbone_stage2_1_blocks_0_conv1_conv)
        backbone_stage2_1_blocks_0_conv1_conv = None
        backbone_stage2_1_blocks_0_conv1_activate = self.backbone_stage2_1_blocks_0_conv1_activate(backbone_stage2_1_blocks_0_conv1_bn)
        backbone_stage2_1_blocks_0_conv1_bn = None
        backbone_stage2_1_blocks_0_conv2_conv = self.backbone_stage2_1_blocks_0_conv2_conv(backbone_stage2_1_blocks_0_conv1_activate)
        backbone_stage2_1_blocks_0_conv1_activate = None
        backbone_stage2_1_blocks_0_conv2_bn = self.backbone_stage2_1_blocks_0_conv2_bn(backbone_stage2_1_blocks_0_conv2_conv)
        backbone_stage2_1_blocks_0_conv2_conv = None
        backbone_stage2_1_blocks_0_conv2_activate = self.backbone_stage2_1_blocks_0_conv2_activate(backbone_stage2_1_blocks_0_conv2_bn)
        backbone_stage2_1_blocks_0_conv2_bn = None
        add_1_f = self.float_functional_simple_2.add(backbone_stage2_1_blocks_0_conv2_activate, backbone_stage2_1_main_conv_activate)
        backbone_stage2_1_blocks_0_conv2_activate = None
        backbone_stage2_1_main_conv_activate = None
        backbone_stage2_1_blocks_1_conv1_conv = self.backbone_stage2_1_blocks_1_conv1_conv(add_1_f)
        backbone_stage2_1_blocks_1_conv1_bn = self.backbone_stage2_1_blocks_1_conv1_bn(backbone_stage2_1_blocks_1_conv1_conv)
        backbone_stage2_1_blocks_1_conv1_conv = None
        backbone_stage2_1_blocks_1_conv1_activate = self.backbone_stage2_1_blocks_1_conv1_activate(backbone_stage2_1_blocks_1_conv1_bn)
        backbone_stage2_1_blocks_1_conv1_bn = None
        backbone_stage2_1_blocks_1_conv2_conv = self.backbone_stage2_1_blocks_1_conv2_conv(backbone_stage2_1_blocks_1_conv1_activate)
        backbone_stage2_1_blocks_1_conv1_activate = None
        backbone_stage2_1_blocks_1_conv2_bn = self.backbone_stage2_1_blocks_1_conv2_bn(backbone_stage2_1_blocks_1_conv2_conv)
        backbone_stage2_1_blocks_1_conv2_conv = None
        backbone_stage2_1_blocks_1_conv2_activate = self.backbone_stage2_1_blocks_1_conv2_activate(backbone_stage2_1_blocks_1_conv2_bn)
        backbone_stage2_1_blocks_1_conv2_bn = None
        add_2_f = self.float_functional_simple_3.add(backbone_stage2_1_blocks_1_conv2_activate, add_1_f)
        backbone_stage2_1_blocks_1_conv2_activate = None
        add_1_f = None
        cat_1_f = self.float_functional_simple_4.cat([add_2_f, backbone_stage2_1_short_conv_activate], dim=1)
        add_2_f = None
        backbone_stage2_1_short_conv_activate = None
        backbone_stage2_1_final_conv_conv = self.backbone_stage2_1_final_conv_conv(cat_1_f)
        cat_1_f = None
        backbone_stage2_1_final_conv_bn = self.backbone_stage2_1_final_conv_bn(backbone_stage2_1_final_conv_conv)
        backbone_stage2_1_final_conv_conv = None
        backbone_stage2_1_final_conv_activate = self.backbone_stage2_1_final_conv_activate(backbone_stage2_1_final_conv_bn)
        backbone_stage2_1_final_conv_bn = None
        backbone_stage3_0_conv = self.backbone_stage3_0_conv(backbone_stage2_1_final_conv_activate)
        backbone_stage3_0_bn = self.backbone_stage3_0_bn(backbone_stage3_0_conv)
        backbone_stage3_0_conv = None
        backbone_stage3_0_activate = self.backbone_stage3_0_activate(backbone_stage3_0_bn)
        backbone_stage3_0_bn = None
        backbone_stage3_1_short_conv_conv = self.backbone_stage3_1_short_conv_conv(backbone_stage3_0_activate)
        backbone_stage3_1_short_conv_bn = self.backbone_stage3_1_short_conv_bn(backbone_stage3_1_short_conv_conv)
        backbone_stage3_1_short_conv_conv = None
        backbone_stage3_1_short_conv_activate = self.backbone_stage3_1_short_conv_activate(backbone_stage3_1_short_conv_bn)
        backbone_stage3_1_short_conv_bn = None
        backbone_stage3_1_main_conv_conv = self.backbone_stage3_1_main_conv_conv(backbone_stage3_0_activate)
        backbone_stage3_0_activate = None
        backbone_stage3_1_main_conv_bn = self.backbone_stage3_1_main_conv_bn(backbone_stage3_1_main_conv_conv)
        backbone_stage3_1_main_conv_conv = None
        backbone_stage3_1_main_conv_activate = self.backbone_stage3_1_main_conv_activate(backbone_stage3_1_main_conv_bn)
        backbone_stage3_1_main_conv_bn = None
        backbone_stage3_1_blocks_0_conv1_conv = self.backbone_stage3_1_blocks_0_conv1_conv(backbone_stage3_1_main_conv_activate)
        backbone_stage3_1_blocks_0_conv1_bn = self.backbone_stage3_1_blocks_0_conv1_bn(backbone_stage3_1_blocks_0_conv1_conv)
        backbone_stage3_1_blocks_0_conv1_conv = None
        backbone_stage3_1_blocks_0_conv1_activate = self.backbone_stage3_1_blocks_0_conv1_activate(backbone_stage3_1_blocks_0_conv1_bn)
        backbone_stage3_1_blocks_0_conv1_bn = None
        backbone_stage3_1_blocks_0_conv2_conv = self.backbone_stage3_1_blocks_0_conv2_conv(backbone_stage3_1_blocks_0_conv1_activate)
        backbone_stage3_1_blocks_0_conv1_activate = None
        backbone_stage3_1_blocks_0_conv2_bn = self.backbone_stage3_1_blocks_0_conv2_bn(backbone_stage3_1_blocks_0_conv2_conv)
        backbone_stage3_1_blocks_0_conv2_conv = None
        backbone_stage3_1_blocks_0_conv2_activate = self.backbone_stage3_1_blocks_0_conv2_activate(backbone_stage3_1_blocks_0_conv2_bn)
        backbone_stage3_1_blocks_0_conv2_bn = None
        add_3_f = self.float_functional_simple_5.add(backbone_stage3_1_blocks_0_conv2_activate, backbone_stage3_1_main_conv_activate)
        backbone_stage3_1_blocks_0_conv2_activate = None
        backbone_stage3_1_main_conv_activate = None
        backbone_stage3_1_blocks_1_conv1_conv = self.backbone_stage3_1_blocks_1_conv1_conv(add_3_f)
        backbone_stage3_1_blocks_1_conv1_bn = self.backbone_stage3_1_blocks_1_conv1_bn(backbone_stage3_1_blocks_1_conv1_conv)
        backbone_stage3_1_blocks_1_conv1_conv = None
        backbone_stage3_1_blocks_1_conv1_activate = self.backbone_stage3_1_blocks_1_conv1_activate(backbone_stage3_1_blocks_1_conv1_bn)
        backbone_stage3_1_blocks_1_conv1_bn = None
        backbone_stage3_1_blocks_1_conv2_conv = self.backbone_stage3_1_blocks_1_conv2_conv(backbone_stage3_1_blocks_1_conv1_activate)
        backbone_stage3_1_blocks_1_conv1_activate = None
        backbone_stage3_1_blocks_1_conv2_bn = self.backbone_stage3_1_blocks_1_conv2_bn(backbone_stage3_1_blocks_1_conv2_conv)
        backbone_stage3_1_blocks_1_conv2_conv = None
        backbone_stage3_1_blocks_1_conv2_activate = self.backbone_stage3_1_blocks_1_conv2_activate(backbone_stage3_1_blocks_1_conv2_bn)
        backbone_stage3_1_blocks_1_conv2_bn = None
        add_4_f = self.float_functional_simple_6.add(backbone_stage3_1_blocks_1_conv2_activate, add_3_f)
        backbone_stage3_1_blocks_1_conv2_activate = None
        add_3_f = None
        backbone_stage3_1_blocks_2_conv1_conv = self.backbone_stage3_1_blocks_2_conv1_conv(add_4_f)
        backbone_stage3_1_blocks_2_conv1_bn = self.backbone_stage3_1_blocks_2_conv1_bn(backbone_stage3_1_blocks_2_conv1_conv)
        backbone_stage3_1_blocks_2_conv1_conv = None
        backbone_stage3_1_blocks_2_conv1_activate = self.backbone_stage3_1_blocks_2_conv1_activate(backbone_stage3_1_blocks_2_conv1_bn)
        backbone_stage3_1_blocks_2_conv1_bn = None
        backbone_stage3_1_blocks_2_conv2_conv = self.backbone_stage3_1_blocks_2_conv2_conv(backbone_stage3_1_blocks_2_conv1_activate)
        backbone_stage3_1_blocks_2_conv1_activate = None
        backbone_stage3_1_blocks_2_conv2_bn = self.backbone_stage3_1_blocks_2_conv2_bn(backbone_stage3_1_blocks_2_conv2_conv)
        backbone_stage3_1_blocks_2_conv2_conv = None
        backbone_stage3_1_blocks_2_conv2_activate = self.backbone_stage3_1_blocks_2_conv2_activate(backbone_stage3_1_blocks_2_conv2_bn)
        backbone_stage3_1_blocks_2_conv2_bn = None
        add_5_f = self.float_functional_simple_7.add(backbone_stage3_1_blocks_2_conv2_activate, add_4_f)
        backbone_stage3_1_blocks_2_conv2_activate = None
        add_4_f = None
        cat_2_f = self.float_functional_simple_8.cat([add_5_f, backbone_stage3_1_short_conv_activate], dim=1)
        add_5_f = None
        backbone_stage3_1_short_conv_activate = None
        backbone_stage3_1_final_conv_conv = self.backbone_stage3_1_final_conv_conv(cat_2_f)
        cat_2_f = None
        backbone_stage3_1_final_conv_bn = self.backbone_stage3_1_final_conv_bn(backbone_stage3_1_final_conv_conv)
        backbone_stage3_1_final_conv_conv = None
        backbone_stage3_1_final_conv_activate = self.backbone_stage3_1_final_conv_activate(backbone_stage3_1_final_conv_bn)
        backbone_stage3_1_final_conv_bn = None
        backbone_stage4_0_conv = self.backbone_stage4_0_conv(backbone_stage3_1_final_conv_activate)
        backbone_stage4_0_bn = self.backbone_stage4_0_bn(backbone_stage4_0_conv)
        backbone_stage4_0_conv = None
        backbone_stage4_0_activate = self.backbone_stage4_0_activate(backbone_stage4_0_bn)
        backbone_stage4_0_bn = None
        backbone_stage4_1_short_conv_conv = self.backbone_stage4_1_short_conv_conv(backbone_stage4_0_activate)
        backbone_stage4_1_short_conv_bn = self.backbone_stage4_1_short_conv_bn(backbone_stage4_1_short_conv_conv)
        backbone_stage4_1_short_conv_conv = None
        backbone_stage4_1_short_conv_activate = self.backbone_stage4_1_short_conv_activate(backbone_stage4_1_short_conv_bn)
        backbone_stage4_1_short_conv_bn = None
        backbone_stage4_1_main_conv_conv = self.backbone_stage4_1_main_conv_conv(backbone_stage4_0_activate)
        backbone_stage4_0_activate = None
        backbone_stage4_1_main_conv_bn = self.backbone_stage4_1_main_conv_bn(backbone_stage4_1_main_conv_conv)
        backbone_stage4_1_main_conv_conv = None
        backbone_stage4_1_main_conv_activate = self.backbone_stage4_1_main_conv_activate(backbone_stage4_1_main_conv_bn)
        backbone_stage4_1_main_conv_bn = None
        backbone_stage4_1_blocks_0_conv1_conv = self.backbone_stage4_1_blocks_0_conv1_conv(backbone_stage4_1_main_conv_activate)
        backbone_stage4_1_blocks_0_conv1_bn = self.backbone_stage4_1_blocks_0_conv1_bn(backbone_stage4_1_blocks_0_conv1_conv)
        backbone_stage4_1_blocks_0_conv1_conv = None
        backbone_stage4_1_blocks_0_conv1_activate = self.backbone_stage4_1_blocks_0_conv1_activate(backbone_stage4_1_blocks_0_conv1_bn)
        backbone_stage4_1_blocks_0_conv1_bn = None
        backbone_stage4_1_blocks_0_conv2_conv = self.backbone_stage4_1_blocks_0_conv2_conv(backbone_stage4_1_blocks_0_conv1_activate)
        backbone_stage4_1_blocks_0_conv1_activate = None
        backbone_stage4_1_blocks_0_conv2_bn = self.backbone_stage4_1_blocks_0_conv2_bn(backbone_stage4_1_blocks_0_conv2_conv)
        backbone_stage4_1_blocks_0_conv2_conv = None
        backbone_stage4_1_blocks_0_conv2_activate = self.backbone_stage4_1_blocks_0_conv2_activate(backbone_stage4_1_blocks_0_conv2_bn)
        backbone_stage4_1_blocks_0_conv2_bn = None
        add_6_f = self.float_functional_simple_9.add(backbone_stage4_1_blocks_0_conv2_activate, backbone_stage4_1_main_conv_activate)
        backbone_stage4_1_blocks_0_conv2_activate = None
        backbone_stage4_1_main_conv_activate = None
        cat_3_f = self.float_functional_simple_10.cat([add_6_f, backbone_stage4_1_short_conv_activate], dim=1)
        add_6_f = None
        backbone_stage4_1_short_conv_activate = None
        backbone_stage4_1_final_conv_conv = self.backbone_stage4_1_final_conv_conv(cat_3_f)
        cat_3_f = None
        backbone_stage4_1_final_conv_bn = self.backbone_stage4_1_final_conv_bn(backbone_stage4_1_final_conv_conv)
        backbone_stage4_1_final_conv_conv = None
        backbone_stage4_1_final_conv_activate = self.backbone_stage4_1_final_conv_activate(backbone_stage4_1_final_conv_bn)
        backbone_stage4_1_final_conv_bn = None
        backbone_stage4_2_conv1_conv = self.backbone_stage4_2_conv1_conv(backbone_stage4_1_final_conv_activate)
        backbone_stage4_1_final_conv_activate = None
        backbone_stage4_2_conv1_bn = self.backbone_stage4_2_conv1_bn(backbone_stage4_2_conv1_conv)
        backbone_stage4_2_conv1_conv = None
        backbone_stage4_2_conv1_activate = self.backbone_stage4_2_conv1_activate(backbone_stage4_2_conv1_bn)
        backbone_stage4_2_conv1_bn = None
        backbone_stage4_2_poolings = self.backbone_stage4_2_poolings(backbone_stage4_2_conv1_activate)
        backbone_stage4_2_poolings_1 = self.backbone_stage4_2_poolings(backbone_stage4_2_poolings)
        backbone_stage4_2_poolings_2 = self.backbone_stage4_2_poolings(backbone_stage4_2_poolings_1)
        cat_4_f = self.float_functional_simple_11.cat([backbone_stage4_2_conv1_activate, backbone_stage4_2_poolings, backbone_stage4_2_poolings_1, backbone_stage4_2_poolings_2], dim=1)
        backbone_stage4_2_conv1_activate = None
        backbone_stage4_2_poolings = None
        backbone_stage4_2_poolings_1 = None
        backbone_stage4_2_poolings_2 = None
        backbone_stage4_2_conv2_conv = self.backbone_stage4_2_conv2_conv(cat_4_f)
        cat_4_f = None
        backbone_stage4_2_conv2_bn = self.backbone_stage4_2_conv2_bn(backbone_stage4_2_conv2_conv)
        backbone_stage4_2_conv2_conv = None
        backbone_stage4_2_conv2_activate = self.backbone_stage4_2_conv2_activate(backbone_stage4_2_conv2_bn)
        backbone_stage4_2_conv2_bn = None
        neck_reduce_layers_0 = self.neck_reduce_layers_0(backbone_stage2_1_final_conv_activate)
        backbone_stage2_1_final_conv_activate = None
        neck_reduce_layers_1 = self.neck_reduce_layers_1(backbone_stage3_1_final_conv_activate)
        backbone_stage3_1_final_conv_activate = None
        neck_reduce_layers_2_conv = self.neck_reduce_layers_2_conv(backbone_stage4_2_conv2_activate)
        backbone_stage4_2_conv2_activate = None
        neck_reduce_layers_2_bn = self.neck_reduce_layers_2_bn(neck_reduce_layers_2_conv)
        neck_reduce_layers_2_conv = None
        neck_reduce_layers_2_activate = self.neck_reduce_layers_2_activate(neck_reduce_layers_2_bn)
        neck_reduce_layers_2_bn = None
        neck_upsample_layers_0 = self.neck_upsample_layers_0(neck_reduce_layers_2_activate)
        cat_5_f = self.float_functional_simple_12.cat([neck_upsample_layers_0, neck_reduce_layers_1], 1)
        neck_upsample_layers_0 = None
        neck_reduce_layers_1 = None
        neck_top_down_layers_0_0_short_conv_conv = self.neck_top_down_layers_0_0_short_conv_conv(cat_5_f)
        neck_top_down_layers_0_0_short_conv_bn = self.neck_top_down_layers_0_0_short_conv_bn(neck_top_down_layers_0_0_short_conv_conv)
        neck_top_down_layers_0_0_short_conv_conv = None
        neck_top_down_layers_0_0_short_conv_activate = self.neck_top_down_layers_0_0_short_conv_activate(neck_top_down_layers_0_0_short_conv_bn)
        neck_top_down_layers_0_0_short_conv_bn = None
        neck_top_down_layers_0_0_main_conv_conv = self.neck_top_down_layers_0_0_main_conv_conv(cat_5_f)
        cat_5_f = None
        neck_top_down_layers_0_0_main_conv_bn = self.neck_top_down_layers_0_0_main_conv_bn(neck_top_down_layers_0_0_main_conv_conv)
        neck_top_down_layers_0_0_main_conv_conv = None
        neck_top_down_layers_0_0_main_conv_activate = self.neck_top_down_layers_0_0_main_conv_activate(neck_top_down_layers_0_0_main_conv_bn)
        neck_top_down_layers_0_0_main_conv_bn = None
        neck_top_down_layers_0_0_blocks_0_conv1_conv = self.neck_top_down_layers_0_0_blocks_0_conv1_conv(neck_top_down_layers_0_0_main_conv_activate)
        neck_top_down_layers_0_0_main_conv_activate = None
        neck_top_down_layers_0_0_blocks_0_conv1_bn = self.neck_top_down_layers_0_0_blocks_0_conv1_bn(neck_top_down_layers_0_0_blocks_0_conv1_conv)
        neck_top_down_layers_0_0_blocks_0_conv1_conv = None
        neck_top_down_layers_0_0_blocks_0_conv1_activate = self.neck_top_down_layers_0_0_blocks_0_conv1_activate(neck_top_down_layers_0_0_blocks_0_conv1_bn)
        neck_top_down_layers_0_0_blocks_0_conv1_bn = None
        neck_top_down_layers_0_0_blocks_0_conv2_conv = self.neck_top_down_layers_0_0_blocks_0_conv2_conv(neck_top_down_layers_0_0_blocks_0_conv1_activate)
        neck_top_down_layers_0_0_blocks_0_conv1_activate = None
        neck_top_down_layers_0_0_blocks_0_conv2_bn = self.neck_top_down_layers_0_0_blocks_0_conv2_bn(neck_top_down_layers_0_0_blocks_0_conv2_conv)
        neck_top_down_layers_0_0_blocks_0_conv2_conv = None
        neck_top_down_layers_0_0_blocks_0_conv2_activate = self.neck_top_down_layers_0_0_blocks_0_conv2_activate(neck_top_down_layers_0_0_blocks_0_conv2_bn)
        neck_top_down_layers_0_0_blocks_0_conv2_bn = None
        cat_6_f = self.float_functional_simple_13.cat([neck_top_down_layers_0_0_blocks_0_conv2_activate, neck_top_down_layers_0_0_short_conv_activate], dim=1)
        neck_top_down_layers_0_0_blocks_0_conv2_activate = None
        neck_top_down_layers_0_0_short_conv_activate = None
        neck_top_down_layers_0_0_final_conv_conv = self.neck_top_down_layers_0_0_final_conv_conv(cat_6_f)
        cat_6_f = None
        neck_top_down_layers_0_0_final_conv_bn = self.neck_top_down_layers_0_0_final_conv_bn(neck_top_down_layers_0_0_final_conv_conv)
        neck_top_down_layers_0_0_final_conv_conv = None
        neck_top_down_layers_0_0_final_conv_activate = self.neck_top_down_layers_0_0_final_conv_activate(neck_top_down_layers_0_0_final_conv_bn)
        neck_top_down_layers_0_0_final_conv_bn = None
        neck_top_down_layers_0_1_conv = self.neck_top_down_layers_0_1_conv(neck_top_down_layers_0_0_final_conv_activate)
        neck_top_down_layers_0_0_final_conv_activate = None
        neck_top_down_layers_0_1_bn = self.neck_top_down_layers_0_1_bn(neck_top_down_layers_0_1_conv)
        neck_top_down_layers_0_1_conv = None
        neck_top_down_layers_0_1_activate = self.neck_top_down_layers_0_1_activate(neck_top_down_layers_0_1_bn)
        neck_top_down_layers_0_1_bn = None
        neck_upsample_layers_1 = self.neck_upsample_layers_1(neck_top_down_layers_0_1_activate)
        cat_7_f = self.float_functional_simple_14.cat([neck_upsample_layers_1, neck_reduce_layers_0], 1)
        neck_upsample_layers_1 = None
        neck_reduce_layers_0 = None
        neck_top_down_layers_1_short_conv_conv = self.neck_top_down_layers_1_short_conv_conv(cat_7_f)
        neck_top_down_layers_1_short_conv_bn = self.neck_top_down_layers_1_short_conv_bn(neck_top_down_layers_1_short_conv_conv)
        neck_top_down_layers_1_short_conv_conv = None
        neck_top_down_layers_1_short_conv_activate = self.neck_top_down_layers_1_short_conv_activate(neck_top_down_layers_1_short_conv_bn)
        neck_top_down_layers_1_short_conv_bn = None
        neck_top_down_layers_1_main_conv_conv = self.neck_top_down_layers_1_main_conv_conv(cat_7_f)
        cat_7_f = None
        neck_top_down_layers_1_main_conv_bn = self.neck_top_down_layers_1_main_conv_bn(neck_top_down_layers_1_main_conv_conv)
        neck_top_down_layers_1_main_conv_conv = None
        neck_top_down_layers_1_main_conv_activate = self.neck_top_down_layers_1_main_conv_activate(neck_top_down_layers_1_main_conv_bn)
        neck_top_down_layers_1_main_conv_bn = None
        neck_top_down_layers_1_blocks_0_conv1_conv = self.neck_top_down_layers_1_blocks_0_conv1_conv(neck_top_down_layers_1_main_conv_activate)
        neck_top_down_layers_1_main_conv_activate = None
        neck_top_down_layers_1_blocks_0_conv1_bn = self.neck_top_down_layers_1_blocks_0_conv1_bn(neck_top_down_layers_1_blocks_0_conv1_conv)
        neck_top_down_layers_1_blocks_0_conv1_conv = None
        neck_top_down_layers_1_blocks_0_conv1_activate = self.neck_top_down_layers_1_blocks_0_conv1_activate(neck_top_down_layers_1_blocks_0_conv1_bn)
        neck_top_down_layers_1_blocks_0_conv1_bn = None
        neck_top_down_layers_1_blocks_0_conv2_conv = self.neck_top_down_layers_1_blocks_0_conv2_conv(neck_top_down_layers_1_blocks_0_conv1_activate)
        neck_top_down_layers_1_blocks_0_conv1_activate = None
        neck_top_down_layers_1_blocks_0_conv2_bn = self.neck_top_down_layers_1_blocks_0_conv2_bn(neck_top_down_layers_1_blocks_0_conv2_conv)
        neck_top_down_layers_1_blocks_0_conv2_conv = None
        neck_top_down_layers_1_blocks_0_conv2_activate = self.neck_top_down_layers_1_blocks_0_conv2_activate(neck_top_down_layers_1_blocks_0_conv2_bn)
        neck_top_down_layers_1_blocks_0_conv2_bn = None
        cat_8_f = self.float_functional_simple_15.cat([neck_top_down_layers_1_blocks_0_conv2_activate, neck_top_down_layers_1_short_conv_activate], dim=1)
        neck_top_down_layers_1_blocks_0_conv2_activate = None
        neck_top_down_layers_1_short_conv_activate = None
        neck_top_down_layers_1_final_conv_conv = self.neck_top_down_layers_1_final_conv_conv(cat_8_f)
        cat_8_f = None
        neck_top_down_layers_1_final_conv_bn = self.neck_top_down_layers_1_final_conv_bn(neck_top_down_layers_1_final_conv_conv)
        neck_top_down_layers_1_final_conv_conv = None
        neck_top_down_layers_1_final_conv_activate = self.neck_top_down_layers_1_final_conv_activate(neck_top_down_layers_1_final_conv_bn)
        neck_top_down_layers_1_final_conv_bn = None
        neck_downsample_layers_0_conv = self.neck_downsample_layers_0_conv(neck_top_down_layers_1_final_conv_activate)
        neck_downsample_layers_0_bn = self.neck_downsample_layers_0_bn(neck_downsample_layers_0_conv)
        neck_downsample_layers_0_conv = None
        neck_downsample_layers_0_activate = self.neck_downsample_layers_0_activate(neck_downsample_layers_0_bn)
        neck_downsample_layers_0_bn = None
        cat_9_f = self.float_functional_simple_16.cat([neck_downsample_layers_0_activate, neck_top_down_layers_0_1_activate], 1)
        neck_downsample_layers_0_activate = None
        neck_top_down_layers_0_1_activate = None
        neck_bottom_up_layers_0_short_conv_conv = self.neck_bottom_up_layers_0_short_conv_conv(cat_9_f)
        neck_bottom_up_layers_0_short_conv_bn = self.neck_bottom_up_layers_0_short_conv_bn(neck_bottom_up_layers_0_short_conv_conv)
        neck_bottom_up_layers_0_short_conv_conv = None
        neck_bottom_up_layers_0_short_conv_activate = self.neck_bottom_up_layers_0_short_conv_activate(neck_bottom_up_layers_0_short_conv_bn)
        neck_bottom_up_layers_0_short_conv_bn = None
        neck_bottom_up_layers_0_main_conv_conv = self.neck_bottom_up_layers_0_main_conv_conv(cat_9_f)
        cat_9_f = None
        neck_bottom_up_layers_0_main_conv_bn = self.neck_bottom_up_layers_0_main_conv_bn(neck_bottom_up_layers_0_main_conv_conv)
        neck_bottom_up_layers_0_main_conv_conv = None
        neck_bottom_up_layers_0_main_conv_activate = self.neck_bottom_up_layers_0_main_conv_activate(neck_bottom_up_layers_0_main_conv_bn)
        neck_bottom_up_layers_0_main_conv_bn = None
        neck_bottom_up_layers_0_blocks_0_conv1_conv = self.neck_bottom_up_layers_0_blocks_0_conv1_conv(neck_bottom_up_layers_0_main_conv_activate)
        neck_bottom_up_layers_0_main_conv_activate = None
        neck_bottom_up_layers_0_blocks_0_conv1_bn = self.neck_bottom_up_layers_0_blocks_0_conv1_bn(neck_bottom_up_layers_0_blocks_0_conv1_conv)
        neck_bottom_up_layers_0_blocks_0_conv1_conv = None
        neck_bottom_up_layers_0_blocks_0_conv1_activate = self.neck_bottom_up_layers_0_blocks_0_conv1_activate(neck_bottom_up_layers_0_blocks_0_conv1_bn)
        neck_bottom_up_layers_0_blocks_0_conv1_bn = None
        neck_bottom_up_layers_0_blocks_0_conv2_conv = self.neck_bottom_up_layers_0_blocks_0_conv2_conv(neck_bottom_up_layers_0_blocks_0_conv1_activate)
        neck_bottom_up_layers_0_blocks_0_conv1_activate = None
        neck_bottom_up_layers_0_blocks_0_conv2_bn = self.neck_bottom_up_layers_0_blocks_0_conv2_bn(neck_bottom_up_layers_0_blocks_0_conv2_conv)
        neck_bottom_up_layers_0_blocks_0_conv2_conv = None
        neck_bottom_up_layers_0_blocks_0_conv2_activate = self.neck_bottom_up_layers_0_blocks_0_conv2_activate(neck_bottom_up_layers_0_blocks_0_conv2_bn)
        neck_bottom_up_layers_0_blocks_0_conv2_bn = None
        cat_10_f = self.float_functional_simple_17.cat([neck_bottom_up_layers_0_blocks_0_conv2_activate, neck_bottom_up_layers_0_short_conv_activate], dim=1)
        neck_bottom_up_layers_0_blocks_0_conv2_activate = None
        neck_bottom_up_layers_0_short_conv_activate = None
        neck_bottom_up_layers_0_final_conv_conv = self.neck_bottom_up_layers_0_final_conv_conv(cat_10_f)
        cat_10_f = None
        neck_bottom_up_layers_0_final_conv_bn = self.neck_bottom_up_layers_0_final_conv_bn(neck_bottom_up_layers_0_final_conv_conv)
        neck_bottom_up_layers_0_final_conv_conv = None
        neck_bottom_up_layers_0_final_conv_activate = self.neck_bottom_up_layers_0_final_conv_activate(neck_bottom_up_layers_0_final_conv_bn)
        neck_bottom_up_layers_0_final_conv_bn = None
        neck_downsample_layers_1_conv = self.neck_downsample_layers_1_conv(neck_bottom_up_layers_0_final_conv_activate)
        neck_downsample_layers_1_bn = self.neck_downsample_layers_1_bn(neck_downsample_layers_1_conv)
        neck_downsample_layers_1_conv = None
        neck_downsample_layers_1_activate = self.neck_downsample_layers_1_activate(neck_downsample_layers_1_bn)
        neck_downsample_layers_1_bn = None
        cat_11_f = self.float_functional_simple_18.cat([neck_downsample_layers_1_activate, neck_reduce_layers_2_activate], 1)
        neck_downsample_layers_1_activate = None
        neck_reduce_layers_2_activate = None
        neck_bottom_up_layers_1_short_conv_conv = self.neck_bottom_up_layers_1_short_conv_conv(cat_11_f)
        neck_bottom_up_layers_1_short_conv_bn = self.neck_bottom_up_layers_1_short_conv_bn(neck_bottom_up_layers_1_short_conv_conv)
        neck_bottom_up_layers_1_short_conv_conv = None
        neck_bottom_up_layers_1_short_conv_activate = self.neck_bottom_up_layers_1_short_conv_activate(neck_bottom_up_layers_1_short_conv_bn)
        neck_bottom_up_layers_1_short_conv_bn = None
        neck_bottom_up_layers_1_main_conv_conv = self.neck_bottom_up_layers_1_main_conv_conv(cat_11_f)
        cat_11_f = None
        neck_bottom_up_layers_1_main_conv_bn = self.neck_bottom_up_layers_1_main_conv_bn(neck_bottom_up_layers_1_main_conv_conv)
        neck_bottom_up_layers_1_main_conv_conv = None
        neck_bottom_up_layers_1_main_conv_activate = self.neck_bottom_up_layers_1_main_conv_activate(neck_bottom_up_layers_1_main_conv_bn)
        neck_bottom_up_layers_1_main_conv_bn = None
        neck_bottom_up_layers_1_blocks_0_conv1_conv = self.neck_bottom_up_layers_1_blocks_0_conv1_conv(neck_bottom_up_layers_1_main_conv_activate)
        neck_bottom_up_layers_1_main_conv_activate = None
        neck_bottom_up_layers_1_blocks_0_conv1_bn = self.neck_bottom_up_layers_1_blocks_0_conv1_bn(neck_bottom_up_layers_1_blocks_0_conv1_conv)
        neck_bottom_up_layers_1_blocks_0_conv1_conv = None
        neck_bottom_up_layers_1_blocks_0_conv1_activate = self.neck_bottom_up_layers_1_blocks_0_conv1_activate(neck_bottom_up_layers_1_blocks_0_conv1_bn)
        neck_bottom_up_layers_1_blocks_0_conv1_bn = None
        neck_bottom_up_layers_1_blocks_0_conv2_conv = self.neck_bottom_up_layers_1_blocks_0_conv2_conv(neck_bottom_up_layers_1_blocks_0_conv1_activate)
        neck_bottom_up_layers_1_blocks_0_conv1_activate = None
        neck_bottom_up_layers_1_blocks_0_conv2_bn = self.neck_bottom_up_layers_1_blocks_0_conv2_bn(neck_bottom_up_layers_1_blocks_0_conv2_conv)
        neck_bottom_up_layers_1_blocks_0_conv2_conv = None
        neck_bottom_up_layers_1_blocks_0_conv2_activate = self.neck_bottom_up_layers_1_blocks_0_conv2_activate(neck_bottom_up_layers_1_blocks_0_conv2_bn)
        neck_bottom_up_layers_1_blocks_0_conv2_bn = None
        cat_12_f = self.float_functional_simple_19.cat([neck_bottom_up_layers_1_blocks_0_conv2_activate, neck_bottom_up_layers_1_short_conv_activate], dim=1)
        neck_bottom_up_layers_1_blocks_0_conv2_activate = None
        neck_bottom_up_layers_1_short_conv_activate = None
        neck_bottom_up_layers_1_final_conv_conv = self.neck_bottom_up_layers_1_final_conv_conv(cat_12_f)
        cat_12_f = None
        neck_bottom_up_layers_1_final_conv_bn = self.neck_bottom_up_layers_1_final_conv_bn(neck_bottom_up_layers_1_final_conv_conv)
        neck_bottom_up_layers_1_final_conv_conv = None
        neck_bottom_up_layers_1_final_conv_activate = self.neck_bottom_up_layers_1_final_conv_activate(neck_bottom_up_layers_1_final_conv_bn)
        neck_bottom_up_layers_1_final_conv_bn = None
        neck_out_layers_0 = self.neck_out_layers_0(neck_top_down_layers_1_final_conv_activate)
        neck_top_down_layers_1_final_conv_activate = None
        neck_out_layers_1 = self.neck_out_layers_1(neck_bottom_up_layers_0_final_conv_activate)
        neck_bottom_up_layers_0_final_conv_activate = None
        neck_out_layers_2 = self.neck_out_layers_2(neck_bottom_up_layers_1_final_conv_activate)
        neck_bottom_up_layers_1_final_conv_activate = None
        bbox_head_head_module_convs_pred_0 = self.bbox_head_head_module_convs_pred_0(neck_out_layers_0)
        neck_out_layers_0 = None
        shape_0_f = bbox_head_head_module_convs_pred_0.shape
        view_0_f = bbox_head_head_module_convs_pred_0.view(shape_0_f[0], 3, 16, shape_0_f[2], shape_0_f[3])
        bbox_head_head_module_convs_pred_0 = None
        shape_0_f = None
        permute_0_f = view_0_f.permute(0, 1, 3, 4, 2)
        view_0_f = None
        contiguous_0_f = permute_0_f.contiguous()
        permute_0_f = None
        bbox_head_head_module_convs_pred_1 = self.bbox_head_head_module_convs_pred_1(neck_out_layers_1)
        neck_out_layers_1 = None
        shape_1_f = bbox_head_head_module_convs_pred_1.shape
        view_1_f = bbox_head_head_module_convs_pred_1.view(shape_1_f[0], 3, 16, shape_1_f[2], shape_1_f[3])
        bbox_head_head_module_convs_pred_1 = None
        shape_1_f = None
        permute_1_f = view_1_f.permute(0, 1, 3, 4, 2)
        view_1_f = None
        contiguous_1_f = permute_1_f.contiguous()
        permute_1_f = None
        bbox_head_head_module_convs_pred_2 = self.bbox_head_head_module_convs_pred_2(neck_out_layers_2)
        neck_out_layers_2 = None
        shape_2_f = bbox_head_head_module_convs_pred_2.shape
        view_2_f = bbox_head_head_module_convs_pred_2.view(shape_2_f[0], 3, 16, shape_2_f[2], shape_2_f[3])
        bbox_head_head_module_convs_pred_2 = None
        shape_2_f = None
        permute_2_f = view_2_f.permute(0, 1, 3, 4, 2)
        view_2_f = None
        contiguous_2_f = permute_2_f.contiguous()
        permute_2_f = None
        shape_3_f = contiguous_0_f.shape
        device_0_f = contiguous_0_f.device
        arange_0_f = torch.arange(shape_3_f[2], device=device_0_f)
        arange_1_f = torch.arange(shape_3_f[3], device=device_0_f)
        meshgrid_0_f = torch.meshgrid([arange_0_f, arange_1_f], indexing='ij')
        arange_0_f = None
        arange_1_f = None
        stack_0_f = torch.stack([meshgrid_0_f[1], meshgrid_0_f[0]], dim=2)
        meshgrid_0_f = None
        expand_0_f = stack_0_f.expand(1, 3, shape_3_f[2], shape_3_f[3], 2)
        stack_0_f = None
        float_0_f = expand_0_f.float()
        expand_0_f = None
        fake_quant_11 = self.fake_quant_11(float_0_f)
        float_0_f = None
        getitem_0_f = self.tensor_0[0]
        clone_0_f = getitem_0_f.clone()
        getitem_0_f = None
        view_3_f = clone_0_f.view([1, 3, 1, 1, 2])
        clone_0_f = None
        expand_1_f = view_3_f.expand([1, 3, shape_3_f[2], shape_3_f[3], 2])
        view_3_f = None
        float_1_f = expand_1_f.float()
        expand_1_f = None
        fake_quant_12 = self.fake_quant_12(float_1_f)
        float_1_f = None
        to_0_f = fake_quant_12.to(device_0_f)
        fake_quant_12 = None
        device_0_f = None
        rewritten_sigmoid_0 = self.rewritten_sigmoid_0(contiguous_0_f)
        contiguous_0_f = None
        getitem_1_f = rewritten_sigmoid_0[..., 0:2]
        mul_0_f = self.float_functional_simple_20.mul_scalar(getitem_1_f, 2)
        getitem_1_f = None
        sub_0_f = self.float_functional_simple_21.add_scalar(mul_0_f, -0.5)
        mul_0_f = None
        add_7_f = self.float_functional_simple_22.add(sub_0_f, fake_quant_11)
        sub_0_f = None
        fake_quant_11 = None
        device_1_f = rewritten_sigmoid_0.device
        as_tensor_0_f = torch.as_tensor(8, dtype=torch.float32, device=device_1_f)
        device_1_f = None
        fake_quant_4 = self.fake_quant_4(as_tensor_0_f)
        as_tensor_0_f = None
        fuse_expand_as_0_f = fake_quant_4.expand_as(add_7_f)
        fake_quant_4 = None
        mul_1_f = self.float_functional_simple_23.mul(add_7_f, fuse_expand_as_0_f)
        add_7_f = None
        fuse_expand_as_0_f = None
        getitem_2_f = rewritten_sigmoid_0[..., 2:4]
        mul_2_f = self.float_functional_simple_24.mul_scalar(getitem_2_f, 2)
        getitem_2_f = None
        fake_dequant_inner_0_0_0 = self.fake_dequant_inner_0_0_0(mul_2_f)
        mul_2_f = None
        pow_0_f = fake_dequant_inner_0_0_0.__pow__(2)
        fake_dequant_inner_0_0_0 = None
        fake_quant_inner_2_0_0 = self.fake_quant_inner_2_0_0(pow_0_f)
        pow_0_f = None
        mul_3_f = self.float_functional_simple_25.mul(fake_quant_inner_2_0_0, to_0_f)
        fake_quant_inner_2_0_0 = None
        to_0_f = None
        shape_4_f = rewritten_sigmoid_0.shape
        sub_1_f = shape_4_f[4].__sub__(4)
        shape_4_f = None
        split_0_f = torch.split(rewritten_sigmoid_0, [4, sub_1_f], dim=-1)
        rewritten_sigmoid_0 = None
        sub_1_f = None
        cat_13_f = self.float_functional_simple_26.cat([mul_1_f, mul_3_f, split_0_f[1]], -1)
        mul_1_f = None
        mul_3_f = None
        split_0_f = None
        view_4_f = cat_13_f.view(shape_3_f[0], -1, 16)
        cat_13_f = None
        shape_3_f = None
        shape_5_f = contiguous_1_f.shape
        device_2_f = contiguous_1_f.device
        arange_2_f = torch.arange(shape_5_f[2], device=device_2_f)
        arange_3_f = torch.arange(shape_5_f[3], device=device_2_f)
        meshgrid_1_f = torch.meshgrid([arange_2_f, arange_3_f], indexing='ij')
        arange_2_f = None
        arange_3_f = None
        stack_1_f = torch.stack([meshgrid_1_f[1], meshgrid_1_f[0]], dim=2)
        meshgrid_1_f = None
        expand_2_f = stack_1_f.expand(1, 3, shape_5_f[2], shape_5_f[3], 2)
        stack_1_f = None
        float_2_f = expand_2_f.float()
        expand_2_f = None
        fake_quant_13 = self.fake_quant_13(float_2_f)
        float_2_f = None
        getitem_3_f = self.tensor_0[1]
        clone_1_f = getitem_3_f.clone()
        getitem_3_f = None
        view_5_f = clone_1_f.view([1, 3, 1, 1, 2])
        clone_1_f = None
        expand_3_f = view_5_f.expand([1, 3, shape_5_f[2], shape_5_f[3], 2])
        view_5_f = None
        float_3_f = expand_3_f.float()
        expand_3_f = None
        fake_quant_14 = self.fake_quant_14(float_3_f)
        float_3_f = None
        to_1_f = fake_quant_14.to(device_2_f)
        fake_quant_14 = None
        device_2_f = None
        rewritten_sigmoid_1 = self.rewritten_sigmoid_1(contiguous_1_f)
        contiguous_1_f = None
        getitem_4_f = rewritten_sigmoid_1[..., 0:2]
        mul_4_f = self.float_functional_simple_27.mul_scalar(getitem_4_f, 2)
        getitem_4_f = None
        sub_2_f = self.float_functional_simple_28.add_scalar(mul_4_f, -0.5)
        mul_4_f = None
        add_8_f = self.float_functional_simple_29.add(sub_2_f, fake_quant_13)
        sub_2_f = None
        fake_quant_13 = None
        device_3_f = rewritten_sigmoid_1.device
        as_tensor_1_f = torch.as_tensor(16, dtype=torch.float32, device=device_3_f)
        device_3_f = None
        fake_quant_7 = self.fake_quant_7(as_tensor_1_f)
        as_tensor_1_f = None
        fuse_expand_as_1_f = fake_quant_7.expand_as(add_8_f)
        fake_quant_7 = None
        mul_5_f = self.float_functional_simple_30.mul(add_8_f, fuse_expand_as_1_f)
        add_8_f = None
        fuse_expand_as_1_f = None
        getitem_5_f = rewritten_sigmoid_1[..., 2:4]
        mul_6_f = self.float_functional_simple_31.mul_scalar(getitem_5_f, 2)
        getitem_5_f = None
        fake_dequant_inner_1_0_0 = self.fake_dequant_inner_1_0_0(mul_6_f)
        mul_6_f = None
        pow_1_f = fake_dequant_inner_1_0_0.__pow__(2)
        fake_dequant_inner_1_0_0 = None
        fake_quant_inner_1_0_0 = self.fake_quant_inner_1_0_0(pow_1_f)
        pow_1_f = None
        mul_7_f = self.float_functional_simple_32.mul(fake_quant_inner_1_0_0, to_1_f)
        fake_quant_inner_1_0_0 = None
        to_1_f = None
        shape_6_f = rewritten_sigmoid_1.shape
        sub_3_f = shape_6_f[4].__sub__(4)
        shape_6_f = None
        split_1_f = torch.split(rewritten_sigmoid_1, [4, sub_3_f], dim=-1)
        rewritten_sigmoid_1 = None
        sub_3_f = None
        cat_14_f = self.float_functional_simple_33.cat([mul_5_f, mul_7_f, split_1_f[1]], -1)
        mul_5_f = None
        mul_7_f = None
        split_1_f = None
        view_6_f = cat_14_f.view(shape_5_f[0], -1, 16)
        cat_14_f = None
        shape_5_f = None
        shape_7_f = contiguous_2_f.shape
        device_4_f = contiguous_2_f.device
        arange_4_f = torch.arange(shape_7_f[2], device=device_4_f)
        arange_5_f = torch.arange(shape_7_f[3], device=device_4_f)
        meshgrid_2_f = torch.meshgrid([arange_4_f, arange_5_f], indexing='ij')
        arange_4_f = None
        arange_5_f = None
        stack_2_f = torch.stack([meshgrid_2_f[1], meshgrid_2_f[0]], dim=2)
        meshgrid_2_f = None
        expand_4_f = stack_2_f.expand(1, 3, shape_7_f[2], shape_7_f[3], 2)
        stack_2_f = None
        float_4_f = expand_4_f.float()
        expand_4_f = None
        fake_quant_15 = self.fake_quant_15(float_4_f)
        float_4_f = None
        getitem_6_f = self.tensor_0[2]
        clone_2_f = getitem_6_f.clone()
        getitem_6_f = None
        view_7_f = clone_2_f.view([1, 3, 1, 1, 2])
        clone_2_f = None
        expand_5_f = view_7_f.expand([1, 3, shape_7_f[2], shape_7_f[3], 2])
        view_7_f = None
        float_5_f = expand_5_f.float()
        expand_5_f = None
        fake_quant_16 = self.fake_quant_16(float_5_f)
        float_5_f = None
        to_2_f = fake_quant_16.to(device_4_f)
        fake_quant_16 = None
        device_4_f = None
        rewritten_sigmoid_2 = self.rewritten_sigmoid_2(contiguous_2_f)
        contiguous_2_f = None
        getitem_7_f = rewritten_sigmoid_2[..., 0:2]
        mul_8_f = self.float_functional_simple_34.mul_scalar(getitem_7_f, 2)
        getitem_7_f = None
        sub_4_f = self.float_functional_simple_35.add_scalar(mul_8_f, -0.5)
        mul_8_f = None
        add_9_f = self.float_functional_simple_36.add(sub_4_f, fake_quant_15)
        sub_4_f = None
        fake_quant_15 = None
        device_5_f = rewritten_sigmoid_2.device
        as_tensor_2_f = torch.as_tensor(32, dtype=torch.float32, device=device_5_f)
        device_5_f = None
        fake_quant_10 = self.fake_quant_10(as_tensor_2_f)
        as_tensor_2_f = None
        fuse_expand_as_2_f = fake_quant_10.expand_as(add_9_f)
        fake_quant_10 = None
        mul_9_f = self.float_functional_simple_37.mul(add_9_f, fuse_expand_as_2_f)
        add_9_f = None
        fuse_expand_as_2_f = None
        getitem_8_f = rewritten_sigmoid_2[..., 2:4]
        mul_10_f = self.float_functional_simple_38.mul_scalar(getitem_8_f, 2)
        getitem_8_f = None
        fake_dequant_inner_2_0_0 = self.fake_dequant_inner_2_0_0(mul_10_f)
        mul_10_f = None
        pow_2_f = fake_dequant_inner_2_0_0.__pow__(2)
        fake_dequant_inner_2_0_0 = None
        fake_quant_inner_0_0_0 = self.fake_quant_inner_0_0_0(pow_2_f)
        pow_2_f = None
        mul_11_f = self.float_functional_simple_39.mul(fake_quant_inner_0_0_0, to_2_f)
        fake_quant_inner_0_0_0 = None
        to_2_f = None
        shape_8_f = rewritten_sigmoid_2.shape
        sub_5_f = shape_8_f[4].__sub__(4)
        shape_8_f = None
        split_2_f = torch.split(rewritten_sigmoid_2, [4, sub_5_f], dim=-1)
        rewritten_sigmoid_2 = None
        sub_5_f = None
        cat_15_f = self.float_functional_simple_40.cat([mul_9_f, mul_11_f, split_2_f[1]], -1)
        mul_9_f = None
        mul_11_f = None
        split_2_f = None
        view_8_f = cat_15_f.view(shape_7_f[0], -1, 16)
        cat_15_f = None
        shape_7_f = None
        cat_16_f = self.float_functional_simple_41.cat([view_4_f, view_6_f, view_8_f], 1)
        view_4_f = None
        view_6_f = None
        view_8_f = None
        fake_dequant_0 = self.fake_dequant_0(cat_16_f)
        cat_16_f = None
        return fake_dequant_0


if __name__ == "__main__":
    model = QYOLODetector()
    model.load_state_dict(torch.load('./work_dirs/yolov5tiny/yolodetector_q.pth'))

    model.eval()
    model.cpu()

    dummy_input_0 = torch.ones((1, 3, 192, 192), dtype=torch.float32)

    output = model(dummy_input_0)
    print(output)

