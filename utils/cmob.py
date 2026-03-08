import torch
import torch.nn.functional as F


class CMoB:
    def __init__(self, device):
        self.device = device

    def calculate_causal_gap(self, model, audio, visual, label, criterion):
        """
        计算音频和视频模态的因果效应差异 (Gap)。

        原理:
        TE_a (Total Effect of Audio) = Loss(Null_Audio, Visual) - Loss(Audio, Visual)
             注意：数值越大，说明去除 Audio 后 Loss 增加越多 -> Audio 越重要。
        TE_v (Total Effect of Visual) = Loss(Audio, Null_Visual) - Loss(Audio, Visual)

        Gap = Normalized(TE_a - TE_v)
        """
        was_training = model.training
        # 确保模型处于评估模式，避免 BatchNorm 在干预过程中更新统计量
        model.eval()

        with torch.no_grad():
            # 注意：这里需要适配你 main.py 中的输入处理 (unsqueeze)
            audio_in = audio.unsqueeze(1).float()
            visual_in = visual.float()

            # 1. 原始联合预测 (Joint Prediction)
            outputs = model(audio_in, visual_in)
            # 兼容返回 5 个值 (旧版) 或 7 个值 (新版含 projection heads)
            if len(outputs) == 5:
                _, _, _, _, out_joint = outputs
            else:
                _, _, _, _, out_joint, _, _ = outputs
            loss_joint = criterion(out_joint, label)

            # 2. 干预 Audio (Audio 置零) -> 计算 Visual 的单模态能力
            # 构造全 0 的 Audio 输入
            audio_null = torch.zeros_like(audio_in).to(self.device)
            outputs_null_a = model(audio_null, visual_in)
            if len(outputs_null_a) == 5:
                _, _, _, _, out_null_a = outputs_null_a
            else:
                _, _, _, _, out_null_a, _, _ = outputs_null_a
            loss_null_a = criterion(out_null_a, label)

            # 3. 干预 Visual (Visual 置零) -> 计算 Audio 的单模态能力
            # 构造全 0 的 Visual 输入
            visual_null = torch.zeros_like(visual_in).to(self.device)
            outputs_null_v = model(audio_in, visual_null)
            if len(outputs_null_v) == 5:
                _, _, _, _, out_null_v = outputs_null_v
            else:
                _, _, _, _, out_null_v, _, _ = outputs_null_v
            loss_null_v = criterion(out_null_v, label)

        if was_training:
            # 恢复训练模式（仅在进入函数前为训练态时恢复）
            model.train()

        # 4. 计算因果效应 (Total Effect)
        # 如果去掉了 Audio，Loss 变大了很多 (loss_null_a >> loss_joint)，说明 Audio 因果效应强
        te_audio = loss_null_a - loss_joint
        te_visual = loss_null_v - loss_joint

        # 5. 计算 Gap (Audio 效应 - Visual 效应)
        # 使用 tanh 进行归一化，使其范围在 (-1, 1) 之间，与你原代码中的 gap 逻辑保持量级一致
        raw_gap = te_audio - te_visual
        gap = torch.tanh(raw_gap)

        return gap, te_audio, te_visual
