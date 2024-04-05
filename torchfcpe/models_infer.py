import json
import pathlib

import torch

from .models import CFNaiveMelPE
from .tools import (
    DotDict,
    catch_none_args_must,
    catch_none_args_opti,
    get_config_json_in_same_path,
    get_device,
    spawn_wav2mel,
)
from .torch_interp import batch_interp_with_replacement_detach


def ensemble_f0(f0_list, key_shift_list, tta_uv_penalty):
    # convert f0 to note
    note_list = []
    for idx, (f0, key_shift) in enumerate(zip(f0_list, key_shift_list)):
        f0 = f0 / (2 ** (key_shift / 12))
        f0_list[idx] = f0
        f0 = f0 + (f0 == 0) * 1e-6
        note = torch.log2(f0 / 440) * 12 + 69
        note[note < 0] = 0
        note_list.append(note)

    # select best note
    # 使用动态规划选择最优的音高
    # 惩罚1：uv的惩罚固定为超参数uv_penalty ** 2，v转为uv时额外惩罚两次
    # 惩罚2：相邻帧音高的L2距离（uv和v互转的过程除外），距离小于0.5时忽略不计
    uv_penalty = tta_uv_penalty**2

    notes = torch.cat(note_list, dim=-1)  # (B, T, len(f0_list))
    dp = torch.zeros_like(notes)  # (B, T, len(f0_list))
    # dp[b,t,c]表示，对于样本b，0到第t帧的所有选择中，选择第c个f0作为第t帧的结尾的最小惩罚
    backtrack = torch.zeros_like(notes, dtype=torch.int64)  # (B, T, len(f0_list))
    # backtrack[b,t,c]表示，对于样本b，0到第t帧的所有选择中，选择第c个f0作为第t帧的结尾时，t-1帧结尾的选择，值域为0到len(f0_list)-1
    # init
    dp[:, 0, :] = (notes[:, 0, :] <= 0) * uv_penalty
    # forward
    for t in range(1, notes.size(1)):
        # 计算第t帧选择c的最小惩罚
        for c in range(notes.size(2)):
            # 情况1：当前帧选择c，是uv
            # 只需要处理uv的惩罚
            if notes[:, t, c] <= 0:
                min_value, min_indices = torch.min(
                    dp[:, t - 1, :] + 1 * uv_penalty,
                    dim=-1,
                )
                dp[:, t, c] = min_value
                backtrack[:, t, c] = min_indices
                continue

            # 情况2：当前帧选择c，是v
            # 首先判断上一个帧是不是v
            is_voiced = notes[:, t - 1, :] > 0
            # 是的话，计算L2距离，不是的话，距离为0
            l2_distance = (
                torch.pow((notes[:, t - 1, :] - notes[:, t, c]), 2) * is_voiced - 0.5
            )
            l2_distance *= l2_distance > 0
            min_value, min_indices = torch.min(
                dp[:, t - 1, :] + l2_distance + (~is_voiced) * uv_penalty * 2,
                dim=-1,
            )
            dp[:, t, c] = min_value
            backtrack[:, t, c] = min_indices
    # backtrack
    f0s = torch.cat(f0_list, dim=-1)  # (B, T, len(f0_list))
    t = f0s.size(1) - 1
    f0_result = torch.zeros_like(f0s[:, :, 0])
    min_indices = torch.argmin(dp[:, t, :], dim=-1)
    for i in range(0, t + 1):
        f0_result[:, t - i] = f0s[:, t - i, min_indices]
        min_indices = backtrack[:, t - i, min_indices]

    return f0_result.unsqueeze(-1)


class InferCFNaiveMelPE(torch.nn.Module):
    """Infer CFNaiveMelPE
    Args:
        args (DotDict): Config.
        state_dict (dict): Model state dict.
    """

    def __init__(self, args, state_dict):
        super().__init__()
        self.wav2mel = spawn_wav2mel(args, device="cpu")
        self.model = spawn_model(args)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.args_dict = dict(args)
        self.register_buffer(
            "tensor_device_marker", torch.tensor(1.0).float(), persistent=False
        )

    def forward(
        self,
        wav: torch.Tensor,
        sr: [int, float],
        decoder_mode: str = "local_argmax",
        threshold: float = 0.006,
        keyshift: [int, float] = 0,
    ) -> torch.Tensor:
        """Infer
        Args:
            wav (torch.Tensor): Input wav, (B, n_sample, 1).
            sr (int, float): Input wav sample rate.
            decoder_mode (str): Decoder type. Default: "local_argmax", support "argmax" or "local_argmax".
            threshold (float): Threshold to mask. Default: 0.006.
        return: f0 (torch.Tensor): f0 Hz, shape (B, (n_sample//hop_size + 1), 1).
        """
        with torch.no_grad():
            wav = wav.to(self.tensor_device_marker.device)
            mel = self.wav2mel(wav, sr, keyshift=keyshift)
            f0 = self.model.infer(mel, decoder=decoder_mode, threshold=threshold)
        return f0  # (B, T, 1)

    def infer(
        self,
        wav: torch.Tensor,
        sr: [int, float],
        decoder_mode: str = "local_argmax",
        threshold: float = 0.006,
        f0_min: float = None,
        f0_max: float = None,
        interp_uv: bool = False,
        output_interp_target_length: int = None,
        retur_uv: bool = False,
        test_time_augmentation: bool = False,
        tta_uv_penalty: float = 12.0,
        tta_key_shifts: list = [0, -12, 12],
        tta_use_origin_uv=False,
    ) -> torch.Tensor or (torch.Tensor, torch.Tensor):
        """Infer
        Args:
            wav (torch.Tensor): Input wav, (B, n_sample, 1).
            sr (int, float): Input wav sample rate.
            decoder_mode (str): Decoder type. Default: "local_argmax", support "argmax" or "local_argmax".
            threshold (float): Threshold to mask. Default: 0.006.
            f0_min (float): Minimum f0. Default: None. Use in post-processing.
            f0_max (float): Maximum f0. Default: None. Use in post-processing.
            interp_uv (bool): Interpolate unvoiced frames. Default: False.
            output_interp_target_length (int): Output interpolation target length. Default: None.
            retur_uv (bool): Return unvoiced frames. Default: False.
            test_time_augmentation (bool): Test time augmentation. If enabled, the output may be better but slower. Default: False.
            tta_uv_penalty (float): Test time augmentation unvoiced penalty. Default: 12.0.
            tta_key_shifts (list): Test time augmentation key shifts. Default: [0, -12, 12].
            tta_use_origin_uv (bool): Use origin uv. Default: False
        return: f0 (torch.Tensor): f0 Hz, shape (B, (n_sample//hop_size + 1) or output_interp_target_length, 1).
            if return_uv is True, return f0, uv. the shape of uv(torch.Tensor) is like f0.
        """
        # infer
        if test_time_augmentation:
            assert len(tta_key_shifts) > 0
            tta_key_shifts.sort(key=lambda x: (x if x >= 0 else -x / 2))
            f0_list = [
                self.__call__(wav, sr, decoder_mode, threshold, key_shift)
                for key_shift in tta_key_shifts
            ]
            f0 = ensemble_f0(
                f0_list,
                tta_key_shifts,
                tta_uv_penalty,
            )
        else:
            f0 = self.__call__(wav, sr, decoder_mode, threshold)
        if test_time_augmentation and tta_use_origin_uv:
            if tta_key_shifts[0] == 0:
                f0_for_uv = f0_list[0]
            else:
                f0_for_uv = self.__call__(wav, sr, decoder_mode, threshold, 0)
        else:
            f0_for_uv = f0
        if f0_min is None:
            f0_min = self.args_dict["model"]["f0_min"]
        uv = (f0_for_uv < f0_min).type(f0_for_uv.dtype)
        f0 = f0 * (1 - uv)
        # interp
        if interp_uv:
            f0 = batch_interp_with_replacement_detach(
                uv.squeeze(-1).bool(), f0.squeeze(-1)
            ).unsqueeze(-1)
        if f0_max is not None:
            f0[f0 > f0_max] = f0_max
        if output_interp_target_length is not None:
            f0 = torch.nn.functional.interpolate(
                f0.transpose(1, 2),
                size=int(output_interp_target_length),
                mode="nearest",
            ).transpose(1, 2)
        # if return_uv is True, interp and return uv
        if retur_uv:
            uv = torch.nn.functional.interpolate(
                uv.transpose(1, 2),
                size=int(output_interp_target_length),
                mode="nearest",
            ).transpose(1, 2)
            return f0, uv
        else:
            return f0

    def get_hop_size(self) -> int:
        """Get hop size"""
        return DotDict(self.args_dict).mel.hop_size

    def get_hop_size_ms(self) -> float:
        """Get hop size in ms"""
        return (
            DotDict(self.args_dict).mel.hop_size / DotDict(self.args_dict).mel.sr * 1000
        )

    def get_model_sr(self) -> int:
        """Get model sample rate"""
        return DotDict(self.args_dict).mel.sr

    def get_mel_config(self) -> dict:
        """Get mel config"""
        return dict(DotDict(self.args_dict).mel)

    def get_device(self) -> str:
        """Get device"""
        return self.tensor_device_marker.device

    def get_model_f0_range(self) -> dict:
        """Get model f0 range like {'f0_min': 32.70, 'f0_max': 1975.5}"""
        return {
            "f0_min": DotDict(self.args_dict).model.f0_min,
            "f0_max": DotDict(self.args_dict).model.f0_max,
        }


class InferCFNaiveMelPEONNX:
    """Infer CFNaiveMelPE ONNX
    Args:
        args (DotDict): Config.
        onnx_path (str): Path to onnx file.
        device (str): Device. must be not None.
    """

    def __init__(self, args, onnx_path, device):
        raise NotImplementedError


def spawn_bundled_infer_model(device: str = None) -> InferCFNaiveMelPE:
    """
    Spawn bundled infer model
    This model has been trained on our dataset and comes with the package.
    You can use it directly without anything else.
    Args:
        device (str): Device. Default: None.
    """
    file_path = pathlib.Path(__file__)
    model_path = file_path.parent / "assets" / "fcpe_c_v001.pt"
    model = spawn_infer_model_from_pt(str(model_path), device, bundled_model=True)
    return model


def spawn_infer_model_from_onnx(
    onnx_path: str, device: str = None
) -> InferCFNaiveMelPEONNX:
    """
    Spawn infer model from onnx file
    Args:
        onnx_path (str): Path to onnx file.
        device (str): Device. Default: None.
    """
    device = get_device(device, "torchfcpe.tools.spawn_infer_cf_naive_mel_pe_from_onnx")
    config_path = get_config_json_in_same_path(onnx_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
        args = DotDict(config_dict)
    if (args.is_onnx is None) or (args.is_onnx is False):
        raise ValueError(
            "  [ERROR] spawn_infer_model_from_onnx: this model is not onnx model."
        )

    if args.model.type == "CFNaiveMelPEONNX":
        infer_model = InferCFNaiveMelPEONNX(args, onnx_path, device)
    else:
        raise ValueError(
            f"  [ERROR] args.model.type is {args.model.type}, but only support CFNaiveMelPEONNX"
        )

    return infer_model


def spawn_infer_model_from_pt(
    pt_path: str, device: str = None, bundled_model: bool = False
) -> InferCFNaiveMelPE:
    """
    Spawn infer model from pt file
    Args:
        pt_path (str): Path to pt file.
        device (str): Device. Default: None.
        bundled_model (bool): Whether this model is bundled model, only used in spawn_bundled_infer_model.
    """
    device = get_device(device, "torchfcpe.tools.spawn_infer_cf_naive_mel_pe_from_pt")
    ckpt = torch.load(pt_path, map_location=torch.device(device))
    if bundled_model:
        ckpt["config_dict"]["model"]["conv_dropout"] = 0.0
        ckpt["config_dict"]["model"]["atten_dropout"] = 0.0
    args = DotDict(ckpt["config_dict"])
    if (args.is_onnx is not None) and (args.is_onnx is True):
        raise ValueError(
            "  [ERROR] spawn_infer_model_from_pt: this model is an onnx model."
        )

    if args.model.type == "CFNaiveMelPE":
        infer_model = InferCFNaiveMelPE(args, ckpt["model"])
        infer_model = infer_model.to(device)
        infer_model.eval()
    else:
        raise ValueError(
            f"  [ERROR] args.model.type is {args.model.type}, but only support CFNaiveMelPE"
        )

    return infer_model


def spawn_model(args: DotDict) -> CFNaiveMelPE:
    """Spawn conformer naive model"""
    if args.model.type == "CFNaiveMelPE":
        pe_model = CFNaiveMelPE(
            input_channels=catch_none_args_must(
                args.mel.num_mels,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.mel.num_mels is None",
            ),
            out_dims=catch_none_args_must(
                args.model.out_dims,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.out_dims is None",
            ),
            hidden_dims=catch_none_args_must(
                args.model.hidden_dims,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.hidden_dims is None",
            ),
            n_layers=catch_none_args_must(
                args.model.n_layers,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.n_layers is None",
            ),
            n_heads=catch_none_args_must(
                args.model.n_heads,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.n_heads is None",
            ),
            f0_max=catch_none_args_must(
                args.model.f0_max,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.f0_max is None",
            ),
            f0_min=catch_none_args_must(
                args.model.f0_min,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.f0_min is None",
            ),
            use_fa_norm=catch_none_args_must(
                args.model.use_fa_norm,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.use_fa_norm is None",
            ),
            conv_only=catch_none_args_opti(
                args.model.conv_only,
                default=False,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.conv_only is None",
            ),
            conv_dropout=catch_none_args_opti(
                args.model.conv_dropout,
                default=0.0,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.conv_dropout is None",
            ),
            atten_dropout=catch_none_args_opti(
                args.model.atten_dropout,
                default=0.0,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.atten_dropout is None",
            ),
            use_harmonic_emb=catch_none_args_opti(
                args.model.use_harmonic_emb,
                default=False,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.use_harmonic_emb is None",
            ),
        )
    else:
        raise ValueError(
            f"  [ERROR] args.model.type is {args.model.type}, but only support CFNaiveMelPE"
        )
    return pe_model


def bundled_infer_model_unit_test(wav_path):
    """Unit test for bundled infer model"""
    # wav_path is your wav file path
    try:
        import librosa
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "  [UNIT_TEST] torchfcpe.tools.spawn_infer_model_from_pt: matplotlib or librosa not found, skip test"
        )
        exit(1)

    infer_model = spawn_bundled_infer_model(device="cpu")
    wav, sr = librosa.load(wav_path, sr=16000)
    f0 = infer_model.infer(torch.tensor(wav).unsqueeze(0), sr, interp_uv=False)
    f0_interp = infer_model.infer(torch.tensor(wav).unsqueeze(0), sr, interp_uv=True)
    plt.plot(f0.squeeze(-1).squeeze(0).numpy(), color="r", linestyle="-")
    plt.plot(f0_interp.squeeze(-1).squeeze(0).numpy(), color="g", linestyle="-")
    # 添加图例
    plt.legend(["f0", "f0_interp"])
    plt.xlabel("frame")
    plt.ylabel("f0")
    plt.title("f0")
    plt.show()
