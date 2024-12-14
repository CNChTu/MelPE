import json
import pathlib

import torch
from einops import rearrange

from torchfcpe.f02midi.transpose import f02midi

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


def ensemble_f0(f0s, key_shift_list, tta_uv_penalty):
    """_summary_

    Args:
        f0s (torch.Tensor): (B, T, len(key_shift_list))
        key_shift_list (list): list of key shifts
        tta_uv_penalty (float,int): uv penalty

    Returns:
        f0: (B, T, 1)
    """
    device = f0s.device
    # convert f0 to note
    f0s = f0s / (
        torch.pow(
            2,
            torch.tensor(key_shift_list, device=device)
            .to(device)
            .unsqueeze(0)
            .unsqueeze(0)
            / 12,
        )
    )
    notes = torch.log2(f0s / 440) * 12 + 69
    notes[notes < 0] = 0

    # select best note
    # 使用动态规划选择最优的音高
    # 惩罚1：uv的惩罚固定为超参数uv_penalty ** 2，v转为uv时额外惩罚两次
    # 惩罚2：相邻帧音高的L2距离（uv和v互转的过程除外），距离小于0.5时忽略不计
    uv_penalty = tta_uv_penalty**2
    dp = torch.zeros_like(notes, device=device)
    # dp[b,t,c]表示，对于样本b，0到第t帧的所有选择中，选择第c个f0作为第t帧的结尾的最小惩罚
    backtrack = torch.zeros_like(notes, device=device).long()
    # backtrack[b,t,c]表示，对于样本b，0到第t帧的所有选择中，选择第c个f0作为第t帧的结尾时，t-1帧结尾的选择，值域为0到len(f0_list)-1
    # init
    dp[:, 0, :] = (notes[:, 0, :] <= 0) * uv_penalty
    # forward
    for t in range(1, notes.size(1)):
        penalty = torch.zeros(
            [notes.size(0), notes.size(2), notes.size(2)], device=device
        )
        # [b,c1,c2]表示第b个样本中，t-1帧选择c1，t帧选择c2的惩罚

        # t帧是uv的情况
        t_uv = notes[:, t, :] <= 0
        penalty += uv_penalty * t_uv.unsqueeze(1)

        # t帧是v的情况
        # t-1帧也是v的情况
        t1_uv = notes[:, t - 1, :] <= 0
        l2 = torch.pow(
            (notes[:, t - 1, :].unsqueeze(-1) - notes[:, t, :].unsqueeze(1))
            * (~t1_uv).unsqueeze(-1)
            * (~t_uv).unsqueeze(1),
            2,
        )
        l2 = l2 - 0.5
        l2 = l2 * (l2 > 0)
        penalty += l2

        # t-1帧是uv的情况，uv转v的惩罚
        penalty += t1_uv.unsqueeze(-1) * (~t_uv).unsqueeze(1) * uv_penalty * 2

        # 选择最小惩罚
        min_value, min_indices = torch.min(
            dp[:, t - 1, :].unsqueeze(-1) + penalty, dim=1
        )
        dp[:, t, :] = min_value
        backtrack[:, t, :] = min_indices

    # backtrack
    t = f0s.size(1) - 1
    f0_result = torch.zeros_like(f0s[:, :, 0], device=device)
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
        key_shifts: list = [0],
    ) -> torch.Tensor:
        """Infer
        Args:
            wav (torch.Tensor): Input wav, (B, n_sample, 1).
            sr (int, float): Input wav sample rate.
            decoder_mode (str): Decoder type. Default: "local_argmax", support "argmax" or "local_argmax".
            threshold (float): Threshold to mask. Default: 0.006.
            key_shifts (list): Key shifts. Default: [0].
        return: f0 (torch.Tensor): f0 Hz, shape (B, (n_sample//hop_size + 1), 1).
        """
        with torch.no_grad():
            wav = wav.to(self.tensor_device_marker.device)
            mels = torch.stack(
                [self.wav2mel(wav, sr, keyshift=keyshift) for keyshift in key_shifts],
                -1,
            )
            mels = rearrange(mels, "B T C K -> (B K) T C")
            f0s = self.model.infer(mels, decoder=decoder_mode, threshold=threshold)
            f0s = rearrange(f0s, "(B K) T 1 -> B T (K 1)", K=len(key_shifts))
        return f0s  # (B, T, len(key_shifts))

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
        return_uv: bool = False,
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
            return_uv (bool): Return unvoiced frames. Default: False.
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
            flag = 0
            if tta_use_origin_uv:
                if 0 not in tta_key_shifts:
                    flag = 1
                    tta_key_shifts.append(0)
            tta_key_shifts.sort(key=lambda x: (x if x >= 0 else -x / 2))
            f0s = self.__call__(wav, sr, decoder_mode, threshold, tta_key_shifts)
            f0 = ensemble_f0(
                f0s[:, :, flag:],
                tta_key_shifts[flag:],
                tta_uv_penalty,
            )
            if tta_use_origin_uv:
                f0_for_uv = f0s[:, :, [0]]
            else:
                f0_for_uv = f0
        else:
            f0 = self.__call__(wav, sr, decoder_mode, threshold)
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
            f0 = torch.where(f0 == 0, float("nan"), f0)
            f0 = torch.nn.functional.interpolate(
                f0.transpose(1, 2),
                size=int(output_interp_target_length),
                mode="linear",
            ).transpose(1, 2)
            f0 = torch.where(f0.isnan(), float(0.0), f0)
        # if return_uv is True, interp and return uv
        if return_uv:
            uv = torch.nn.functional.interpolate(
                uv.transpose(1, 2),
                size=int(output_interp_target_length),
                mode="nearest",
            ).transpose(1, 2)
            return f0, uv
        else:
            return f0

    def extact_midi(
        self,
        wav: torch.Tensor,
        sr: [int, float],
        output_path: str,
        decoder_mode: str = "local_argmax",
        threshold: float = 0.006,
        f0_min: float = None,
        f0_max: float = None,
        tempo: float = None,
    ):
        f0 = self.infer(
            wav,
            sr,
            decoder_mode,
            threshold,
            f0_min,
            f0_max,
        )
        f0 = f0.squeeze(-1).squeeze(0).cpu().numpy()
        wav = wav.squeeze(0).squeeze(-1).cpu().numpy()
        return f02midi(f0, tempo=tempo, output_path=output_path, sr=sr, y=wav)

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
