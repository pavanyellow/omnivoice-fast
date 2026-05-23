
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.model_executor.models.omnivoice.config import OmniVoiceConfig

logger = init_logger(__name__)


class HiggsAudioVQLayer(nn.Module):

    def __init__(self, codebook_size: int = 1024, codebook_dim: int = 64, hidden_size: int = 1024):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.project_out = nn.Linear(codebook_dim, hidden_size)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        quantized = self.codebook(indices)
        quantized = self.project_out(quantized)
        return quantized.permute(0, 2, 1)


class HiggsAudioRVQ(nn.Module):

    def __init__(
        self, num_quantizers: int = 8, codebook_size: int = 1024, codebook_dim: int = 64, hidden_size: int = 1024
    ):
        super().__init__()
        self.quantizers = nn.ModuleList(
            [HiggsAudioVQLayer(codebook_size, codebook_dim, hidden_size) for _ in range(num_quantizers)]
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(
            codes.shape[1],
            self.quantizers[0].project_out.out_features,
            codes.shape[2],
            device=codes.device,
            dtype=torch.float32,
        )
        for i, quantizer in enumerate(self.quantizers):
            result = result + quantizer.decode(codes[i])
        return result


class OmniVoiceDecoder(nn.Module):

    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.config = config
        self.sample_rate = config.sample_rate
        self._loaded = False

        self.quantizer = None
        self.fc2 = None
        self.acoustic_decoder = None

    @torch.inference_mode()
    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        if not self._loaded:
            raise RuntimeError("Decoder not loaded. Call load_weights() first.")

        device = audio_codes.device

        codes = audio_codes.transpose(0, 1).long()

        quantized = self.quantizer.decode(codes)

        quantized = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)

        audio = self.acoustic_decoder(quantized)

        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        return audio.to(device)

    def _adjust_output_padding(self, decoder: nn.Module):
        for module in decoder.modules():
            if isinstance(module, nn.ConvTranspose1d):
                stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                module.output_padding = (stride % 2,)

    def load_weights(self, model_dir: str, device: torch.device) -> None:
        from safetensors.torch import load_file
        from transformers import DacConfig, DacModel

        audio_tokenizer_path = os.path.join(model_dir, "audio_tokenizer")
        config_path = os.path.join(audio_tokenizer_path, "config.json")
        weights_path = os.path.join(audio_tokenizer_path, "model.safetensors")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Audio tokenizer weights not found at {weights_path}")

        with open(config_path) as f:
            tokenizer_config = json.load(f)

        state_dict = load_file(weights_path, device=str(device))

        codebook_dim = tokenizer_config.get("codebook_dim", 64)
        codebook_size = tokenizer_config.get("codebook_size", 1024)
        hidden_size = state_dict["quantizer.quantizers.0.project_out.weight"].shape[0]
        num_quantizers = sum(
            1 for k in state_dict if k.startswith("quantizer.quantizers.") and k.endswith(".codebook.embed")
        )

        self.quantizer = HiggsAudioRVQ(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            hidden_size=hidden_size,
        ).to(device)

        for i in range(num_quantizers):
            prefix = f"quantizer.quantizers.{i}"
            embed_key = f"{prefix}.codebook.embed"
            if embed_key in state_dict:
                self.quantizer.quantizers[i].codebook.weight.data.copy_(state_dict[embed_key])
            proj_out_w = f"{prefix}.project_out.weight"
            proj_out_b = f"{prefix}.project_out.bias"
            if proj_out_w in state_dict:
                self.quantizer.quantizers[i].project_out.weight.data.copy_(state_dict[proj_out_w])
            if proj_out_b in state_dict:
                self.quantizer.quantizers[i].project_out.bias.data.copy_(state_dict[proj_out_b])

        fc2_w = state_dict["fc2.weight"]
        fc2_b = state_dict["fc2.bias"]
        self.fc2 = nn.Linear(fc2_w.shape[1], fc2_w.shape[0]).to(device)
        self.fc2.weight.data.copy_(fc2_w)
        self.fc2.bias.data.copy_(fc2_b)

        dac_cfg = DacConfig(**tokenizer_config["acoustic_model_config"])
        dac_model = DacModel(dac_cfg)
        self.acoustic_decoder = dac_model.decoder.to(device)

        loaded = 0
        for name, param in self.acoustic_decoder.named_parameters():
            higgs_name = f"acoustic_decoder.{name}"
            if higgs_name in state_dict:
                param.data.copy_(state_dict[higgs_name])
                loaded += 1

        self._adjust_output_padding(self.acoustic_decoder)

        if hasattr(self.acoustic_decoder, "tanh"):
            self.acoustic_decoder.tanh = nn.Identity()

        self.acoustic_decoder.eval()
        self._loaded = True

        logger.info(
            "Loaded OmniVoice decoder: %d quantizers, fc2(%d→%d), acoustic decoder (%d weights)",
            num_quantizers,
            fc2_w.shape[1],
            fc2_w.shape[0],
            loaded,
        )
