#!/usr/bin/env python3
"""
Export a Depth Anything 3 checkpoint to ONNX with fixed-size input using Loguru logging.
"""

from __future__ import annotations

import click
import os
from pathlib import Path

import torch
from torch import nn
import onnx
from loguru import logger

from depth_anything_3.api import DepthAnything3

PATCH_SIZE = 14


class EagleWrapper(nn.Module):
    """Simplified forward with intrinsics / extrinsics capability"""

    def __init__(self, model: DepthAnything3):
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor):
        """
        Single image --> B = 1, single view --> N = 1

        image: CHW (3xHxW) --> (1, 1, 3, H, W)
        extrinsics: (4, 4) --> (1, 1, 4, 4)
        intrinsics: (3, 3) --> (1, 1, 3, 3)
        """

        input_batch = batch.unsqueeze(1)
        input_extrinsics = extrinsics.unsqueeze(1)
        input_intrinsics = intrinsics.unsqueeze(1)

        output = self.model(
            input_batch,
            extrinsics=input_extrinsics,
            intrinsics=input_intrinsics,
            export_feat_layers=[],
            infer_gs=False,
        )
        return output["depth"]


def load_model(model_dir: Path, device: torch.device) -> DepthAnything3:
    api_model = DepthAnything3.from_pretrained(model_dir.as_posix())
    api_model = api_model.to(device)
    api_model.eval()
    return api_model


def _print_io_shapes(onnx_model) -> None:
    def _dims(tensor):
        dims = []
        for d in tensor.type.tensor_type.shape.dim:
            dims.append(d.dim_param if d.dim_param else d.dim_value)
        return dims

    for inp in onnx_model.graph.input:
        logger.info(f"ONNX Input {inp.name}: {_dims(inp)}")
    for out in onnx_model.graph.output:
        logger.info(f"ONNX Output {out.name}: {_dims(out)}")


def export_onnx(
    model_dir: str,
    onnx_path: Path,
    height: int,
    width: int,
    batch_size: int,
    opset: int,
    device: torch.device,
) -> None:
    if height % PATCH_SIZE != 0 or width % PATCH_SIZE != 0:
        raise ValueError(f"Height and width must be divisible by {PATCH_SIZE}.")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading checkpoint from {model_dir} to {device}...")
    api_model = load_model(Path(model_dir), device)
    logger.info(f"Model parameters: {sum(p.numel() for p in api_model.parameters())/1e6:.2f}M")

    wrapper = EagleWrapper(api_model).to(device)
    batch = torch.zeros(batch_size, 3, height, width, device=device, dtype=torch.float32)
    extrinsics = torch.zeros(batch_size, 4, 4, device=device, dtype=torch.float32)
    intrinsics = torch.zeros(batch_size, 3, 3, device=device, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (batch, extrinsics, intrinsics),
            onnx_path.as_posix(),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image", "extrinsics", "intrinsics"],
            output_names=["depth"],
            training=torch.onnx.TrainingMode.EVAL,
        )

    logger.success(f"ONNX model written to {onnx_path.resolve()}")

    # Validate the ONNX model
    logger.info("Validating ONNX model...")
    onnx_model = onnx.load(onnx_path.as_posix())
    onnx.checker.check_model(onnx_model)
    logger.success("ONNX model is valid!")
    _print_io_shapes(onnx_model)


@click.command()
@click.option(
    "--model-dir", type=str, required=True, help="Checkpoint directory or Hugging Face repo id."
)
@click.option(
    "--onnx-path", type=Path, default=None, help="Output ONNX path (defaults to <model>.onnx)."
)
@click.option("--height", type=int, required=True, help="Fixed input height (divisible by 14).")
@click.option("--width", type=int, required=True, help="Fixed input width (divisible by 14).")
@click.option("--batch-size", type=int, default=1, help="Batch size for dummy export input.")
@click.option("--opset", type=int, default=20, help="ONNX opset version.")
@click.option("--device", type=str, default="cuda", help="Device to export on (cpu or cuda).")
@click.option("--output-dir", type=Path, default=".", help="Directory to save ONNX output.")
def main(model_dir, onnx_path, height, width, batch_size, opset, device, output_dir):
    model_name = (
        Path(model_dir).name if Path(model_dir).exists() else model_dir.rstrip("/").split("/")[-1]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if onnx_path is None:
        onnx_path = output_dir / f"{model_name}.onnx"

    export_onnx(
        model_dir=model_dir,
        onnx_path=onnx_path,
        height=height,
        width=width,
        batch_size=batch_size,
        opset=opset,
        device=torch.device(device),
    )


if __name__ == "__main__":
    main()
