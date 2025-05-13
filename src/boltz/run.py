from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy

from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzWriter
from boltz.main import (
    BoltzDiffusionParams,
    check_inputs,
    download,
    process_inputs,
)
from boltz.model.model import Boltz1


def run_boltz(
    data: Union[str, Path],
    out_dir: Union[str, Path] = "./",
    cache: Union[str, Path] = "~/.boltz",
    checkpoint: Optional[Union[str, Path]] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    step_scale: float = 1.638,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    max_msa_seqs: int = 4096,
    max_unpaired_msa_seqs: Optional[int] = None,
    previous_msa_dir: Optional[Union[str, Path]] = None,
    return_results: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Run Boltz directly as a package function for protein structure prediction.

    Parameters
    ----------
    data : str or Path
        Path to input data file (.fasta, .yaml) or directory containing input files
    out_dir : str or Path
        Path where to save the predictions (default: current directory)
    cache : str or Path
        Directory where to download model and data (default: ~/.boltz)
    checkpoint : str or Path, optional
        Path to a checkpoint file, will use the default Boltz-1 model if None
    devices : int
        Number of devices to use for prediction (default: 1)
    accelerator : str
        Accelerator to use: "gpu", "cpu", or "tpu" (default: "gpu")
    recycling_steps : int
        Number of recycling steps for prediction (default: 3)
    sampling_steps : int
        Number of sampling steps for prediction (default: 200)
    diffusion_samples : int
        Number of diffusion samples to generate (default: 1)
    step_scale : float
        Step size for diffusion temperature control; lower value = higher diversity (default: 1.638)
    write_full_pae : bool
        Whether to dump predicted aligned error to npz file (default: False)
    write_full_pde : bool
        Whether to dump predicted distance error to npz file (default: False)
    output_format : str
        Output format for predictions: "pdb" or "mmcif" (default: "mmcif")
    num_workers : int
        Number of dataloader workers (default: 2)
    override : bool
        Whether to override existing predictions (default: False)
    seed : int, optional
        Random number generator seed; None means no seeding
    use_msa_server : bool
        Whether to use MMSeqs2 server for MSA generation (default: False)
    msa_server_url : str
        URL for MSA server (default: "https://api.colabfold.com")
    msa_pairing_strategy : str
        MSA pairing strategy: "greedy" or "complete" (default: "greedy")
    max_msa_seqs : int
        Maximum number of paired MSA sequences (default: 4096)
    max_unpaired_msa_seqs : int, optional
        Maximum number of unpaired MSA sequences (default: 2 * max_msa_seqs)
    previous_msa_dir : str or Path, optional
        Path to previously processed MSA data (useful for multiple seeds)
    return_results : bool
        Whether to return results dictionary rather than just saving files (default: False)

    Returns
    -------
    Optional[Dict[str, Any]]
        If return_results is True, returns a dictionary with prediction results
    """
    # Convert path inputs to Path objects
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    cache = Path(cache).expanduser()

    if checkpoint is not None:
        checkpoint = Path(checkpoint).expanduser()

    if previous_msa_dir is not None:
        previous_msa_dir = Path(previous_msa_dir).expanduser()
        use_msa_server = False

    # CPU warning
    if accelerator == "cpu":
        print("Running on CPU, this will be slow. Consider using a GPU.")

    # Set up PyTorch
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("medium")

    # Handle seeding
    if seed is not None:
        seed_everything(seed)

    # Create directories
    cache.mkdir(parents=True, exist_ok=True)

    # Set output directory
    results_dir = out_dir / f"boltz_results_{data.stem}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Download model and data
    download(cache)

    # Validate inputs
    input_files = check_inputs(data, results_dir, override)
    if not input_files:
        print("No predictions to run, exiting.")
        return None

    # Set up strategy based on number of devices
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        strategy = DDPStrategy()
        if len(input_files) < devices:
            raise ValueError("Number of requested devices is greater than number of predictions.")

    print(
        f"Running predictions for {len(input_files)} structure{'s' if len(input_files) > 1 else ''}"
    )

    # Set default for max_unpaired_msa_seqs if not specified
    if max_unpaired_msa_seqs is None:
        max_unpaired_msa_seqs = max_msa_seqs * 2

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    process_inputs(
        data=input_files,
        out_dir=results_dir,
        ccd_path=ccd_path,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        max_msa_seqs=max_msa_seqs,
        max_unpaired_msa_seqs=max_unpaired_msa_seqs,
        previous_msa_dir=previous_msa_dir,
    )

    # Load processed data
    processed_dir = results_dir / "processed"
    processed = {
        "manifest": Manifest.load(processed_dir / "manifest.json"),
        "targets_dir": processed_dir / "structures",
        "msa_dir": processed_dir / "msa",
    }

    # Create data module
    data_module = BoltzInferenceDataModule(
        manifest=processed["manifest"],
        target_dir=processed["targets_dir"],
        msa_dir=processed["msa_dir"],
        num_workers=num_workers,
    )

    # Load model
    if checkpoint is None:
        checkpoint = cache / "boltz1_conf.ckpt"

    # Set up prediction arguments for the model
    # - recycling_steps: number of times to feed outputs back as inputs for refinement
    # - sampling_steps: controls the granularity of the diffusion sampling process
    # - diffusion_samples: number of independent structures to generate for each target
    # - write flags: control which confidence metrics will be saved to output files
    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
    }

    # Configure the diffusion process parameters
    # - step_scale controls the "temperature" of sampling:
    #   - Higher values produce more conservative predictions
    #   - Lower values increase diversity but may reduce accuracy
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = step_scale

    # Load the pretrained model from checkpoint
    # - strict=True ensures all weights are correctly loaded
    # - predict_args configures the prediction behavior
    # - map_location="cpu" ensures initial loading happens on CPU before transferring to target device
    # - diffusion_process_args controls the noise schedule and sampling behavior
    # - ema=False disables use of Exponential Moving Average weights
    model_module = Boltz1.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
    )

    # Set model to evaluation mode
    # - Disables dropout, batch normalization updates, and other training-specific behaviors
    model_module.eval()

    # Create a writer to save prediction outputs
    # - Handles converting model outputs to PDB/mmCIF files and metadata
    # - Manages file organization and formats
    pred_writer = BoltzWriter(
        data_dir=processed["targets_dir"],
        output_dir=results_dir / "predictions",
        output_format=output_format,
    )

    # Set up PyTorch Lightning trainer
    # - Manages device placement, distributed execution, and prediction workflow
    # - strategy controls distributed training approach (DDP for multi-GPU)
    # - callbacks includes our writer to process outputs during prediction
    trainer = Trainer(
        default_root_dir=results_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
    )

    # Execute model prediction
    # - Runs inference on the processed data
    # - If return_results=True, collects predictions for return value
    # - Otherwise, relies on the writer callback to save outputs
    predictions = trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=return_results,
    )

    # Provide information about where results are stored
    print(f"Predictions completed. Results saved to: {results_dir}")

    if return_results:
        return {
            "predictions": predictions,
            "output_dir": results_dir,
            "manifest": processed["manifest"],
        }

    return None


def run_boltz_from_precomputed_latents(
    manifest_with_latents_path: Union[str, Path],
    out_dir: Union[str, Path] = "./",
    cache: Union[str, Path] = "~/.boltz",
    checkpoint: Optional[Union[str, Path]] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    step_scale: float = 1.638,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    seed: Optional[int] = None,
    return_results: bool = False,
    # MSA related args are not needed as latents are post-MSA/trunk
    msa_dropout: float = 0.0,  # Default to no dropout as it's for inference from latents
    msa_z_dropout: float = 0.0,
    pairformer_dropout: float = 0.0,
    mask_trunkz_confidence: bool = False,
    ablation_z: bool = False,
    ablation_s: bool = False,
    ablation_x: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Run Boltz structure prediction using precomputed latent tensors.

    This function assumes that a manifest file is provided where each record
    points to saved `s_trunk_latent.pt`, `z_trunk_latent.pt`,
    `s_inputs_latent.pt`, `rel_pos_encoding_latent.pt`, and `feats_batch.pt` files.
    The BoltzInferenceDataModule needs to be adapted to load these when a flag like
    `load_precomputed_latents=True` is passed.

    Parameters
    ----------
    manifest_with_latents_path : str or Path
        Path to a manifest JSON file. Each record in the manifest must contain
        paths to the precomputed latent tensors and the original features batch.
        (e.g., record.s_trunk_path, record.z_trunk_path, ..., record.feats_batch_path)
    out_dir : str or Path
        Path where to save the predictions (default: current directory)
    cache : str or Path
        Directory where to download model (default: ~/.boltz)
    checkpoint : str or Path, optional
        Path to a checkpoint file, will use the default Boltz-1 model if None
    devices : int
        Number of devices to use for prediction (default: 1)
    accelerator : str
        Accelerator to use: "gpu", "cpu", or "tpu" (default: "gpu")
    sampling_steps : int
        Number of sampling steps for prediction (default: 200)
    diffusion_samples : int
        Number of diffusion samples to generate (default: 1)
    step_scale : float
        Step size for diffusion temperature control (default: 1.638)
    write_full_pae : bool
        Whether to dump predicted aligned error to npz file (default: False)
    write_full_pde : bool
        Whether to dump predicted distance error to npz file (default: False)
    output_format : str
        Output format for predictions: "pdb" or "mmcif" (default: "mmcif")
    num_workers : int
        Number of dataloader workers (default: 2)
    seed : int, optional
        Random number generator seed; None means no seeding
    return_results : bool
        Whether to return results dictionary rather than just saving files (default: False)
    msa_dropout, msa_z_dropout, pairformer_dropout : float
        Dropout rates, typically 0.0 for inference from latents.
    mask_trunkz_confidence, ablation_z, ablation_s, ablation_x : bool
        Confidence model and ablation flags.

    Returns
    -------
    Optional[Dict[str, Any]]
        If return_results is True, returns a dictionary with prediction results
    """
    manifest_with_latents_path = Path(manifest_with_latents_path).expanduser()
    out_dir = Path(out_dir).expanduser()
    cache = Path(cache).expanduser()

    if checkpoint is not None:
        checkpoint = Path(checkpoint).expanduser()

    if accelerator == "cpu":
        print("Running on CPU, this will be slow. Consider using a GPU.")

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("medium")

    if seed is not None:
        seed_everything(seed)

    cache.mkdir(parents=True, exist_ok=True)

    # Output directory will be based on the manifest name or a generic name
    results_dir = out_dir / f"boltz_results_from_latents_{manifest_with_latents_path.stem}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Download model (weights only, CCD not strictly needed if feats_batch.pt contains all)
    # However, Boltz1 initialization might still expect some global const or ccd info indirectly.
    # For safety, ensure ccd.pkl is available if model loading has implicit dependencies.
    # download(cache) # CCD might not be needed if all info is in feats_batch.pt

    # Load the manifest that points to precomputed latents
    loaded_manifest = Manifest.load(manifest_with_latents_path)
    if not loaded_manifest.records:
        print("No records found in the provided manifest, exiting.")
        return None

    print(
        f"Running predictions for {len(loaded_manifest.records)} structure(s) from precomputed latents."
    )

    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        strategy = DDPStrategy()
        if len(loaded_manifest.records) < devices:
            raise ValueError(
                "Number of requested devices is greater than number of items in manifest."
            )

    # Create data module
    # IMPORTANT: BoltzInferenceDataModule needs to be modified to accept
    # 'load_precomputed_latents=True' and then load tensors from paths
    # specified in each manifest record. This change is not part of this response.
    data_module = BoltzInferenceDataModule(
        manifest=loaded_manifest,
        target_dir=None,  # Not used if loading from feats_batch.pt
        msa_dir=None,  # Not used if loading from feats_batch.pt
        num_workers=num_workers,
        load_precomputed_latents=True,  # This flag needs to be implemented in BoltzInferenceDataModule
    )

    if checkpoint is None:
        model_path = cache / "boltz1_conf.ckpt"
        if not model_path.exists():
            download(cache)  # Download if not present and checkpoint not specified
        checkpoint = model_path

    predict_args = {
        "recycling_steps": 0,  # Recycling is part of trunk, already done.
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
        "mask_trunkZ_confidence": mask_trunkz_confidence,
        "ablation_Z": ablation_z,
        "ablation_S": ablation_s,
        "ablation_X": ablation_x,
        "save_latents": False,  # Not saving latents when running from latents
    }

    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = step_scale

    model_module = Boltz1.load_from_checkpoint(
        checkpoint,
        strict=True,  # Be strict, as we are doing inference.
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        msa_dropout=msa_dropout,
        msa_z_dropout=msa_z_dropout,
        pairformer_dropout=pairformer_dropout,
        ema=False,  # Consider if EMA weights should be used
    )
    model_module.eval()

    # The data_dir for BoltzWriter might need adjustment if original target files are not available
    # or if IDs need to be sourced differently. For now, assume manifest IDs are sufficient.
    # If feats_batch.pt contains original target paths, writer could use them.
    # For simplicity, we assume the writer can work with the IDs from the manifest.
    pred_writer = BoltzWriter(
        data_dir=None,  # This might need to be a dummy path or handled by writer
        output_dir=results_dir / "predictions",
        output_format=output_format,
        # Potentially pass manifest to writer if it needs to map IDs to original file names
        # manifest=loaded_manifest
    )

    trainer = Trainer(
        default_root_dir=results_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
    )

    predictions = trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=return_results,
    )

    print(f"Predictions from latents completed. Results saved to: {results_dir}")

    if return_results:
        return {
            "predictions": predictions,
            "output_dir": results_dir,
            "manifest": loaded_manifest,
        }
    return None
