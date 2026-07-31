"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: main_pipeline.py

Description:
   This is the main entry point for the OpenSAR pipeline. It allows executing
   training, evaluation, or inference for any of the supported models/use-cases:
   - VD (Vessel Detection)
   - RFI (Radio Frequency Interference) detection

   The script provides a command-line interface to select the model and mode,
   and executes the appropriate script from the pipeline subdirectories.
   The script also supports the execution of tools such as dataset generation, 
   label conversion, and geocoding.
   
History:
    - 2025-08-26:
        Initial creation by Anya Forestell.
    - 2025-08-29:
        Adjusted pipeline order.
    - 2026-03-26:
        Major refactor: Implemented functional entry point with CLI interface
        to execute training, evaluation, and inference scripts for all models.
    - 2026-04-22:
        Integrated label conversion and geocoding tools into the main pipeline with support for custom arguments.

Usage:
    python main_pipeline.py --model <model_name> --mode <mode> [additional args]
    
    Examples:
        python main_pipeline.py --model rfi_large --mode train
        python main_pipeline.py --model vd_large --mode inference

---------------------------------------------------------------------
Authors<br>
Name: Anya Forestell (AMFF)<br>
E-mail: amforestell@indracompany.com<br>
<br>
Name: Daniel Dutra <br>
E-mail: dadutra@indracompany.com <br>
Creation Date: 2025-08-26
---------------------------------------------------------------------
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import Optional, List, Tuple, Any
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from SARFI import cli
sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.geocoding.test_geocode import geocode_patch
from pipeline.main.utils import get_logger
from dataset_generation_scripts.l0_to_range_compressed.main import range_compression_demo
from dataset_generation_scripts.l0_to_range_compressed.l0_to_range_compressed import compress_split
from pipeline.RFI_usecase.inference import inference_only, get_run_config as rfi_get_run_config
from logging import Logger

# ═══════════════════════════════════════════════════════════════
# Script Paths Configuration
# ═══════════════════════════════════════════════════════════════

# Get the pipeline base directory (parent of 'main')
PIPELINE_BASE = Path(__file__).parent.parent

# Map model names to their script paths
SCRIPT_PATHS = {
    'rfi_large': {
        'train': PIPELINE_BASE / 'RFI_usecase' / 'train_rfi_large.py',
        'inference': 'rfi_inference_only',  # Handled via inference_only()
        'evaluate': PIPELINE_BASE / 'RFI_usecase' / 'inference.py',
    },
    'rfi_small': {
        'train': PIPELINE_BASE / 'RFI_usecase' / 'train_rfi_smallKD.py',
        'inference': 'rfi_inference_only',  # Handled via inference_only()
        'evaluate': PIPELINE_BASE / 'RFI_usecase' / 'inference.py',
    },
    'vd_large': {
        'train': PIPELINE_BASE / 'dvd_use_case' / 'large_model' / 'train.py',
        'inference': PIPELINE_BASE / 'dvd_use_case' / 'large_model' / 'scripts' / 'inference.py',
        'evaluate': None,  # Not implemented
    },  
    'vd_small': {
        'train': PIPELINE_BASE / 'dvd_use_case' / 'small_model' / 'train_KD.py',
        'inference': PIPELINE_BASE / 'dvd_use_case' / 'large_model' / 'scripts' / 'inference.py',
        'evaluate': None,  # Not implemented
    },      
}


# ═══════════════════════════════════════════════════════════════
# Pipeline Execution Functions
# ═══════════════════════════════════════════════════════════════

def validate_model_and_mode(model: str, mode: str) -> tuple[bool, Optional[str]]:
    """
    Validate that the specified model and mode combination is supported.
    
    Args:
        model: The model name ('rfi_large', 'rfi_small', 'vd_large', 'vd_small')
        mode: The execution mode ('train', 'inference', 'evaluate')
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if model not in SCRIPT_PATHS:
        available = ', '.join(SCRIPT_PATHS.keys())
        return False, f"Unknown model '{model}'. Available models: {available}"
    
    if mode not in SCRIPT_PATHS[model]:
        available_modes = ', '.join(SCRIPT_PATHS[model].keys())
        return False, f"Unknown mode '{mode}'. Available modes for {model}: {available_modes}"
    
    script_path = SCRIPT_PATHS[model][mode]
    if script_path is None:
        return False, f"Mode '{mode}' is not implemented for model '{model}'"
    
    # Special marker for RFI inference_only (handled in-process)
    if script_path == 'rfi_inference_only':
        return True, None

    if not script_path.exists():
        return False, f"Script not found: {script_path}"
    
    return True, None


def execute_script(script_path: Path, extra_args: Optional[List[str]] = None, logger:Logger=None) -> int:
    """
    Execute a Python script using subprocess.
    
    Args:
        script_path: Path to the Python script to execute
        extra_args: Additional command-line arguments to pass to the script
    
    Returns:
        int: Exit code of the executed script
    """
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    
    logger.info("=" * 70)
    logger.info(f"Executing: {' '.join(cmd)}")
    logger.info("=" * 70)
    
    try:
        # Execute and stream output in real-time
        result = subprocess.run(cmd, cwd=script_path.parent)
        return result.returncode
    except KeyboardInterrupt:
        logger.warning("\n\nExecution interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\nError executing script: {e}")
        return 1


def execute_pipeline(model: str, mode: str, extra_args: Optional[List[str]] = None, logger:Logger=None) -> int:
    """
    Main pipeline execution function.
    
    Args:
        model: The model to execute ('rfi_large', 'rfi_small', 'vd_large', 'vd_small')
        mode: The execution mode (train, inference, evaluate)
        extra_args: Additional arguments to pass to the underlying script
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    # Validate inputs
    is_valid, error_msg = validate_model_and_mode(model, mode)
    if not is_valid:
        logger.error(f"{error_msg}")
        return 1
    
    # Get the script path
    script_path = SCRIPT_PATHS[model][mode]

    # Handle RFI inference via inference_only (in-process)
    if script_path == 'rfi_inference_only':
        model_size = 'large' if model == 'rfi_large' else 'small'
        run_cfg = rfi_get_run_config(model_size)
        logger.info("\n" + "=" * 70)
        logger.info("OpenSAR Pipeline Execution (RFI inference_only)")
        logger.info("=" * 70)
        logger.info(f"Model:       {model}")
        logger.info(f"Mode:        {mode}")
        logger.info(f"Config:      {run_cfg['cfg_path']}")
        logger.info(f"Weights:     {run_cfg['weights']}")
        logger.info("=" * 70 + "\n")
        try:
            inference_only(
                model_size=model_size,
                cfg_path=run_cfg['cfg_path'],
                weights_path=run_cfg['weights'],
                test_dir=run_cfg['test_dir'],
                prediction_mask_dir=run_cfg['postprocessing_output_dir'].parent / f"prediction_masks_{model_size}",
                postprocessing_output_dir=run_cfg['postprocessing_output_dir'],
                stride_multiple=run_cfg['stride_multiple'],
                thr=run_cfg['thr'],
            )
            return 0
        except Exception as e:
            logger.error(f"RFI inference_only failed: {e}")
            return 1
    
    # For RFI evaluate, inject --model-size if not already provided
    if model in ('rfi_large', 'rfi_small') and mode == 'evaluate':
        model_size_arg = 'large' if model == 'rfi_large' else 'small'
        if extra_args is None:
            extra_args = []
        if '--model-size' not in extra_args:
            extra_args = ['--model-size', model_size_arg] + extra_args

    # Display execution information
    logger.info("\n" + "=" * 70)
    logger.info(f"OpenSAR Pipeline Execution")
    logger.info("=" * 70)
    logger.info(f"Model:       {model}")
    logger.info(f"Mode:        {mode}")
    logger.info(f"Script:      {script_path}")
    if extra_args:
        logger.info(f"Extra args:  {' '.join(extra_args)}")
    logger.info("=" * 70 + "\n")
    
    # Execute the script
    return execute_script(script_path, extra_args, logger=logger)

def execute_tool(tool: str, extra_args: Optional[List[str]] = None, logger:Logger=None) -> int:
    """
    Execute a specific tool (e.g., label conversion, geocoding).
    
    Args:
        tool: The tool to execute (e.g., 'label_conversion', 'geocoding')
        extra_args: Additional arguments to pass to the underlying script
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    mode = "best"
    max_matches = 10
    patch_filename = None 
    if extra_args and len(extra_args) > 0:
        for i in range(0, len(extra_args), 2):
            if extra_args[i] == "-f" and i + 1 < len(extra_args):
                mode = extra_args[i + 1]
            elif extra_args[i] == "-m" and i + 1 < len(extra_args):
                max_matches = int(extra_args[i + 1])
            elif extra_args[i] == "-p" and i + 1 < len(extra_args):
                patch_filename = extra_args[i + 1]
    if tool == "label_conversion":
        cli.sarfi_main_wrapper(mode=mode, max_matches=max_matches)
    elif tool == "geocoding":
        logger = get_logger()
        cfg = Path(__file__).parent.joinpath("configuration","geocoding_config.yaml")
        geocode_patch(logger=logger, cfg_fp=cfg, patch_filename=patch_filename)
    elif tool == "dataset_generation_dvd":
        dataset_script = PIPELINE_BASE.parent / "dataset_generation_scripts" / "DVD_dataset_generation" / "generate_dataset.py"
        if not dataset_script.exists():
            logger.error(f"Dataset generation script not found: {dataset_script}")
            return 1
        return execute_script(dataset_script, extra_args, logger=logger)
    elif tool == "dataset_generation_rfi":
        dataset_script = PIPELINE_BASE.parent / "dataset_generation_scripts" / "RFI_dataset_generation" / "generate_dataset.py"
        if not dataset_script.exists():
            logger.error(f"Dataset generation script not found: {dataset_script}")
            return 1
        return execute_script(dataset_script, extra_args, logger=logger)
    elif tool == "dataset_generation_flood":
        dataset_script = PIPELINE_BASE.parent / "dataset_generation_scripts" / "FD_dataset_generation" / "generate_dataset.py"
        if not dataset_script.exists():
            logger.error(f"Dataset generation script not found: {dataset_script}")
            return 1
        return execute_script(dataset_script, extra_args, logger=logger)
    elif tool == "range_compression":
        range_compression_demo()
        return 1
    return 0


def end2end_vd_large(extra_args: Optional[List[str]] = None, logger: Logger = None) -> Tuple[int, Any, float, float, int]:
    """
    End-to-end complexity profiling for the VD large (object_detection) model.

    Steps:
        1. Run compress_split (L0 to range-compressed) and measure time.
        2. Run inference.py and measure time, FLOPs, MACs.

    Args:
        extra_args: Additional arguments (unused currently).
        logger: Logger instance.

    Returns:
        Tuple[int, Any, float, float, int]: Exit code, profile stats, step 1 time, step 2 time, number of inference patches.
    """
    import torch

    obj_det_dir = PIPELINE_BASE / 'dvd_use_case' / 'large_model'
    inference_script = obj_det_dir / 'scripts' / 'inference.py'
    vd_config_path = obj_det_dir / 'config.yaml'
    
    sys.path.insert(0, str(obj_det_dir))
    from utilities.read_yaml import read_yaml

    config = read_yaml(vd_config_path)

    # Default output path for range-compressed data
    rescaled_output_path = Path(config["datapaths"]["rescaled_rc_patches_path_test"])

    if not rescaled_output_path.exists():
        rescaled_output_path.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Range compression ────────────────────────────
    logger.info("=" * 70)
    logger.info("Step 1/2: Range Compression")
    logger.info("=" * 70)

    t0 = time.perf_counter()
    try:
        compress_results = compress_split(
            split='test',
            dataset='vessel',
            rescaled_output_dir=rescaled_output_path,
            limit=1  # Limit to 1 patch for profiling purposes
        )
    except Exception as e:
        logger.error(f"Range Compression failed: {e}")
        return (1, None, 0.0, 0.0, 0)
    step1_time_s = time.perf_counter() - t0
    num_rc_patches = len(compress_results) if compress_results else 1

    logger.info(f"Wall time: {step1_time_s:.2f} s")
    logger.info(f"Patches processed: {num_rc_patches}")
    logger.info(f"Output dir: {rescaled_output_path}")

    # ─── Step 2: Inference + model profiling ──────────────────
    logger.info("=" * 70)
    logger.info("Step 2/2: Inference")
    logger.info("=" * 70)

    t0 = time.perf_counter()
    inference_exit_code = execute_script(inference_script, extra_args=None, logger=logger)
    step2_time_s = time.perf_counter() - t0

    # Count patches in slc_patches_dir from config
    num_inference_patches = 1
    profile_stats = None
    try:
        slc_patches_dir = Path(config.get('inference', {}).get('slc_patches_dir', ''))
        if slc_patches_dir.exists():
            patch_count = len(list(slc_patches_dir.iterdir()))
            if patch_count > 0:
                num_inference_patches = patch_count
    except Exception as e:
        logger.warning(f"Could not count inference patches: {e}")

    # Profile the YOLO model for FLOPs/MACs
    try:
        from scripts.profile_model import profile_model
        from ultralytics import YOLO

        training_config = config.get('training', {})
        model_file = training_config.get('model', 'yolo26n-pose.yaml')
        yolo_model = YOLO(model_file)

        device = 0 if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            yolo_model.model.to(torch.device(f'cuda:{device}'))

        profile_stats = profile_model(yolo_model, imgsz=training_config.get('imgsz', 512), device=device)
    except Exception as e:
        logger.warning(f"Model profiling failed: {e}")

    logger.info(f"Inference wall time: {step2_time_s:.2f} s")
    logger.info(f"Patches inferred: {num_inference_patches}")

    # ─── Summary ──────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("End-to-End Summary (vd_large)")
    logger.info("=" * 70)
    total_time = step1_time_s + step2_time_s
    logger.info(f"Step 1 (range compression) Total time (s):            {step1_time_s:.5f}")
    logger.info(f"Step 1 (range compression) Average time per patch (s): {step1_time_s / num_rc_patches:.5f}")
    logger.info(f"Step 2 (inference) Total time (s):                    {step2_time_s:.5f}")
    logger.info(f"Step 2 (inference) Average time per patch (s):        {step2_time_s / num_inference_patches:.5f}")
    logger.info(f"Total time taken (s):                                 {total_time:.5f}")
    logger.info(f"Average time per patch (s):                           {total_time / num_inference_patches:.5f}")
    if profile_stats:
        logger.info(f"Model GFLOPs:                                         {profile_stats.get('profile/GFLOPs', 'N/A')}")
        logger.info(f"Model GMACs:                                          {profile_stats.get('profile/GMACs', 'N/A')}")
        logger.info(f"Model Params (M):                                     {profile_stats.get('profile/params_M', 'N/A')}")
        logger.info(f"Avg Latency (ms):                                     {profile_stats.get('profile/latency_ms', 'N/A')}")
        logger.info(f"FPS:                                                  {profile_stats.get('profile/fps', 'N/A')}")
        logger.info(f"GPU Mem Peak (GB):                                    {profile_stats.get('profile/gpu_mem_peak_GB', 'N/A')}")
    logger.info("=" * 70 + "\n")

    return (0, profile_stats, step1_time_s, step2_time_s, num_inference_patches)

def end2end_vd_small(extra_args: Optional[List[str]] = None, logger: Logger = None) -> Tuple[int, Any, float, float, int]:
    """
    End-to-end complexity profiling for the VD small (object_detection) model.

    Steps:
        1. Run compress_split (L0 to range-compressed) and measure time.
        2. Run inference.py and measure time, FLOPs, MACs.

    Args:
        extra_args: Additional arguments (unused currently).
        logger: Logger instance.

    Returns:
        Tuple[int, Any, float, float, int]: Exit code, profile stats, step 1 time, step 2 time, number of inference patches.
    """
    import torch

    obj_det_dir = PIPELINE_BASE / 'dvd_use_case' / 'large_model'
    inference_script = obj_det_dir / 'scripts' / 'inference.py'
    vd_config_path = PIPELINE_BASE / 'dvd_use_case' / 'small_model' / 'config.yaml'
    
    sys.path.insert(0, str(obj_det_dir))
    from utilities.read_yaml import read_yaml

    config = read_yaml(vd_config_path)

    # Default output path for range-compressed data
    rescaled_output_path = Path(config["datapaths"]["rescaled_rc_patches_path_test"])

    if not rescaled_output_path.exists():
        rescaled_output_path.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Range compression ────────────────────────────
    logger.info("=" * 70)
    logger.info("Step 1/2: Range Compression")
    logger.info("=" * 70)

    t0 = time.perf_counter()
    try:
        compress_results = compress_split(
            split='test',
            dataset='vessel',
            rescaled_output_dir=rescaled_output_path,
            limit=1  # Limit to 1 patch for profiling purposes
        )
    except Exception as e:
        logger.error(f"Range Compression failed: {e}")
        return (1, None, 0.0, 0.0, 0)
    step1_time_s = time.perf_counter() - t0
    num_rc_patches = len(compress_results) if compress_results else 1

    logger.info(f"Wall time: {step1_time_s:.2f} s")
    logger.info(f"Patches processed: {num_rc_patches}")
    logger.info(f"Output dir: {rescaled_output_path}")

    # ─── Step 2: Inference + model profiling ──────────────────
    logger.info("=" * 70)
    logger.info("Step 2/2: Inference")
    logger.info("=" * 70)

    t0 = time.perf_counter()
    inference_exit_code = execute_script(inference_script, extra_args=None, logger=logger)
    step2_time_s = time.perf_counter() - t0

    # Count patches in slc_patches_dir from config
    num_inference_patches = 1
    profile_stats = None
    try:
        slc_patches_dir = Path(config.get('inference', {}).get('slc_patches_dir', ''))
        if slc_patches_dir.exists():
            patch_count = len(list(slc_patches_dir.iterdir()))
            if patch_count > 0:
                num_inference_patches = patch_count
    except Exception as e:
        logger.warning(f"Could not count inference patches: {e}")

    # Profile the YOLO model for FLOPs/MACs
    try:
        from scripts.profile_model import profile_model
        from ultralytics import YOLO

        training_config = config.get('training', {})
        model_file = training_config.get('model', 'yolo26n-pose.yaml')
        yolo_model = YOLO(model_file)

        device = 0 if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            yolo_model.model.to(torch.device(f'cuda:{device}'))

        profile_stats = profile_model(yolo_model, imgsz=training_config.get('imgsz', 512), device=device)
    except Exception as e:
        logger.warning(f"Model profiling failed: {e}")

    logger.info(f"Inference wall time: {step2_time_s:.2f} s")
    logger.info(f"Patches inferred: {num_inference_patches}")

    # ─── Summary ──────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("End-to-End Summary (vd_small)")
    logger.info("=" * 70)
    total_time = step1_time_s + step2_time_s
    logger.info(f"Step 1 (range compression) Total time (s):            {step1_time_s:.5f}")
    logger.info(f"Step 1 (range compression) Average time per patch (s): {step1_time_s / num_rc_patches:.5f}")
    logger.info(f"Step 2 (inference) Total time (s):                    {step2_time_s:.5f}")
    logger.info(f"Step 2 (inference) Average time per patch (s):        {step2_time_s / num_inference_patches:.5f}")
    logger.info(f"Total time taken (s):                                 {total_time:.5f}")
    logger.info(f"Average time per patch (s):                           {total_time / num_inference_patches:.5f}")
    if profile_stats:
        logger.info(f"Model GFLOPs:                                         {profile_stats.get('profile/GFLOPs', 'N/A')}")
        logger.info(f"Model GMACs:                                          {profile_stats.get('profile/GMACs', 'N/A')}")
        logger.info(f"Model Params (M):                                     {profile_stats.get('profile/params_M', 'N/A')}")
        logger.info(f"Avg Latency (ms):                                     {profile_stats.get('profile/latency_ms', 'N/A')}")
        logger.info(f"FPS:                                                  {profile_stats.get('profile/fps', 'N/A')}")
        logger.info(f"GPU Mem Peak (GB):                                    {profile_stats.get('profile/gpu_mem_peak_GB', 'N/A')}")
    logger.info("=" * 70 + "\n")

    return (0, profile_stats, step1_time_s, step2_time_s, num_inference_patches)


def end2end_rfi_large(
    extra_args: Optional[List[str]] = None,
    logger: Logger = None,
) -> Tuple[int, Any, float, float, int]:
    """
    End-to-end complexity profiling for the RFI large model.

    Steps:
        1. Run compress_split (L0 to range-compressed) and measure time.
        2. Run inference_only (generate masks + optional post-processing) and measure time.
        3. Profile the UNet model for FLOPs/MACs using profile_model.

    Args:
        extra_args: Additional arguments (unused currently).
        logger: Logger instance.

    Returns:
        Tuple[int, Any, float, float, int]: Exit code (0 for success, non-zero for failure), profile stats, step 1 time, step 2 time, number of inference patches.
    """
    import torch
    import yaml

    cfg_fp = Path(__file__).parent.joinpath("configuration", "end2end_config.yaml")
    with open(cfg_fp, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg_path  = Path(cfg["rfi_large"]["cfg_path"])
    weights   = Path(cfg["rfi_large"]["weights_path"])
    test_dir  = Path(cfg["rfi_large"]["test_path"])
    out_dir   = Path(cfg["rfi_large"]["out_path"])
    prediction_mask_dir = Path(cfg["rfi_large"]["prediction_mask_dir"])
    postprocessing_output_dir = Path(cfg["rfi_large"]["postprocessing_output_dir"])
    thr = float(cfg["rfi_large"]["thr"])
    stride_multiple = int(cfg["rfi_large"]["stride_multiple"])
    input_shape = tuple(cfg["rfi_large"]["input_shape"])

    # Load RFI model config to get post-processing settings
    from pipeline.RFI_usecase.utils.rc_dataloader import load_yaml_config
    model_cfg = load_yaml_config(str(cfg_path))
    postproc_cfg = model_cfg.get("postprocessing", {})
    
    run_postprocessing = postproc_cfg.get("enabled", True)
    save_images = postproc_cfg.get("save_images", True)
    filter_level = postproc_cfg.get("filter_level", "region")

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    rescaled_output_path = Path(cfg["rfi_large"]["rescaled_output_path"])

    if not rescaled_output_path.exists():
        rescaled_output_path.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Range compression ────────────────────────────
    logger.info("=" * 70)
    logger.info("Step 1/3: Range Compression")
    logger.info("=" * 70)

    t0 = time.perf_counter()
    try:
        compress_results = compress_split(
            split='test',
            dataset='rfi',
            rescaled_output_dir=rescaled_output_path,
            limit=1
        )
    except Exception as e:
        logger.error(f"Range Compression failed: {e}")
        return (1, None, 0.0, 0.0, 0)

    step1_time_s = time.perf_counter() - t0
    num_rc_patches = len(compress_results) if compress_results else 1

    logger.info(f"Wall time: {step1_time_s:.2f} s")
    logger.info(f"Patches processed: {num_rc_patches}")
    logger.info(f"Output dir: {rescaled_output_path}")

    # ─── Step 2: Inference (inference_only) ───────────────────
    logger.info("=" * 70)
    logger.info("Step 2/3: Inference (inference_only)")
    logger.info("=" * 70)

    t0 = time.perf_counter()
    try:
        inference_only(
            model_size="large",
            cfg_path=cfg_path,
            weights_path=weights,
            test_dir=test_dir,
            prediction_mask_dir=prediction_mask_dir,
            postprocessing_output_dir=postprocessing_output_dir,
            stride_multiple=stride_multiple,
            thr=thr,
            run_postprocessing=run_postprocessing,
            filter_level=filter_level,
            save_results=save_images,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return (1, None, step1_time_s, 0.0, 0)
    step2_time_s = time.perf_counter() - t0

    # Count output PNG patches
    num_inference_patches = 1
    try:
        if prediction_mask_dir.exists():
            patch_count = len(list(Path(prediction_mask_dir).glob("*.png")))
            if patch_count > 0:
                num_inference_patches = patch_count
    except Exception as e:
        logger.warning(f"Could not count inference patches: {e}")

    logger.info(f"Inference wall time: {step2_time_s:.2f} s")
    logger.info(f"Patches inferred: {num_inference_patches}")

    # ─── Step 3: Model profiling ──────────────────────────────
    logger.info("=" * 70)
    logger.info("Step 3/3: Model Profiling")
    logger.info("=" * 70)

    profile_stats = None
    try:
        from pipeline.RFI_usecase.inference import _load_model as rfi_load_model
        from pipeline.RFI_usecase.utils.rc_dataloader import load_yaml_config
        obj_det_scripts = PIPELINE_BASE / 'dvd_use_case' / 'large_model'
        sys.path.insert(0, str(obj_det_scripts))
        from scripts.profile_model import HardwareSampler, ProfileStats, pretty_print_profile
        from thop import profile as thop_profile

        model_cfg_data = load_yaml_config(str(cfg_path))
        model_cfg = model_cfg_data.get("model", {})
        in_channels = input_shape[0]
        imgsz = input_shape[1]

        device_idx = 0 if torch.cuda.is_available() else 'cpu'
        device_obj = torch.device(f'cuda:{device_idx}') if torch.cuda.is_available() else torch.device('cpu')

        rfi_model = rfi_load_model(
            model_size="large",
            model_cfg=model_cfg,
            weights=weights,
            device=device_obj,
        )
        dummy = torch.zeros(1, in_channels, imgsz, imgsz).to(device_obj)

        macs, params = thop_profile(rfi_model, inputs=(dummy,), verbose=False)

        for _ in range(50):
            rfi_model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device_idx)

        sampler = HardwareSampler(device_idx=device_idx, interval=0.05)
        sampler.start()

        runs = 200
        t0 = time.perf_counter()
        for _ in range(runs):
            rfi_model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / runs * 1000

        sampler.stop()

        if torch.cuda.is_available():
            gpu_mem_alloc_gb = torch.cuda.memory_allocated(device_idx) / 1e9
            gpu_mem_peak_gb  = torch.cuda.max_memory_allocated(device_idx) / 1e9
        else:
            gpu_mem_alloc_gb = 0.0
            gpu_mem_peak_gb  = 0.0

        s = sampler
        profile_stats = ProfileStats({
            "profile/GMACs":            macs / 1e9,
            "profile/GFLOPs":           2 * macs / 1e9,
            "profile/params_M":         params / 1e6,
            "profile/latency_ms":       latency_ms,
            "profile/fps":              1000 / latency_ms,
            "profile/gpu_mem_alloc_GB": gpu_mem_alloc_gb,
            "profile/gpu_mem_peak_GB":  gpu_mem_peak_gb,
            "profile/gpu_util_avg_%":   s._safe(s.gpu_util_samples,  lambda x: sum(x) / len(x)),
            "profile/gpu_util_peak_%":  s._safe(s.gpu_util_samples,  max),
            "profile/gpu_power_avg_W":  s._safe(s.gpu_power_samples, lambda x: sum(x) / len(x)),
            "profile/gpu_power_peak_W": s._safe(s.gpu_power_samples, max),
            "profile/cpu_util_avg_%":   s._safe(s.cpu_util_samples,  lambda x: sum(x) / len(x)),
            "profile/cpu_util_peak_%":  s._safe(s.cpu_util_samples,  max),
            "profile/cpu_power_avg_W":  s._safe(s.cpu_power_samples, lambda x: sum(x) / len(x)),
            "profile/cpu_power_peak_W": s._safe(s.cpu_power_samples, max),
        })
    except Exception as e:
        logger.warning(f"Model profiling failed: {e}")

    # ─── Summary ──────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("End-to-End Summary (rfi_large)")
    logger.info("=" * 70)
    total_time = step1_time_s + step2_time_s
    logger.info(f"Step 1 (range compression) Total time (s):            {step1_time_s:.5f}")
    logger.info(f"Step 1 (range compression) Average time per patch (s): {step1_time_s / num_rc_patches:.5f}")
    logger.info(f"Step 2 (inference) Total time (s):                    {step2_time_s:.5f}")
    logger.info(f"Step 2 (inference) Average time per patch (s):        {step2_time_s / num_inference_patches:.5f}")
    logger.info(f"Total time taken (s):                                 {total_time:.5f}")
    logger.info(f"Average time per patch (s):                           {total_time / num_inference_patches:.5f}")
    if profile_stats:
        logger.info(f"Model GFLOPs:                                         {profile_stats.get('profile/GFLOPs', 'N/A')}")
        logger.info(f"Model GMACs:                                          {profile_stats.get('profile/GMACs', 'N/A')}")
        logger.info(f"Model Params (M):                                     {profile_stats.get('profile/params_M', 'N/A')}")
        logger.info(f"Avg Latency (ms):                                     {profile_stats.get('profile/latency_ms', 'N/A')}")
        logger.info(f"FPS:                                                  {profile_stats.get('profile/fps', 'N/A')}")
        logger.info(f"GPU Mem Peak (GB):                                    {profile_stats.get('profile/gpu_mem_peak_GB', 'N/A')}")
    logger.info("=" * 70 + "\n")

    return (0, profile_stats, step1_time_s, step2_time_s, num_inference_patches)


def end2end_rfi_small(
    extra_args: Optional[List[str]] = None,
    logger: Logger = None,
) -> Tuple[int, Any, float, float, int]:
    """
    End-to-end complexity profiling for the RFI small model.

    Steps:
        1. Run compress_split (L0 to range-compressed) and measure time.
        2. Run inference_only (generate masks + optional post-processing) and measure time.
        3. Profile the UNetSmall model for FLOPs/MACs using profile_model.

    Args:
        extra_args: Additional arguments (unused currently).
        logger: Logger instance.

    Returns:
        Tuple[int, Any, float, float, int]: Exit code, profile stats, step 1 time, step 2 time, number of inference patches.
    """
    import torch
    import yaml

    cfg_fp = Path(__file__).parent.joinpath("configuration", "end2end_config.yaml")
    with open(cfg_fp, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg_path = Path(cfg["rfi_small"]["cfg_path"])
    weights  = Path(cfg["rfi_small"]["weights_path"])
    test_dir = Path(cfg["rfi_small"]["test_path"])
    out_dir  = Path(cfg["rfi_small"]["out_path"])
    prediction_mask_dir = Path(cfg["rfi_small"]["prediction_mask_dir"])
    postprocessing_output_dir = Path(cfg["rfi_small"]["postprocessing_output_dir"])
    thr = float(cfg["rfi_small"]["thr"])
    stride_multiple = int(cfg["rfi_small"]["stride_multiple"])
    input_shape = tuple(cfg["rfi_small"]["input_shape"])

    # Load RFI model config to get post-processing settings
    from pipeline.RFI_usecase.utils.rc_dataloader import load_yaml_config
    model_cfg = load_yaml_config(str(cfg_path))
    postproc_cfg = model_cfg.get("postprocessing", {})
    
    run_postprocessing = postproc_cfg.get("enabled", True)
    save_images = postproc_cfg.get("save_images", True)
    filter_level = postproc_cfg.get("filter_level", "patch")

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    rescaled_output_path = Path(cfg["rfi_small"]["rescaled_output_path"])

    if not rescaled_output_path.exists():
        rescaled_output_path.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Range compression ────────────────────────────
    logger.info("=" * 70)
    logger.info("Step 1/3: Range Compression")
    logger.info("=" * 70)

    t0 = time.perf_counter()
    try:
        compress_results = compress_split(
            split='test',
            dataset='rfi',
            rescaled_output_dir=rescaled_output_path,
            limit=1
        )
    except Exception as e:
        logger.error(f"Range Compression failed: {e}")
        return (1, None, 0.0, 0.0, 0)
    step1_time_s = time.perf_counter() - t0
    num_rc_patches = len(compress_results) if compress_results else 1

    logger.info(f"Wall time: {step1_time_s:.2f} s")
    logger.info(f"Patches processed: {num_rc_patches}")
    logger.info(f"Output dir: {rescaled_output_path}")

    # ─── Step 2: Inference (inference_only) ───────────────────
    logger.info("=" * 70)
    logger.info("Step 2/3: Inference (inference_only)")
    logger.info("=" * 70)

    t0 = time.perf_counter()
    try:
        inference_only(
            model_size="small",
            cfg_path=cfg_path,
            weights_path=weights,
            test_dir=test_dir,
            prediction_mask_dir=prediction_mask_dir,
            postprocessing_output_dir=postprocessing_output_dir,
            stride_multiple=stride_multiple,
            thr=thr,
            run_postprocessing=run_postprocessing,
            filter_level=filter_level,
            save_results=save_images,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return (1, None, step1_time_s, 0.0, 0)
    step2_time_s = time.perf_counter() - t0

    num_inference_patches = 1
    try:
        if prediction_mask_dir.exists():
            patch_count = len(list(Path(prediction_mask_dir).glob("*.png")))
            if patch_count > 0:
                num_inference_patches = patch_count
    except Exception as e:
        logger.warning(f"Could not count inference patches: {e}")

    logger.info(f"Inference wall time: {step2_time_s:.2f} s")
    logger.info(f"Patches inferred: {num_inference_patches}")

    # ─── Step 3: Model profiling ──────────────────────────────
    logger.info("=" * 70)
    logger.info("Step 3/3: Model Profiling")
    logger.info("=" * 70)

    profile_stats = None
    try:
        from pipeline.RFI_usecase.inference import _load_model as rfi_load_model
        from pipeline.RFI_usecase.utils.rc_dataloader import load_yaml_config
        obj_det_scripts = PIPELINE_BASE / 'dvd_use_case' / 'large_model'
        sys.path.insert(0, str(obj_det_scripts))
        from scripts.profile_model import HardwareSampler, ProfileStats, pretty_print_profile
        from thop import profile as thop_profile

        model_cfg_data = load_yaml_config(str(cfg_path))
        model_cfg = model_cfg_data.get("model", {})
        in_channels = input_shape[0]
        imgsz = input_shape[1]

        device_idx = 0 if torch.cuda.is_available() else 'cpu'
        device_obj = torch.device(f'cuda:{device_idx}') if torch.cuda.is_available() else torch.device('cpu')

        rfi_model = rfi_load_model(
            model_size="small",
            model_cfg=model_cfg,
            weights=weights,
            device=device_obj,
        )
        dummy = torch.zeros(1, in_channels, imgsz, imgsz).to(device_obj)

        macs, params = thop_profile(rfi_model, inputs=(dummy,), verbose=False)

        for _ in range(50):
            rfi_model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device_idx)

        sampler = HardwareSampler(device_idx=device_idx, interval=0.05)
        sampler.start()

        runs = 200
        t0 = time.perf_counter()
        for _ in range(runs):
            rfi_model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / runs * 1000

        sampler.stop()

        if torch.cuda.is_available():
            gpu_mem_alloc_gb = torch.cuda.memory_allocated(device_idx) / 1e9
            gpu_mem_peak_gb  = torch.cuda.max_memory_allocated(device_idx) / 1e9
        else:
            gpu_mem_alloc_gb = 0.0
            gpu_mem_peak_gb  = 0.0

        s = sampler
        profile_stats = ProfileStats({
            "profile/GMACs":            macs / 1e9,
            "profile/GFLOPs":           2 * macs / 1e9,
            "profile/params_M":         params / 1e6,
            "profile/latency_ms":       latency_ms,
            "profile/fps":              1000 / latency_ms,
            "profile/gpu_mem_alloc_GB": gpu_mem_alloc_gb,
            "profile/gpu_mem_peak_GB":  gpu_mem_peak_gb,
            "profile/gpu_util_avg_%":   s._safe(s.gpu_util_samples,  lambda x: sum(x) / len(x)),
            "profile/gpu_util_peak_%":  s._safe(s.gpu_util_samples,  max),
            "profile/gpu_power_avg_W":  s._safe(s.gpu_power_samples, lambda x: sum(x) / len(x)),
            "profile/gpu_power_peak_W": s._safe(s.gpu_power_samples, max),
            "profile/cpu_util_avg_%":   s._safe(s.cpu_util_samples,  lambda x: sum(x) / len(x)),
            "profile/cpu_util_peak_%":  s._safe(s.cpu_util_samples,  max),
            "profile/cpu_power_avg_W":  s._safe(s.cpu_power_samples, lambda x: sum(x) / len(x)),
            "profile/cpu_power_peak_W": s._safe(s.cpu_power_samples, max),
        })
    except Exception as e:
        logger.warning(f"Model profiling failed: {e}")

    # ─── Summary ──────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("End-to-End Summary (rfi_small)")
    logger.info("=" * 70)
    total_time = step1_time_s + step2_time_s
    logger.info(f"Step 1 (range compression) Total time (s):            {step1_time_s:.5f}")
    logger.info(f"Step 1 (range compression) Average time per patch (s): {step1_time_s / num_rc_patches:.5f}")
    logger.info(f"Step 2 (inference) Total time (s):                    {step2_time_s:.5f}")
    logger.info(f"Step 2 (inference) Average time per patch (s):        {step2_time_s / num_inference_patches:.5f}")
    logger.info(f"Total time taken (s):                                 {total_time:.5f}")
    logger.info(f"Average time per patch (s):                           {total_time / num_inference_patches:.5f}")
    if profile_stats:
        logger.info(f"Model GFLOPs:                                         {profile_stats.get('profile/GFLOPs', 'N/A')}")
        logger.info(f"Model GMACs:                                          {profile_stats.get('profile/GMACs', 'N/A')}")
        logger.info(f"Model Params (M):                                     {profile_stats.get('profile/params_M', 'N/A')}")
        logger.info(f"Avg Latency (ms):                                     {profile_stats.get('profile/latency_ms', 'N/A')}")
        logger.info(f"FPS:                                                  {profile_stats.get('profile/fps', 'N/A')}")
        logger.info(f"GPU Mem Peak (GB):                                    {profile_stats.get('profile/gpu_mem_peak_GB', 'N/A')}")
    logger.info("=" * 70 + "\n")

    return (0, profile_stats, step1_time_s, step2_time_s, num_inference_patches)


def list_available_options(logger:Logger=None):
    """Print all available models and their supported modes."""
    logger.info("=" * 70)
    logger.info("Available Models and Modes")
    logger.info("=" * 70)
    
    for model, modes in SCRIPT_PATHS.items():
        logger.info(f"Model: {model}")
        for mode, path in modes.items():
            status = "✓" if path and path.exists() else "✗"
            implemented = "implemented" if path else "not implemented"
            exists = "available" if (path and path.exists()) else "missing"
            logger.info(f"  {status} {mode:12s} - {implemented:16s} ({exists})")
    
    logger.info("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════
# Command-Line Interface
# ═══════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenSAR Pipeline - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
# List all available models and modes
python main_pipeline.py --list

# Train the large vessel detection point regression based model
python main_pipeline.py --model vd_large --mode train

# Run inference with the large vessel detection point regression based model
python main_pipeline.py --model vd_large --mode inference

# Train the RFI small model
python main_pipeline.py --model rfi_small --mode train

# Run the label conversion tool with custom arguments 
python main_pipeline.py --tool label_conversion --extra-args "-f all"

# Run the geocoding tool with custom arguments
python main_pipeline.py --tool geocoding --extra-args "-p DB_OPENSAR_VD_4330.xml"

# Run the DVD dataset generation script 
python main_pipeline.py --tool dataset_generation_dvd

# Run the flood dataset generation script
python main_pipeline.py --tool dataset_generation_flood

# Run the RFI dataset generation script 
python main_pipeline.py --tool dataset_generation_rfi

# Run the range compression demo 
python main_pipeline.py --tool range_compression

# Run end-to-end complexity profiling for VD large model
python main_pipeline.py --model vd_large --mode end2end
python main_pipeline.py --model vd_large --extra-args "end2end"

# Run end-to-end complexity profiling for VD small model
python main_pipeline.py --model vd_small --mode end2end
python main_pipeline.py --model vd_small --extra-args "end2end"

# Run end-to-end complexity profiling for RFI large model
python main_pipeline.py --model rfi_large --mode end2end
python main_pipeline.py --model rfi_large --extra-args "end2end"

# Run end-to-end complexity profiling for RFI small model
python main_pipeline.py --model rfi_small --mode end2end
python main_pipeline.py --model rfi_small --extra-args "end2end"
"""
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=list(SCRIPT_PATHS.keys()),
        help='Select the model/use-case to execute',
        required=False 
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'inference', 'evaluate', 'end2end'],
        help='Execution mode: train, inference, evaluate, or end2end',
        required=False 
    )
    
    parser.add_argument(
        '--tool',
        type=str,
        choices=['label_conversion', 'geocoding', 'dataset_generation_dvd', 'dataset_generation_flood', 'dataset_generation_rfi', 'range_compression'],
        help='Select a specific tool to run (e.g. label conversion, geocoding)',
        required=False
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available models and modes'
    )
    
    parser.add_argument(
        '--extra-args',
        type=str,
        nargs='?',
        help='Additional arguments to pass to the underlying script (in quotes)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the pipeline."""
    logger = get_logger()
    args = parse_args()
    
    # Handle list option
    if args.list:
        list_available_options(logger)
        return 0

    extra_args = None
    if args.extra_args:
        # Simple split - users should quote complex arguments
        extra_args = args.extra_args.split()
    
    # Handle end2end via --extra-args or --mode
    if args.model == 'vd_large' and (
        args.mode == 'end2end' or (extra_args and 'end2end' in extra_args)
    ):
        return end2end_vd_large(extra_args, logger)[0]

    elif args.model == 'vd_small' and (
        args.mode == 'end2end' or (extra_args and 'end2end' in extra_args)
    ):
        return end2end_vd_small(extra_args, logger)[0]

    if args.model == 'rfi_large' and (
        args.mode == 'end2end' or (extra_args and 'end2end' in extra_args)
    ):
        return end2end_rfi_large(extra_args, logger)[0]

    if args.model == 'rfi_small' and (
        args.mode == 'end2end' or (extra_args and 'end2end' in extra_args)
    ):
        return end2end_rfi_small(extra_args, logger)[0]

    # Execute the pipeline
    if args.model and args.mode:
        return execute_pipeline(args.model, args.mode, extra_args, logger)

    elif args.tool:
        return execute_tool(args.tool, extra_args, logger)

    else:
        logger.error("You must specify either a model and mode to execute, or a tool to run.")
        logger.info("Use --help for more information.")
        return 1

# Entry point
if __name__ == "__main__":
    sys.exit(main())