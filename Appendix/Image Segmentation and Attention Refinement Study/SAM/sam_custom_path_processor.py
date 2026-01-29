# sam_custom_path_processor.py
# Batch process images from any specified folder structure
# Usage examples:
#   # Process images from a custom folder
#   python sam_custom_path_processor.py --checkpoint sam_vit_h_4b8939.pth --model-type vit_h --input-path /path/to/images --output-path /path/to/output --compress --quality 75 --max-size 512
#   
#   # Process with recursive search and maintaining folder structure
#   python sam_custom_path_processor.py --checkpoint sam_vit_h_4b8939.pth --model-type vit_h --input-path /path/to/images --output-path /path/to/output --recursive --preserve-structure
#   
#   # Memory-conservative settings
#   python sam_custom_path_processor.py --checkpoint sam_vit_h_4b8939.pth --model-type vit_h --input-path /path/to/images --output-path /path/to/output --min-mask-region-area 10000 --points-per-side 10 --points-per-batch 32

import argparse
import json
import os
from pathlib import Path
import glob

import cv2
import numpy as np
from PIL import Image

import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def img_read_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def tight_bbox(mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1  # [x0,y0,x1,y1)

def avg_color_rgb(image_rgb: np.ndarray, mask_bool: np.ndarray):
    if mask_bool.sum() == 0:
        return [0.0, 0.0, 0.0]
    region = image_rgb[mask_bool]
    return [float(region[:, i].mean()) for i in range(3)]

def is_all_white_segment(image_rgb: np.ndarray, mask_bool: np.ndarray, white_threshold=240, white_ratio_threshold=0.95):
    """
    Check if a segment is predominantly white pixels
    
    Args:
        image_rgb: Original image in RGB format
        mask_bool: Boolean mask for the segment
        white_threshold: RGB value threshold to consider a pixel "white" (default: 240)
        white_ratio_threshold: Minimum ratio of white pixels to consider segment "all white" (default: 0.95)
    
    Returns:
        bool: True if segment should be discarded as all-white
    """
    if mask_bool.sum() == 0:
        return True  # Empty segment
    
    # Get the masked region
    region = image_rgb[mask_bool]
    
    # Check if pixels are white (all RGB values above threshold)
    white_pixels = np.all(region >= white_threshold, axis=1)
    white_ratio = np.sum(white_pixels) / len(region)
    
    return white_ratio >= white_ratio_threshold

def is_low_variance_segment(image_rgb: np.ndarray, mask_bool: np.ndarray, variance_threshold=100):
    """
    Check if a segment has very low color variance (potentially uninteresting)
    
    Args:
        image_rgb: Original image in RGB format
        mask_bool: Boolean mask for the segment
        variance_threshold: Maximum variance to consider segment "low variance"
    
    Returns:
        bool: True if segment should be discarded as low variance
    """
    if mask_bool.sum() == 0:
        return True
    
    region = image_rgb[mask_bool]
    
    # Calculate variance across all RGB channels
    total_variance = np.var(region.astype(np.float32))
    
    return total_variance < variance_threshold

def resize_image_if_needed(image, max_size):
    """Resize image if it's larger than max_size while maintaining aspect ratio"""
    if max_size is None:
        return image
    
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    
    # Calculate new dimensions
    if width > height:
        new_width = max_size
        new_height = int((height * max_size) / width)
    else:
        new_height = max_size
        new_width = int((width * max_size) / height)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def save_segment(i, mask_bool, image_rgb, outdir: Path, compress=False, quality=85, max_size=None, output_format="PNG", 
                 filter_white=True, filter_low_variance=False, white_threshold=240, white_ratio_threshold=0.95, variance_threshold=100):
    """Save segment with optional compression and filtering"""
    bbox = tight_bbox(mask_bool)
    area = int(mask_bool.sum())
    avg_rgb = [round(v, 2) for v in avg_color_rgb(image_rgb, mask_bool)]

    # Apply filters to discard uninteresting segments
    segment_filtered = False
    filter_reason = None
    
    if filter_white and is_all_white_segment(image_rgb, mask_bool, white_threshold, white_ratio_threshold):
        segment_filtered = True
        filter_reason = "all_white"
    elif filter_low_variance and is_low_variance_segment(image_rgb, mask_bool, variance_threshold):
        segment_filtered = True
        filter_reason = "low_variance"

    # Skip saving mask with transparency
    mask_path = None

    # Save crop (RGBA with transparency or compressed format)
    crop_path = None
    x0 = y0 = x1 = y1 = 0
    
    if bbox is not None and not segment_filtered:
        x0, y0, x1, y1 = bbox
        crop_rgb = image_rgb[y0:y1, x0:x1]
        crop_mask = mask_bool[y0:y1, x0:x1].astype(np.uint8) * 255
        
        if compress and output_format.upper() == "JPEG":
            # For JPEG compression, convert to RGB (no transparency)
            # Apply mask as white background
            crop_with_bg = np.zeros_like(crop_rgb)
            crop_with_bg[:] = [255, 255, 255]  # White background
            mask_3d = np.stack([crop_mask > 0] * 3, axis=-1)
            crop_with_bg = np.where(mask_3d, crop_rgb, crop_with_bg)
            crop_img = Image.fromarray(crop_with_bg.astype(np.uint8), mode="RGB")
            crop_path = outdir / f"seg_{i:03d}_crop.jpg"
        else:
            # PNG format with transparency
            crop_rgba = np.dstack([crop_rgb, crop_mask])
            crop_img = Image.fromarray(crop_rgba, mode="RGBA")
            crop_path = outdir / f"seg_{i:03d}_crop.png"
        
        # Resize if needed
        if max_size is not None:
            crop_img = resize_image_if_needed(crop_img, max_size)
        
        # Save with appropriate options
        save_kwargs = {}
        if compress:
            if output_format.upper() == "JPEG":
                save_kwargs["quality"] = quality
                save_kwargs["optimize"] = True
            elif output_format.upper() == "PNG":
                save_kwargs["optimize"] = True
                save_kwargs["compress_level"] = 9  # Maximum PNG compression
        
        crop_img.save(crop_path, **save_kwargs)
    elif segment_filtered:
        # Set bbox to zeros for filtered segments
        x0 = y0 = x1 = y1 = 0

    return {
        "id": i,
        "area_px": area,
        "bbox_xyxy": [x0, y0, x1, y1],
        "avg_color_rgb_0_255": avg_rgb,
        "mask_path": mask_path,
        "crop_path": str(crop_path) if crop_path else None,
        "filtered": segment_filtered,
        "filter_reason": filter_reason,
    }

def find_images_in_path(input_path, recursive=True):
    """Find all images in the specified path"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    found_images = []
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return found_images
    
    print(f"Searching for images in: {input_path}")
    
    for ext in image_extensions:
        if recursive:
            # Recursive search
            images = list(input_path.glob(f"**/{ext}"))
        else:
            # Only search in the specified directory
            images = list(input_path.glob(ext))
        
        for img_path in images:
            # Calculate relative path from input directory
            try:
                relative_path = img_path.relative_to(input_path)
                found_images.append({
                    'path': img_path,
                    'relative_path': relative_path,
                    'parent_dirs': relative_path.parts[:-1] if len(relative_path.parts) > 1 else ()
                })
                print(f"  Found: {relative_path}")
            except ValueError:
                # If relative_to fails, just use the filename
                found_images.append({
                    'path': img_path,
                    'relative_path': img_path.name,
                    'parent_dirs': ()
                })
                print(f"  Found: {img_path.name}")
    
    return found_images

def create_output_path(image_info, output_base_path, preserve_structure=True):
    """Create output path for SAM results"""
    output_base = Path(output_base_path)
    
    # Get image name without extension
    image_name = image_info['path'].stem
    
    if preserve_structure and image_info['parent_dirs']:
        # Preserve the folder structure
        output_dir = output_base
        for parent_dir in image_info['parent_dirs']:
            output_dir = output_dir / parent_dir
        output_dir = output_dir / image_name
    else:
        # Flat structure - all results in the output directory
        output_dir = output_base / image_name
    
    return output_dir

def process_single_image(image_path, sam_model, output_dir, args):
    """Process a single image with SAM"""
    try:
        # Clear CUDA cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete
        
        # Load image
        image_rgb = img_read_rgb(str(image_path))
        original_height, original_width = image_rgb.shape[:2]
        
        # Check image dimensions and resize if too large for CUDA
        max_dimension = getattr(args, 'max_image_size', 2048)  # Use args value or default
        if max(original_height, original_width) > max_dimension:
            print(f"  Warning: Large image ({original_width}x{original_height}), resizing for CUDA compatibility")
            scale_factor = max_dimension / max(original_height, original_width)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            print(f"  Resized to: {new_width}x{new_height}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use more conservative parameters for problematic images
        points_per_side = args.points_per_side
        points_per_batch = args.points_per_batch
        
        # Reduce parameters for very large images or if we've hit CUDA errors before
        image_pixels = image_rgb.shape[0] * image_rgb.shape[1]
        if image_pixels > 1024 * 1024:  # 1MP+
            points_per_side = min(points_per_side, 16)
            points_per_batch = min(points_per_batch, 32)
            print(f"  Using conservative parameters: points_per_side={points_per_side}, points_per_batch={points_per_batch}")
        
        # Setup SAM for automatic mask generation with error handling
        try:
            mask_generator = SamAutomaticMaskGenerator(
                model=sam_model,
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=args.pred_iou_thresh,
                stability_score_thresh=args.stability_score_thresh,
                min_mask_region_area=args.min_mask_region_area,
            )
            
            # Generate masks with fallback strategy
            masks = mask_generator.generate(image_rgb)
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "invalid configuration" in str(e):
                print(f"  CUDA error encountered, trying with reduced parameters...")
                # Clear CUDA cache and try with very conservative settings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Fallback to very conservative parameters
                mask_generator = SamAutomaticMaskGenerator(
                    model=sam_model,
                    points_per_side=8,  # Much smaller
                    points_per_batch=16,  # Much smaller
                    pred_iou_thresh=args.pred_iou_thresh,
                    stability_score_thresh=args.stability_score_thresh,
                    min_mask_region_area=args.min_mask_region_area,
                )
                
                masks = mask_generator.generate(image_rgb)
                print(f"  Success with fallback parameters")
            else:
                raise
        
        # Process and save segments
        meta = {
            "original_image": str(image_path.resolve()),
            "model_type": args.model_type,
            "device": args.device,
            "compression": {
                "enabled": args.compress,
                "quality": args.quality if args.compress else None,
                "max_size": args.max_size,
                "format": args.format
            },
            "sam_parameters": {
                "points_per_side": points_per_side,  # Use actual values used
                "points_per_batch": points_per_batch,  # Use actual values used
                "pred_iou_thresh": args.pred_iou_thresh,
                "stability_score_thresh": args.stability_score_thresh,
                "min_mask_region_area": args.min_mask_region_area,
            },
            "segments": []
        }
        
        for i, m in enumerate(masks):
            mask_bool = m["segmentation"].astype(bool)
            entry = save_segment(i, mask_bool, image_rgb, output_dir, 
                               compress=args.compress, quality=args.quality, 
                               max_size=args.max_size, output_format=args.format,
                               filter_white=args.filter_white, filter_low_variance=args.filter_low_variance,
                               white_threshold=args.white_threshold, white_ratio_threshold=args.white_ratio_threshold,
                               variance_threshold=args.variance_threshold)
            entry.update({
                "score_pred_iou": float(m.get("predicted_iou", 0.0)),
                "stability_score": float(m.get("stability_score", 0.0)),
            })
            meta["segments"].append(entry)
        
        # Count filtered segments
        total_segments = len(meta["segments"])
        filtered_segments = sum(1 for seg in meta["segments"] if seg.get("filtered", False))
        saved_segments = total_segments - filtered_segments
        
        # Add filtering statistics to metadata
        meta["filtering_stats"] = {
            "total_segments_generated": total_segments,
            "segments_filtered": filtered_segments,
            "segments_saved": saved_segments,
            "filtering_enabled": {
                "white_filter": args.filter_white,
                "low_variance_filter": args.filter_low_variance
            },
            "filtering_parameters": {
                "white_threshold": args.white_threshold,
                "white_ratio_threshold": args.white_ratio_threshold,
                "variance_threshold": args.variance_threshold
            }
        }
        
        # Save metadata
        with open(output_dir / "segments.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        
        # Store segment count before clearing
        segment_count = saved_segments  # Return only saved segments
        
        # Print filtering info if segments were filtered
        if filtered_segments > 0:
            print(f"  Filtered {filtered_segments} segments (saved {saved_segments}/{total_segments})")
        
        # Clear variables and CUDA cache after processing
        del masks, mask_generator, image_rgb, meta
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return segment_count
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        
        # Enhanced error handling for CUDA errors
        if "CUDA" in str(e) or "invalid configuration" in str(e):
            print(f"  CUDA-related error detected. Attempting recovery...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Force garbage collection
                import gc
                gc.collect()
                print(f"  CUDA cache cleared and synchronized")
        
        # Clear CUDA cache even on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 0

def main():
    ap = argparse.ArgumentParser("SAM Custom Path Processor")
    ap.add_argument("--checkpoint", required=True, help="Path to SAM .pth checkpoint")
    ap.add_argument("--model-type", required=True, choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--input-path", required=True, help="Input path containing images to process")
    ap.add_argument("--output-path", required=True, help="Output path for SAM results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Input/Output options
    ap.add_argument("--recursive", action="store_true", default=True, help="Search for images recursively (default: True)")
    ap.add_argument("--preserve-structure", action="store_true", default=True, help="Preserve folder structure in output (default: True)")
    ap.add_argument("--flat-output", action="store_true", help="Output all results in flat structure (overrides --preserve-structure)")
    
    # Compression options
    ap.add_argument("--compress", action="store_true", help="Enable image compression")
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality (1-100, default: 85)")
    ap.add_argument("--max-size", type=int, default=512, help="Maximum image dimension in pixels (default: 512)")
    ap.add_argument("--format", choices=["PNG", "JPEG"], default="PNG", 
                    help="Output format (PNG with transparency or JPEG with white background)")
    
    # SAM parameters
    ap.add_argument("--pred-iou-thresh", type=float, default=0.88)
    ap.add_argument("--stability-score-thresh", type=float, default=0.95)
    ap.add_argument("--points-per-side", type=int, default=32)
    ap.add_argument("--points-per-batch", type=int, default=64, help="Reduce for lower memory usage")
    ap.add_argument("--min-mask-region-area", type=int, default=10000)
    
    # Processing options
    ap.add_argument("--skip-existing", action="store_true", help="Skip images that already have output folders")
    ap.add_argument("--max-image-size", type=int, default=2048, help="Maximum image dimension before resizing (default: 2048)")
    ap.add_argument("--force-cpu", action="store_true", help="Force CPU processing (useful for CUDA issues)")
    
    # Filtering options
    ap.add_argument("--filter-white", action="store_true", default=True, help="Filter out all-white segments (default: True)")
    ap.add_argument("--no-filter-white", action="store_true", help="Disable white segment filtering")
    ap.add_argument("--white-threshold", type=int, default=240, help="RGB threshold for white pixels (0-255, default: 240)")
    ap.add_argument("--white-ratio-threshold", type=float, default=0.95, help="Minimum ratio of white pixels to filter segment (default: 0.95)")
    ap.add_argument("--filter-low-variance", action="store_true", help="Filter out low-variance segments")
    ap.add_argument("--variance-threshold", type=float, default=100, help="Maximum variance for low-variance filtering (default: 100)")
    
    args = ap.parse_args()

    # Handle filter options
    if args.no_filter_white:
        args.filter_white = False

    # Override device if force-cpu is specified
    if args.force_cpu:
        args.device = "cpu"
        print("Forcing CPU processing due to --force-cpu flag")

    # Override preserve-structure if flat-output is specified
    if args.flat_output:
        args.preserve_structure = False
        print("Using flat output structure")

    # Validate arguments
    if args.quality < 1 or args.quality > 100:
        raise ValueError("Quality must be between 1 and 100")
    
    # Set environment variable for better CUDA debugging if needed
    if args.device == "cuda":
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error reporting
    
    print("SAM Custom Path Processor")
    print("=" * 50)
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print(f"Recursive search: {args.recursive}")
    print(f"Preserve structure: {args.preserve_structure}")
    
    # Find all images in the specified path
    print("\n1. Searching for images...")
    found_images = find_images_in_path(args.input_path, recursive=args.recursive)
    
    if not found_images:
        print("No images found in the specified path!")
        return
    
    print(f"Found {len(found_images)} images total")
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load SAM model
    print("\n2. Loading SAM model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    print(f"Model loaded on {args.device}")
    
    # Clear initial CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Initial CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Process each image
    print("\n3. Processing images...")
    processed_count = 0
    skipped_count = 0
    total_segments = 0
    
    for i, image_info in enumerate(found_images, 1):
        image_path = image_info['path']
        output_dir = create_output_path(image_info, args.output_path, args.preserve_structure)
        
        print(f"\n[{i}/{len(found_images)}] Processing: {image_info['relative_path']}")
        print(f"  Output: {output_dir}")
        
        # Check if already processed
        if args.skip_existing and output_dir.exists() and (output_dir / "segments.json").exists():
            print(f"  Skipped (already exists)")
            skipped_count += 1
            continue
        
        # Process the image with fallback to CPU if CUDA fails
        segments_count = process_single_image(image_path, sam, output_dir, args)
        
        # If CUDA processing failed and we're on CUDA, try CPU fallback
        if segments_count == 0 and args.device == "cuda":
            print(f"  Attempting CPU fallback...")
            # Create CPU model for fallback
            try:
                sam_cpu = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
                sam_cpu.to(device="cpu")
                
                # Temporarily change args device for this image
                original_device = args.device
                args.device = "cpu"
                
                segments_count = process_single_image(image_path, sam_cpu, output_dir, args)
                
                # Restore original device
                args.device = original_device
                
                # Clean up CPU model
                del sam_cpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if segments_count > 0:
                    print(f"  CPU fallback successful: {segments_count} segments generated")
                else:
                    print(f"  CPU fallback also failed")
                    
            except Exception as e:
                print(f"  CPU fallback failed: {str(e)}")
                args.device = original_device
        
        if segments_count > 0:
            processed_count += 1
            total_segments += segments_count
            print(f"  Success: {segments_count} segments generated")
        else:
            print(f"  Failed to process")
        
        # Show memory usage if on CUDA
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"  Memory: {memory_used:.2f}GB used, {memory_cached:.2f}GB cached")
    
    # Summary
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print(f"Images found: {len(found_images)}")
    print(f"Images processed: {processed_count}")
    print(f"Images skipped: {skipped_count}")
    print(f"Total segments generated: {total_segments}")
    
    compression_info = ""
    if args.compress:
        compression_info = f" (compressed: {args.format}, quality: {args.quality}"
        if args.max_size:
            compression_info += f", max-size: {args.max_size}"
        compression_info += ")"
    
    print(f"Output saved in: {args.output_path}{compression_info}")

if __name__ == "__main__":
    main()
