#!/usr/bin/env python3
"""
Direct3D-S2 Inference Script
Generate 3D meshes from images.

Usage:
    python inference.py
"""

from direct3d_s2.pipeline import Direct3DS2Pipeline
import os
import glob
import random
import time
import pandas as pd
from tqdm import tqdm


def infer_directory(input_dir, output_dir, pipeline, **kwargs):
    """
    Process all images in input_dir and save meshes to output_dir.
    Filters by target_ids from pairs.csv if available.
    Skips already processed images and randomizes processing order.
    """
    print(f"Processing images from directory: {input_dir}")

    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Check for pairs.csv to filter by target_id
    pairs_csv_path = os.path.join(os.path.dirname(input_dir.rstrip('/')), 'pairs.csv')
    target_ids = None
    if os.path.exists(pairs_csv_path):
        print(f"Found pairs.csv at {pairs_csv_path}")
        try:
            pairs_df = pd.read_csv(pairs_csv_path)
            if 'target_id' in pairs_df.columns:
                target_ids = set(pairs_df['target_id'].astype(str).unique())
                print(f"Filtering by {len(target_ids)} unique target_ids from pairs.csv")
            else:
                print("Warning: pairs.csv found but 'target_id' column not present")
        except Exception as e:
            print(f"Warning: Failed to read pairs.csv: {e}")

    # Find all images
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    image_paths = sorted(list(set(image_paths)))
    
    # Filter by target_ids if pairs.csv was found
    if target_ids is not None:
        filtered_paths = []
        for image_path in image_paths:
            sample_id = os.path.splitext(os.path.basename(image_path))[0]
            if sample_id in target_ids:
                filtered_paths.append(image_path)
        print(f"Filtered {len(image_paths)} images down to {len(filtered_paths)} based on target_ids")
        image_paths = filtered_paths

    if len(image_paths) == 0:
        print(f"No images found in {input_dir}")
        return

    # Filter out images that already have output meshes
    images_to_process = []
    output_format = kwargs.get('output_format', 'glb')
    for image_path in image_paths:
        sample_id = os.path.splitext(os.path.basename(image_path))[0]
        sample_output_dir = os.path.join(output_dir, sample_id)
        output_path = os.path.join(sample_output_dir, f"shape_mesh.{output_format}")
        
        if not os.path.exists(output_path):
            images_to_process.append(image_path)
    
    total_images = len(image_paths)
    already_processed = total_images - len(images_to_process)
    
    print(f"Total images: {total_images}")
    print(f"Already processed: {already_processed}")
    print(f"Remaining to process: {len(images_to_process)}")
    
    if len(images_to_process) == 0:
        print("All images have already been processed!")
        return

    # Randomize processing order using current timestamp as seed
    random.seed(int(time.time()))
    random.shuffle(images_to_process)
    print(f"Randomized processing order (seed: {int(time.time())})")

    success_count = 0
    for image_path in tqdm(images_to_process, desc="Processing images"):
        # Get sample_id from image filename (without extension)
        sample_id = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create sample-specific output directory
        sample_output_dir = os.path.join(output_dir, sample_id)
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # Output mesh always named shape_mesh.glb
        output_path = os.path.join(sample_output_dir, f"shape_mesh.{output_format}")
        
        # Double-check if the output already exists (in case another process created it)
        if os.path.exists(output_path):
            print(f"Skipping {sample_id}: output already exists")
            continue
        
        try:
            print(f"\nProcessing: {image_path}")
            mesh = pipeline(
                image_path, 
                sdf_resolution=kwargs.get('sdf_resolution', 512),
                remove_interior=kwargs.get('remove_interior', True),
                remesh=kwargs.get('remesh', True),
                simplify_ratio=kwargs.get('simplify_ratio', 0.925),
            )["mesh"]
            mesh.export(output_path)
            print(f"  Mesh saved to: {output_path}")
            success_count += 1
        except Exception as e:
            print(f"  Error processing image: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✓ Processing complete! {success_count}/{len(images_to_process)} meshes generated successfully")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    # Initialize pipeline
    pipeline = Direct3DS2Pipeline.from_pretrained(
        'wushuang98/Direct3D-S2', 
        subfolder="direct3d-s2-v-1-1"
    )
    pipeline.to("cuda:0")

    # Dataset configuration
    data_dir = "/workspace/outputs/"
    images_dir = os.path.join(data_dir, "hair_aligned_image")
    output_meshes_dir = os.path.join(data_dir, "direct3d_s2")

    # Run inference
    infer_directory(
        input_dir=images_dir,
        output_dir=output_meshes_dir,
        pipeline=pipeline,
        sdf_resolution=512,  # 512 or 1024
        remove_interior=True,
        remesh=True,  # Switch to True if you need to reduce the number of triangles.
        simplify_ratio=0.925,
        output_format='glb',
    )