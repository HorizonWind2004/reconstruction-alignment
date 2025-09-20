# -*- coding: utf-8 -*-
import os
import torch
import shutil
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import time
import concurrent.futures
import argparse # Import the argparse module

# --- Global Configuration (Adjust as needed) ---

# Filenames to be converted to bfloat16
FILES_TO_CONVERT = {"ema.safetensors", "ema.safetensors"}

# Maximum number of processes to use for parallel processing (None means use all available CPU cores)
MAX_WORKERS = None # Can be set to a specific number, e.g., 4

# --- Worker Function (for parallel processing) ---

def convert_file_to_bf16(file_info):
    """
    Converts a single file to bfloat16 format in a single process.
    This is a standalone function to facilitate parallelization.
    
    Args:
        file_info (tuple): A tuple containing (filename, source_folder, target_folder).
        
    Returns:
        str: A message describing the result of the operation.
    """
    filename, source_folder, target_folder = file_info
    source_path = os.path.join(source_folder, filename)
    target_path = os.path.join(target_folder, filename)
    
    try:
        # Load the weights file to the CPU to avoid using GPU memory
        tensors = load_file(source_path, device="cpu")
        tensors_bf16 = {}
        
        # Use tqdm to show the conversion progress of tensors within a single file
        # leave=False means the progress bar will disappear upon completion
        item_iterator = tqdm(tensors.items(), desc=f"   -> Converting '{filename}'", leave=False, position=1)
        for k, v in item_iterator:
            # Convert the tensor to bfloat16 type
            tensors_bf16[k] = v.to(torch.bfloat16)
        
        # Save the converted file
        save_file(tensors_bf16, target_path)
        return f"✅ [Subprocess] Successfully converted and saved: '{target_path}'"
        
    except Exception as e:
        return f"❌ [Subprocess] Error processing '{filename}': {e}"


# --- Main Script ---

def main(source_folder, master_model_folder, target_folder):
    """
    The main function that executes the entire model conversion and weight completion process.
    
    Args:
        source_folder (str): Path to the source folder containing the original checkpoints.
        master_model_folder (str): Path to the folder containing the master model's configuration, tokenizer, etc.
        target_folder (str): Path to the target folder for storing the processed files.
    """
    start_time = time.time()
    print("🚀 Starting the model processing script (parallel accelerated version)...")
    print(f"Source checkpoint folder: {source_folder}")
    print(f"Source master model folder: {master_model_folder}")
    print(f"Target folder: {target_folder}")
    print("-" * 60)

    # Ensure the target folder exists, create it if it doesn't
    os.makedirs(target_folder, exist_ok=True)
    
    # Dynamically construct master_ema_path based on the provided master_model_folder
    master_ema_path = os.path.join(master_model_folder, "ema.safetensors")


    # --- Step 1: Copy non-weight files (e.g., config, tokenizer) ---
    print("\n--- Step 1: Copying non-weight files (e.g., config, tokenizer) ---")
    try:
        master_model_files = os.listdir(master_model_folder)
        
        # Use tqdm to show file copy progress
        copy_iterator = tqdm(master_model_files, desc="Copying non-weight files")
        
        copied_count = 0
        for filename in copy_iterator:
            # Skip all weight files, only copy other file types
            if filename.endswith('ema.safetensors'):
                continue

            src_path = os.path.join(master_model_folder, filename)
            dst_path = os.path.join(target_folder, filename)
            
            # Ensure we are copying a file, not a subdirectory
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                copied_count += 1
        
        print(f"✅ Successfully copied {copied_count} non-weight files to '{target_folder}'")

    except FileNotFoundError:
        print(f"❌ Error: Master model folder not found: {master_model_folder}")
    except Exception as e:
        print(f"❌ An error occurred while copying files: {e}")

    print("\n--- Step 1 Complete ---")
    print("-" * 60)


    # --- Step 2: Convert specified files to bfloat16 format in parallel ---
    print("\n--- Step 2: Parallel conversion of weight files to bfloat16 ---")
    
    try:
        all_source_files = os.listdir(source_folder)
    except FileNotFoundError:
        print(f"❌ Error: Source checkpoint folder not found: {source_folder}")
        return

    # Filter out the files that need conversion
    files_to_process = [f for f in all_source_files if f in FILES_TO_CONVERT]
    
    if not files_to_process:
        print("🟡 No weight files to convert were found in the source checkpoint folder.")
    else:
        tasks = [(filename, source_folder, target_folder) for filename in files_to_process]
        print(f"📨 Submitting {len(tasks)} weight file conversion tasks to the process pool...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(tqdm(executor.map(convert_file_to_bf16, tasks), total=len(tasks), desc="Parallel Conversion Progress"))
        
        print("\n--- Parallel Conversion Results ---")
        for res in results:
            print(res)
        print("----------------------")

    print("\n--- Step 2 Complete ---")
    print("-" * 60)

    # --- Step 3: Complete missing weights for ema.safetensors ---
    print("\n--- Step 3: Completing missing weights for ema.safetensors ---")
    
    target_model_path = os.path.join(target_folder, "ema.safetensors")
    
    if not os.path.exists(target_model_path):
        print(f"⚠️  Warning: Target model '{target_model_path}' not found. Skipping weight completion step.")
    elif not os.path.exists(master_ema_path):
        print(f"⚠️  Warning: Weight source (Master) '{master_ema_path}' not found. Skipping weight completion step.")
    else:
        try:
            print("Loading Master model and Target ema...")
            master_ema_tensors = load_file(master_ema_path, device="cpu")
            target_model_tensors = load_file(target_model_path, device="cpu")

            master_keys = set(master_ema_tensors.keys())
            target_keys = set(target_model_tensors.keys())
            
            missing_keys = master_keys - target_keys

            if not missing_keys:
                print("✅ Weights in 'ema.safetensors' are complete, no completion needed.")
            else:
                print(f"🟡 Found {len(missing_keys)} missing weights, starting completion...")
                
                merged_tensors = target_model_tensors.copy()

                key_iterator = tqdm(sorted(list(missing_keys)), desc="   -> Completing missing weights")
                for key in key_iterator:
                    merged_tensors[key] = master_ema_tensors[key].to(torch.bfloat16)
                
                print(f"💾 Saving the completed model, total weights: {len(merged_tensors)}...")
                save_file(merged_tensors, target_model_path)
                print(f"✅ Successfully completed and saved to: '{target_model_path}'")

        except Exception as e:
            print(f"❌ An error occurred during weight completion: {e}")

    print("\n--- Step 3 Complete ---")
    print("-" * 60)

    end_time = time.time()
    print(f"🎉 All tasks completed. Total time taken: {end_time - start_time:.2f} seconds.")
    
    flag_path = os.path.join(target_folder, "processing_complete.txt")
    with open(flag_path, "w", encoding="utf-8") as f:
        f.write(f"Processing completed at: {time.ctime()}.\n")
        f.write("All non-weight files have been copied.\n")
        f.write(f"Converted {FILES_TO_CONVERT} to bfloat16 format.\n")
        f.write("Completed the weights for ema.safetensors using the Master EMA ema.\n")
    print(f"📄 Flag file created: '{flag_path}'")


if __name__ == "__main__":
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Convert training checkpoints to Hugging Face format and complete weights.")
    parser.add_argument(
        "--training_checkpoint_path",
        type=str,
        required=True,
        help="Path to the source folder containing original checkpoints (e.g., ema.safetensors)."
    )
    parser.add_argument(
        "--template_model_path",
        type=str,
        required=True,
        help="Path to the folder containing the master model's config, tokenizer, and complete ema.safetensors."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the target folder for storing the processed files."
    )
    args = parser.parse_args()

    # When using 'spawn' or 'forkserver' start methods on Windows or macOS,
    # the main logic must be placed inside the if __name__ == "__main__": block.
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # If the start method is already set, ignore the error
        pass
        
    # Call the main function using arguments parsed from the command line
    main(
        source_folder=args.training_checkpoint_path,
        master_model_folder=args.template_model_path,
        target_folder=args.output_path
    )