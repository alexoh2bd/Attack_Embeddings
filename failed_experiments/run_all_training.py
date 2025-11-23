#!/usr/bin/env python3
"""run_all_training.py

Orchestrates the training of both MAT and FAT models sequentially.
Captures full logs for each training run.

Usage:
    python run_all_training.py
"""

import subprocess
import os
import time
import sys

def run_command(command, log_file):
    """Run a command and pipe output to a log file and stdout."""
    print(f"\n{'='*60}")
    print(f"STARTING: {command}")
    print(f"LOGGING TO: {log_file}")
    print(f"{'='*60}\n")
    
    with open(log_file, "w") as f:
        # Use unbuffered python execution
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output to both console and file
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()
            
        process.wait()
        
    if process.returncode != 0:
        print(f"\n❌ COMMAND FAILED with exit code {process.returncode}")
        return False
    
    print(f"\n✅ COMMAND COMPLETED SUCCESSFULLY")
    return True

def main():
    # Create logs directory
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    python_executable = sys.executable  # Use current python interpreter
    
    # 1. Run MAT Training
    mat_log = f"logs/train_mat_{timestamp}.log"
    mat_cmd = f"{python_executable} defend_clip_mat.py --effective_batch_size 16 --micro_batch_size 2 --max_steps 1000"
    
    if not run_command(mat_cmd, mat_log):
        print("Stopping execution due to MAT failure.")
        sys.exit(1)
        
    # 2. Run FAT Training
    fat_log = f"logs/train_fat_{timestamp}.log"
    fat_cmd = f"{python_executable} defend_clip_fat.py --effective_batch_size 16 --micro_batch_size 2 --max_steps 1000"
    
    if not run_command(fat_cmd, fat_log):
        print("Stopping execution due to FAT failure.")
        sys.exit(1)
        
    print(f"\n{'='*60}")
    print("🎉 ALL TRAINING COMPLETE!")
    print(f"MAT Logs: {mat_log}")
    print(f"FAT Logs: {fat_log}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
