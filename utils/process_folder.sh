#!/bin/bash

# Check if a folder path was provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/your/folder"
  exit 1
fi

# Define the input folder and the name for the copied folder
input_folder="$1"
copy_folder="email_$(basename "$1")_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$copy_folder"

# Copy the folder except the local cifar copy in sub_dir_like_0/data/CIFAR10/cifar-10-batches-py
rsync -av --exclude="*.ckpt" --exclude="*.pkl" --exclude="*.tar.gz" --exclude="*/*/*/cifar-*" "$input_folder/" "$copy_folder/"

# Compress the copied folder using tar
tar -czvf "${copy_folder}.tar.gz" "$copy_folder"

# Output completion message
echo "Folder copied, files removed, and zipped as ${copy_folder}.tar.gz"
