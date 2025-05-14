from PIL import Image
import os

# Define the base path and image names
base_path = "/users/vsharm44/projects/ssl-bioacoustics/pca_kmeans_results"
image_names = ["8clusters.png", "16clusters.png", "32clusters.png", "64clusters.png", "128clusters.png"]

# Open all images
images = []
for img_name in image_names:
    img_path = os.path.join(base_path, img_name)
    if os.path.exists(img_path):
        images.append(Image.open(img_path))
    else:
        print(f"Warning: {img_path} not found")

if not images:
    print("No images found to combine")
    exit(1)

# Get dimensions of the first image
width, height = images[0].size

# Create a new image with the combined width
total_width = width * 2
total_height = height * 3
combined_image = Image.new('RGB', (total_width, total_height))

# Paste each image side by side
for i, img in enumerate(images):
    width_idx = i % 2
    height_idx = i // 2
    combined_image.paste(img, (width_idx * width, height_idx * height))

# Save the combined image
output_path = os.path.join(base_path, "combined_clusters.png")
combined_image.save(output_path)
print(f"Combined image saved to: {output_path}")

# Close all images
for img in images:
    img.close() 