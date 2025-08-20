import kagglehub

# Download latest version
path = kagglehub.dataset_download("kandij/mall-customers")

print("Path to dataset files:", path)