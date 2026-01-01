import kagglehub

# Download latest version
path = kagglehub.dataset_download("wiratrnn/banana-ripeness-image-dataset")

print("Path to dataset files:", path)