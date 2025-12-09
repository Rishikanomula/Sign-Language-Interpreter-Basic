import kagglehub

path = kagglehub.dataset_download(
    "prasadshet/indian-sign-language-video-dataset"
)

print("Path to dataset files:", path)
