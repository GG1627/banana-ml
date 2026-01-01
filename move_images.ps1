# Get the base paths
$sourcePath = ".\banana-ripeness-image-dataset\versions\1\DATASETS\ambon"
$destinationPath = ".\data"

# Find all .jpg files recursively in the ambon folder
Get-ChildItem -Path $sourcePath -Recurse -Include "*.jpg" | ForEach-Object {
    $file = $_
    
    # Find the H folder in the path (H1, H2, H3, etc.)
    $pathParts = $file.FullName -split '\\'
    $hFolder = $pathParts | Where-Object { $_ -match '^H\d+$' } | Select-Object -First 1
    
    if ($hFolder) {
        # Use the H folder name directly (H1 stays H1, H2 stays H2, etc.)
        $destFolderName = $hFolder
        $destFolderPath = Join-Path $destinationPath $destFolderName
        
        # Create destination folder if it doesn't exist
        if (!(Test-Path $destFolderPath)) {
            New-Item -ItemType Directory -Path $destFolderPath -Force
        }
        
        # Move the file
        $destFilePath = Join-Path $destFolderPath $file.Name
        Move-Item -Path $file.FullName -Destination $destFilePath -Force
        
        Write-Host "Moved: $($file.Name) -> $destFolderName\"
    }
}

Write-Host "All images have been moved successfully!"