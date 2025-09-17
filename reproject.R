library(terra)
library(fs)

crs_target <- "EPSG:4326"
root_dir <- "D:/SEBAL/datasets/validation/LBDC_validations/150039/"

tif_files <- list.files(root_dir, pattern = "\\.tif$", full.names = TRUE)
total_files <-length(tif_files)
print(paste("Total .tif files found:", total_files))

# Iterate through each file
for (i in seq_along(tif_files)) {
  file <- tif_files[i]
  # print(paste("Processing:", file))
  message("[", i, "/", total_files, "] Processing: ", file)
  
  # Read the raster
  raster <- rast(file)
  
  # Reproject the raster to the target CRS
  raster_reprojected <- project(raster, crs_target)
  
  # Overwrite the original file with the reprojected raster
  writeRaster(raster_reprojected, file, overwrite = TRUE)
}

print("All files processed and reprojected.")

# Cleanup: Remove environment variables and reset R session
rm(list = ls())
gc()  # Garbage collection to free memory
cat("\014")  # Clear the console (works in RStudio)
print("Environment cleaned up.")
