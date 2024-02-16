#!/bin/zsh

# Directory containing the PDF files
pdf_dir="/Users/bytedance/Projects/DeVAn/webpage/assets"

# Change to the directory
cd "$pdf_dir" || exit

# Loop through all PDF files in the directory
for pdf_file in *.pdf; do
  # Use ImageMagick's convert command to convert PDF to PNG
  # This command assumes ImageMagick's policy allows PDF processing
  # If you encounter issues, you might need to adjust ImageMagick's policy file
  convert "$pdf_file" "${pdf_file:r}.png"
done
