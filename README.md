# Juvenile Salmon Size Estimation

Simple workflow for estimating juvenile salmon size using stereo camera data from ZED recordings.

## Project overview

- `extract_data/` processes ZED `.svo` source files and extracts image data.
- `clear_image.py` cleans extracted left/right images in the `HD720_SN15044540_14-44-20/masked_image/left` and `.../right` folders.
- `extract_length.py` computes fish length from the cleaned dataset.
- `size.ipynb` repeats the same extraction and measurement workflow with visualization for each step and results.

## Main workflow

1. Run `extract_data/main.py` to extract TIFF images and left/right stereo pairs from ZED `.svo` files located in the `HD720_SN15044540_14-44-20/` folder.
2. Use `clear_image.py` to clean the extracted images in `HD720_SN15044540_14-44-20/masked_image/left` and `HD720_SN15044540_14-44-20/masked_image/right`.
3. Run `extract_length.py` on the cleaned images to compute fish length measurements.
4. Open `size.ipynb` to inspect the same process interactively, with visualizations for each step and the final measurement results.

## Notes

- `HD720_SN15044540_14-44-20/` contains example sample data needed to run the extraction and length measurement pipeline.
- The source data comes from ZED stereo videos collected with a ZED camera called HD720_SN15044540_14-44-20. (collected in April )

## Example use of code 

extract_length.py --images HD720_SN15044540_14-44-20/masked_image/left/ --depths HD720_SN15044540_14-44-20/justtiff/ --output lengthresult --yolo best.pt 