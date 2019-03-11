# Liscence-Plate-Number-Detection

Requirements:
1. Tesseract Engine
2. Python Libraries

Run the extract_data.py file in command prompt 
Sample: python extract_data.py --input_dir inputs/ --output_dir outputs/

Image from which the text is to be extracted are to be placed in input directory

The code here applies 7 filters to the image and after each filter the image is given as input to the tesseract engine to detect text
