# OMR: Optical Music Recognition
Image Processing (_CMP362_) course project, Fall 2020.

## Instructions
1. `conda env create -f requirements.txt`
2. `python main.py​ <input-folder> <output-folder>`  
Where ​`<input-folder>`​ is the absolute path for input folder which contains the input
samples and ​`<output-folder>`​ is the absolute path for the output folder.

## Project Description
- Given a set of images containing music sheets, for each image:  
    - Extract sheet music
    - Represent sheet music in the [Guido Music Notation](https://guidodoc.grame.fr/)

## Our Approach
- The Optical Music Recognition problem involves images where the locus of information lies only
in a subset of the input image. This leaves room for a lot of preprocessing to facilitate operation
on the input image.
- The preprocessing step involves thresholding the image and inverting it, representing information as
**true** values in the image.
- It also involves deskewing the image, in case it is camera-captured. Through dilation and 
edge detection, a box around the staff is formed. By extracting the largest **convex hull**,
we extract the border of the music sheet. This border is then used to apply a perspective transform.
- For our first processing step, we removed staff lines, clefs, barlines, and other such elements
which do not affect the final output.
- Afterwards, we **segmented** the image into notes, and split connected beamed notes into separate segments.
- Each segment was later passed to a classifier pipeline, which assigned it its **pitch and time**.
- Finally, each note was output in the Guido Notation.

## Classification
- We used a tree-like hierarchy for classification where the input is a base component, and the final output is a fully classified element.
- First we determine if the component is a meter and classify its time, otherwise we determine if it’s an accidental and classify its type.
- If the component is neither a meter nor an accidental, it is therefore a note, and the next step is to determine whether it’s beamed, flagged, or hollow. Once its type is determined, it’s then passed to the corresponding classifier which classifies its timing.
- Each classifier was trained on a sizable dataset (~3000 images), which we gathered by generating sheets using the Guido Editor, extracting the preview of these sheets, and segmenting them using our algorithm. A process which we automated using JavaScript and Python. 
- Then we serialized the trained models and saved them in a binary format, which we load each time we run the script.
