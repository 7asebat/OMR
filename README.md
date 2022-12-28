# OMR: Optical Music Recognition
Image Processing (_CMP362_) course project, Fall 2020.

## Instructions
1. `conda env create -f requirements.txt`
2. `python main.py​ <input-folder> <output-folder>`  
Where ​`<input-folder>`​ is the absolute path for input folder which contains the input
samples and ​`<output-folder>`​ is the absolute path for the output folder.

## Project Overview
- Given a set of images containing music sheets, for each image:  
    - Extract sheet music
    - Represent sheet music in the [Guido Music Notation](https://guidodoc.grame.fr/)

### Original Image

![](doc/images/original.png)

### Thresholding 

![](doc/images/threshold.png)

### Brace Removal

We get the bounding boxes of the image and check the first bounding box with a low aspect ratio and a very high height compared to the image height, and then remove it

### Group Extraction
A horizontal projection of the whole image is calculated and then thresholded and then we extract minimas between those peaks and cut the image to groups on them

![](doc/images/groups_projection.png)

![](doc/images/groups.png)

### Handling each group
We take each group separately and handle it:
### Augmentation Dots Detection
We can retrieve the augmentation dots by looking for bounding boxes with ~1 aspect ratio and small area, and then we can extract them from the image and remove them at this point in the pipeline
![](doc/images/dots.png)

### Clef removal
We remove the clef by taking the first peak in the group with a reasonable aspect ratio and then remove everything from the start of the image to that point

### Removing Staff Lines
We can detect the staff lines by retrieving the horizontal projection and then getting the peaks, and subtracting them from the original image to get rid of the staff lines

* To avoid cutting notes, we can check if a point belongs to a note by inspecting its neighbourhood and if it does, we do not cut it from the image.
* We can retrieve the dimensions of the staff lines now as we will need it below
  
![](doc/images/group_projection.png)

### Sanitizing The Image
Right after removing the staff lines, we can get the original image without any lines and also we can apply vertical closing to also prevent any unhandled cuts in the notes, after that we can retrieve the bounding boxes from the closed image.
![](doc/images/sanitized_closed.png)

### Handling Beamed Notes
We can segment according to the bounding boxes generated above, but we will have a problem that we need to detect which notes are beamed so that we can divide them into separate notes to prepare them for classification, so we can detect the note heads by doing a closing morphological step with an elliptical structring element with specific dimensions calculated from the staff dimensions.
![](doc/images/beam_slicing.png)

![](doc/images/beam_slicing_2.png)

### Final Segmentation Result

![](doc/images/final_segmentation.png)

Our boxes are now ready for segmentation classification

### Classification

As the image can contain various components like:
1.  Meters
2.  Augmentation Dots
3.  Various Accidentals
4.  Various Flagged Notes
5.  Various Beamed Notes
6.  Various Hollow Notes
7.  Chords

We need to construct a complex classifiers heirarchy to be able to handle all symbols.

![](doc/images/classifier_heirarchy.png)

### Identifying the note tone
After classifying which symbols are which, and knowing their timings, we also need to identify their tones.
This can be acheived by retrieving the staff lines from the image and also retrieving the note head by closing, and then checking the note head position with respect to the staff lines

![](doc/images/head_line.png)

### Tying Everything Together

After separating all symbols to separate components, we can now sort all components based on their X coordinates and tying all accidentals/dots to their closest note and we can then generate the final guido format now.

```
{
    [\meter<"4/4"> d1/4 e1/32 e2/2 e1/8 e1/16 e1/32 {e1/4,g1/4} e1/4 d1/8 c1/8 g1/32 c1/16 e1/32],

    [\meter<"4/4"> {b1/4,e1/4,g1/4} a1/8 d1/8 c1/16 g1/16 d1/16 e1/16 c2/16 g2/16 d2/16 e2/16 {b1/4,f1/4,g1/4} c1/4 a1/4. a1/8 a1/32..],

    [\meter<"4/4"> e1/16 e1/16 e1/16 e1/16 e1/4 e#1/4 g1/4 g&&1/4 g1/4 e#2/4]
}
```