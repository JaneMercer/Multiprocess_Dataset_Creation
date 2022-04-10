# Dataset_Creation
Script for creating .pkl files with Images and Labels for NNs by folder structure.

If you want to create 2 pickle files: one with images and the second one with coresponding labels to those images, you can use this simple script.

You basicly:
1) Create directories with the names of your labels anywhere on your computer
2) Put your dataset images to coresponding labed folders
3) Copy and open the file "create_set_from_folder.py" (you probably need to instal some dependencies)
   there are "DATADIR" and "DIRECTORIES" vars:
4) Put the path to the folder with labeled folders to "DATADIR"  
5) Put the labeled folders names into the list of "DIRECTORIES"
6) "IMG_SIZE" - is the size you want to resize your images to, it will be a square. Or just put the size that they already have.
7) "METHODS" - it is a list of functions wich can be applied to the images of your dataset. You can add yours. 
    You choose the function you want to use by "I_NDEX" - index of the element in "METHODS" list
9) Run the script. It will create "F_features.pickle" and "L_lables.pickle" in a project folder.
