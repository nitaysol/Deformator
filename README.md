# Deformator
Deformating image using nearest/bilinear/cubic interpolations

## How it works:
1. Loads and displays an image I.
2. Allows the user to select the radius of influence, Re.
3. Provides a mechanism to select an initial point, Pi , and the application
shows the circle of influence around the selected point. 
4. When the user selects a target point, the program deform and display the 
resulting image. The deformation is computed from the source point to the target point .
5. Supports image quality (interpolation) at three levels: nearest neighbor, 
bilinear interpolation, or cubic

## Assumption:
* The influence circles of the two selected points intersect.
* Used OpenCV for the interpolations