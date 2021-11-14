# Image-scaling-and-rotation
Scales an image of any size to a specified size and rotates it by a specified angle. The image interpolation algorithm supports bilinear interpolation and nearest neighbor interpolation. There is an option to prevent the rotated image from sticking out.

## Functions
```py
resize_img(fp: str, scale, interpolator="bilinear", rotation=0, rotate_fit: bool = True) -> ndarray
```
Return a resized image by interporation.

### Parameters
  - fp : A file name (string).  
  - scale : float or tuple of float.  
      - A scale factor of image enlarging/reducing.  
  - interpolator : {"bilinear", "nearest"}, optional, default: "bilinear".  
      - An algorithm for interpolating images.  
  - rotation : float.  
      - How much of an angle to rotate the image.  
  - rotate_fit: bool.  
      - Whether or not to rotate the image without sticking out.  
    
---

```py
save(obj, fp: str) -> None
```
Saves array-like object as an image.

### Parameters
  - obj: Array-like object.
  - fp : A file name (string).


## Requirement
  - **NumPy**  
    - For matrix operations.
  - **Pillow**
    - For loading and saving images
