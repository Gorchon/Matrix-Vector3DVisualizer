
# Matrix & Vector 3D Visualizer

This project is a small web application built with Python, Streamlit, NumPy, and Plotly.  
It allows you to load vectors and point sets in 3D space, apply matrix transformations, and see the results in real time. I coded this because I wanted to practice linear algebra for Deep Learning Research. 

---

## Features

- **3D Visualization**  
  Plot points and vectors in a 3D interactive view. Compare original and transformed data side by side.

- **Transformation Pipeline**  
  Build transformations step by step using rotations, scaling, shear, translation, or a custom 3×3 matrix.  
  The transformations are composed in the order you add them.

- **Basis Mapping**  
  Visualize how the standard basis vectors (x, y, z axes) are transformed by your matrix.

- **Matrix Tools**  
  - Inspect determinant, rank, and inverse (when it exists).  
  - Download the composed matrix as CSV.  
  - Quick matrix calculator to multiply two matrices or apply a matrix to a vector.

- **Custom Data**  
  Paste your own set of 3D points or vectors to test how they transform.

---

## Installation

Clone or download this repository, then set up a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
# venv\Scripts\activate    # On Windows (PowerShell)

pip install --upgrade pip
pip install streamlit numpy plotly
````

---

## Usage

Start the app with:

```bash
streamlit run matrix3d_app.py
```

Your browser will open a local page (usually at [http://localhost:8501](http://localhost:8501)).

---

## Example Workflows

* Rotate a cube around the Z axis and see how its shape changes.
* Apply a shear to the unit sphere and visualize the distortion.
* Paste your own vectors (For example: `1,0,0` and `0,1,0`) and compare them with their transformed versions.
* Multiply two 3×3 matrices in the calculator tab and quickly test the result on a vector.

---

## Notes

* Order of transforms matters: adding a rotation before a scale is not the same as scaling before rotating.
* Singular matrices (determinant = 0) will not have an inverse, and the app will indicate this.
* For large point sets (several thousand), performance may depend on your machine.

---


