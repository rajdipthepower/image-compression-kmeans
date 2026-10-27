# Image Compression Using K-Means Clustering

## Overview

This project demonstrates image compression through **color quantization using K-Means clustering**.

The main idea is to reduce the number of distinct colors in an image while preserving as much visual quality as possible. Each RGB pixel is treated as a data point, and K-Means clustering groups similar colors together.

The resulting cluster centroids represent the reduced color palette, and each pixel is reconstructed using the color of its assigned centroid.

The K-Means algorithm is implemented **from scratch using NumPy** rather than using a machine-learning library implementation.

---

## Web Application Showcase (Streamlit)

In addition to the script-based workflow, this project now includes an interactive **Streamlit Web Application** to showcase the compression live! 

### Features of the Web App:
- **Interactive K-Means Compressor:** Upload any custom image (JPG/PNG) or use the default sample image (`bird.png`), select your preferred cluster size ($K$), and watch the compression happen live.
- **File Size Comparison & Download:** Instantly see the exact file size reduction percentage and download your compressed image directly to your device.
- **Exploratory Experiment Dashboard:** Explore pre-computed benchmarks, experiment result tables, and evaluation plots across different cluster sizes.
- **Night Mode Toggle:** Includes a seamless dark/light theme toggle switch for comfortable viewing.

### Running the Web App Locally:
1. Navigate to the deployment folder:
   ```bash
   cd deployment
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## How It Works

The image compression pipeline consists of the following steps:

1. Load the input image using Pillow.
2. Convert the image to RGB format.
3. Normalize the RGB pixel values.
4. Reshape the image into a collection of RGB pixel vectors.
5. Initialize `K` cluster centroids.
6. Assign each pixel to its nearest centroid.
7. Recalculate the cluster centroids.
8. Repeat the assignment and centroid update steps until convergence or the maximum number of iterations is reached.
9. Replace every pixel with its corresponding centroid color.
10. Reshape the pixels back into the original image dimensions.
11. Save the reconstructed image.

A smaller value of `K` produces stronger color reduction, while a larger value of `K` preserves more color information.

---

## Project Structure

```text
image-compression-kmeans/
│
├── data/
│   ├── input/
│   │   └── bird.png
│   │
│   └── output/
│       ├── reconstructed/
│       ├── plots/
│       ├── experiment_results.csv
│       └── final_images.png
│
├── deployment/
│   └── app.py (Streamlit Web App)
│
├── src/
│   ├── kmeans.py
│   ├── image_utils.py
│   ├── metrics.py
│   └── visualization.py
│
├── notebooks/
│   └── kmeans.ipynb
│
├── main.py
├── test.py
├── requirements.txt
├── .gitignore
└── README.md
```

### File Descriptions

- `deployment/app.py`
  The interactive Streamlit web application for live compression and benchmark visualization.

- `src/kmeans.py`
  Contains the K-Means clustering implementation.

- `src/image_utils.py`
  Handles image loading, reshaping, compression, reconstruction, and image saving.

- `src/metrics.py`
  Contains the evaluation metrics used to measure reconstruction quality and compression.

- `src/visualization.py`
  Generates plots and visual comparisons of the reconstructed images.

- `main.py`
  Runs the main image compression workflow.

- `test.py`
  Tests the connection between the different project modules and verifies that the workflow runs correctly.

- `notebooks/kmeans.ipynb`
  Original Jupyter Notebook implementation and experimentation.

- `data/input/`
  Contains the original input image.

- `data/output/`
  Contains reconstructed images, plots, and experiment results.

---

## Evaluation

The project evaluates the effect of different numbers of clusters using:

```text
K = 2, 4, 6, 8, 16, 32, 64
```

Multiple random seeds were used for each value of `K` to observe the effect of centroid initialization.

The following metrics were measured:

### Mean Squared Error (MSE)

MSE measures the average squared difference between the original and reconstructed image.

Lower MSE indicates lower reconstruction error.

### PSNR

Peak Signal-to-Noise Ratio measures the quality of the reconstructed image.

Higher PSNR generally indicates better reconstruction quality.

### Inertia

K-Means inertia is the sum of the squared distances between each pixel and its assigned cluster centroid.

Lower inertia indicates that pixels are, on average, closer to their assigned centroids.

### Compression Ratio

The project uses a theoretical representation-based compression ratio based on:

- the number of pixels,
- the number of cluster centroids,
- and the number of bits required to represent each cluster assignment.

This is used to study the relationship between the number of colors and the potential storage reduction.

### Runtime

The execution time required to perform K-Means clustering is measured for different values of `K`.

### Memory Usage

Peak memory allocation during the compression process is also measured.

---

## Experiments and Visualizations

The experiments generate the following plots:

- **Inertia vs Number of Clusters**
- **MSE vs Number of Clusters**
- **PSNR vs Number of Clusters**
- **Compression Ratio vs Number of Clusters**
- **Runtime vs Number of Clusters**
- **Memory Usage vs Number of Clusters**

Reconstructed images are also generated for:

```text
K = 2, 4, 6, 8, 16, 32, 64
```

This allows visual comparison between the original image and the compressed/reconstructed versions.

---

## Results

The experiments demonstrate the trade-off between image quality, compression, and computational cost.

As the number of clusters increases:

- The reconstructed image generally preserves more colors.
- MSE decreases.
- PSNR increases.
- K-Means inertia decreases.
- The theoretical compression ratio decreases.
- Runtime increases.
- Memory usage increases.

Therefore, increasing `K` improves image quality but comes at the cost of computational resources and reduced compression.

The results can be used to select a suitable value of `K` depending on the desired balance between visual quality and compression.

---

## Example Reconstruction

The project generates reconstructed versions of the original image using different numbers of color clusters.

The output includes a comparison of:

```text
Original
K = 2
K = 4
K = 6
K = 8
K = 16
K = 32
K = 64
```

The generated visualization is stored in the `data/output/` directory.

---

## Requirements

The project requires Python and the following packages:

- NumPy
- Pandas
- Matplotlib
- Pillow
- Streamlit

Install them using:

```bash
pip install -r requirements.txt
```

The project does not require any external machine-learning library for K-Means because the algorithm is implemented from scratch.

---

## Running the Project

After installing the requirements, run:

```bash
python main.py
```

The test workflow can be run using:

```bash
python test.py
```

The visualization workflow can be run using:

```bash
python src/visualization.py
```

Generated reconstructed images, plots, and experiment results are stored in `data/output/`.

---

## Limitations

This project focuses on color quantization using K-Means rather than a complete production-grade image compression format.

Some limitations include:

- K-Means can become computationally expensive as the number of pixels and clusters increases.
- The result can depend on the random initialization of centroids.
- The compression ratio is a theoretical representation-based estimate rather than a direct comparison of PNG or JPEG file sizes.
- Spatial relationships between neighboring pixels are not explicitly considered.
- The current implementation is primarily demonstrated using a single input image.

---

## Future Improvements

Possible future improvements include:

- PCA-based dimensionality reduction.
- More efficient distance calculations.
- Improved centroid initialization.
- Testing on multiple images.
- Comparison with other image compression techniques.
- More detailed performance profiling.
- Investigation of alternative color spaces.
- Improved storage/encoding of the compressed representation.

---

## Learning Objective

The purpose of this project is to understand how **K-Means clustering can be applied to image color quantization**, while also exploring the relationship between:

**Number of Clusters → Image Quality → Compression → Runtime → Memory Usage**

The project also demonstrates how an exploratory Jupyter Notebook implementation can be converted into a modular Python project.
