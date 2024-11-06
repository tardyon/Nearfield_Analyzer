import logging
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Manager, Pool, cpu_count
import sys
import os
import numpy as np
import pandas as pd
import datetime
import tkinter as tk
from tkinter import filedialog
from tifffile import TiffFile
from skimage import exposure, filters, measure, feature, transform
from skimage.draw import ellipse
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.stats import skew, kurtosis
import cv2
import pywt  # For wavelet transforms
from tqdm import tqdm
import functools
import contextlib
from functools import partial
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from skimage.transform import AffineTransform, warp  # Existing import
import math  # Added import for trigonometric functions

def setup_logging(log_queue, log_filepath):
    """Improved logging setup with both file and console output"""
    # Clear any existing handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler via queue
    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)
    
    listener = QueueListener(log_queue, file_handler)
    return listener, logger

@contextlib.contextmanager
def managed_pool(processes=None):
    """Context manager for pool resources"""
    pool = Pool(processes=processes)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()

def process_grid_cell(args):
    """Add error handling to grid cell processing"""
    try:
        i, j, grid_size, grid_height, grid_width, ellipse_mask, image_norm = args
        y_start = i * grid_height
        y_end = min((i + 1) * grid_height, ellipse_mask.shape[0])
        x_start = j * grid_width
        x_end = min((j + 1) * grid_width, ellipse_mask.shape[1])
        
        grid_mask = ellipse_mask[y_start:y_end, x_start:x_end]
        grid_intensities = image_norm[y_start:y_end, x_start:x_end][grid_mask]
        
        return np.mean(grid_intensities) if grid_intensities.size > 0 else np.nan
    except Exception as e:
        logging.error(f"Error processing grid cell: {e}")
        return np.nan

def compute_intensity_grid(grid_size, grid_height, grid_width, ellipse_mask, image_norm):
    """Improved grid computation with proper resource management"""
    intensity_grid = np.full((grid_size, grid_size), np.nan)
    tasks = [(i, j, grid_size, grid_height, grid_width, ellipse_mask, image_norm) 
             for i in range(grid_size) for j in range(grid_size)]
    
    with managed_pool() as pool:
        results = list(tqdm(
            pool.imap(process_grid_cell, tasks),
            total=len(tasks),
            desc="Processing grid cells"
        ))
        
        for idx, result in enumerate(results):
            i, j = divmod(idx, grid_size)
            intensity_grid[i, j] = result
            
    return intensity_grid

def compute_local_std(image, mask, block_size=5):
    """Optimized local standard deviation computation"""
    from scipy.ndimage import uniform_filter
    
    # Use uniform filter for better performance
    mean = uniform_filter(image, size=block_size)
    mean_sq = uniform_filter(image**2, size=block_size)
    std = np.sqrt(mean_sq - mean**2)
    
    return std * mask

def wait_for_input():
    """Wait for user input to continue"""
    while True:
        response = input("\nPress ENTER to continue to next step (or 'q' to quit)...")
        if response.lower() == 'q':
            return False
        return True

# Add the custom radial_profile function
def radial_profile(data, center, theta, radius, num_points=1000):
    """
    Compute the intensity values along a radial line from the center at angle theta.
    
    Parameters:
        data (ndarray): 2D array of image data.
        center (tuple): (x, y) coordinates of the center.
        theta (float): angle in radians.
        radius (int): maximum radius to sample.
        num_points (int): number of points along the radial line.
        
    Returns:
        intensities (ndarray): interpolated intensity values along the radial line.
    """
    x0, y0 = center
    x1 = x0 + radius * np.cos(theta)
    y1 = y0 + radius * np.sin(theta)
    # Generate coordinates along the radial line
    x, y = np.linspace(x0, x1, num_points), np.linspace(y0, y1, num_points)
    # Use map_coordinates for interpolation
    intensities = ndi.map_coordinates(data, [y, x], order=1, mode='reflect')
    return intensities

# Modify the compute_radial_slices function to use the custom radial_profile
def compute_radial_slices(image, center, radius, num_slices):
    """Compute intensity along radial slices from the center."""
    radial_data = {}
    angles = np.linspace(0, 2 * np.pi, num_slices, endpoint=False)
    
    for idx, angle in enumerate(angles):
        intensity_values = radial_profile(image, center, angle, radius)
        radial_data[f"Slice_{idx+1}"] = intensity_values.tolist()
    
    return radial_data

def main():
    """Modified main function with step-by-step execution"""
    manager = Manager()
    log_queue = manager.Queue()
    root = None
    
    try:
        # Initialize Tkinter
        root = tk.Tk()
        root.withdraw()
        
        # Setup initial logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        initial_log_filepath = f"analysis_{timestamp}.log"
        initial_listener, logger = setup_logging(log_queue, initial_log_filepath)
        initial_listener.start()
        
        logger.info("Starting analysis...")
        
        # File selection with error handling
        file_path = filedialog.askopenfilename(
            title="Select TIFF Image",
            filetypes=[("TIFF files", "*.tif *.tiff")]
        )
        
        if not file_path:
            logger.error("No file selected")
            return
            
        logger.info(f"Processing file: {file_path}")
        
        # Save metadata and set up output logging
        metadata_output_folder = os.path.join(os.path.dirname(file_path), 'output')
        os.makedirs(metadata_output_folder, exist_ok=True)
        metadata_filename = os.path.splitext(os.path.basename(file_path))[0] + '_metadata.txt'
        metadata_filepath = os.path.join(metadata_output_folder, metadata_filename)
        output_log_filepath = os.path.join(metadata_output_folder, 'analysis.log')

        # Setup output logging
        output_listener, _ = setup_logging(log_queue, output_log_filepath)
        output_listener.start()
        
        logger = logging.getLogger()
        logger.info("Analysis started.")

        # Load TIFF image and metadata
        with TiffFile(file_path) as tif:
            image = tif.asarray()
            metadata = tif.imagej_metadata

        # Save metadata to a text file
        with open(metadata_filepath, 'w') as f:
            if metadata:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write("No metadata found.")
        logger.info("Metadata saved.")

        # Normalize image between -1 and 1 (32-bit float)
        image = image.astype(np.float32)
        min_val = np.min(image)
        max_val = np.max(image)
        image_norm = 2 * (image - min_val) / (max_val - min_val + 1e-8) - 1
        logger.info("Image normalized.")

        # Perform measurements
        results = {}

        # 1. Ellipticity, Aspect Ratio, Eccentricity
        # Convert image to binary using thresholding
        threshold = filters.threshold_otsu(image_norm)
        binary_image = image_norm > threshold
        logger.info(f"Otsu threshold applied: {threshold}")

        # Label connected regions
        label_image = measure.label(binary_image)
        # Assuming the largest region is the beam
        regions = measure.regionprops(label_image, intensity_image=image_norm)
        if len(regions) == 0:
            logger.error("No regions found in the image.")
            print("No regions found in the image. Exiting.")
            sys.exit()

        # Find the largest region
        regions.sort(key=lambda x: x.area, reverse=True)
        beam_region = regions[0]
        logger.info("Largest region identified.")

        # Ellipse parameters
        major_axis_length = beam_region.major_axis_length
        minor_axis_length = beam_region.minor_axis_length

        ellipticity = (major_axis_length - minor_axis_length) / ((major_axis_length + minor_axis_length) / 2)
        aspect_ratio = minor_axis_length / major_axis_length
        eccentricity = beam_region.eccentricity

        results['Ellipticity'] = ellipticity
        logger.info(f"Ellipticity: {ellipticity}")
        results['Aspect Ratio'] = aspect_ratio
        logger.info(f"Aspect Ratio: {aspect_ratio}")
        results['Eccentricity'] = eccentricity
        logger.info(f"Eccentricity: {eccentricity}")

        # Extract the elliptical area
        # Create a mask for the elliptical area
        cy, cx = beam_region.centroid
        orientation = beam_region.orientation
        yy, xx = np.mgrid[:image_norm.shape[0], :image_norm.shape[1]]
        ellipse_mask = (((xx - cx) * np.cos(orientation) + (yy - cy) * np.sin(orientation)) ** 2) / (major_axis_length / 2 + 1e-8) ** 2 + \
                       (((xx - cx) * np.sin(orientation) - (yy - cy) * np.cos(orientation)) ** 2) / (minor_axis_length / 2 + 1e-8) ** 2 <= 1
        logger.info("Elliptical mask created.")

        intensities = image_norm[ellipse_mask]
        logger.info(f"Extracted {intensities.size} intensity values within the ellipse.")

        # 2. RMS Intensity
        rms_intensity = np.sqrt(np.mean(intensities**2))
        results['RMS Intensity'] = rms_intensity
        logger.info(f"RMS Intensity: {rms_intensity}")

        # 3. Mean Intensity
        mean_intensity = np.mean(intensities)
        results['Mean Intensity'] = mean_intensity
        logger.info(f"Mean Intensity: {mean_intensity}")

        # 4. Variance and Standard Deviation
        variance_intensity = np.var(intensities)
        std_intensity = np.std(intensities)
        results['Intensity Variance'] = variance_intensity
        logger.info(f"Intensity Variance: {variance_intensity}")
        results['Intensity Standard Deviation'] = std_intensity
        logger.info(f"Intensity Standard Deviation: {std_intensity}")

        # 5. Peak-to-Min Range
        peak_to_min = np.max(intensities) - np.min(intensities)
        results['Peak-to-Min Range'] = peak_to_min
        logger.info(f"Peak-to-Min Range: {peak_to_min}")

        # 6. Uniformity Index
        uniformity_index = 1 - (variance_intensity / (mean_intensity**2 + 1e-8))
        results['Uniformity Index'] = uniformity_index
        logger.info(f"Uniformity Index: {uniformity_index}")

        # 7. Skewness and Kurtosis
        skewness_intensity = skew(intensities)
        kurtosis_intensity = kurtosis(intensities)
        results['Intensity Skewness'] = skewness_intensity
        logger.info(f"Intensity Skewness: {skewness_intensity}")
        results['Intensity Kurtosis'] = kurtosis_intensity
        logger.info(f"Intensity Kurtosis: {kurtosis_intensity}")

        # 8. Coefficient of Variation (CV)
        cv_intensity = std_intensity / (mean_intensity + 1e-8)
        results['Coefficient of Variation'] = cv_intensity
        logger.info(f"Coefficient of Variation: {cv_intensity}")

        # 9. Entropy
        hist, bin_edges = np.histogram(intensities, bins=256, density=True)
        probabilities = hist * np.diff(bin_edges)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))
        results['Entropy'] = entropy
        logger.info(f"Entropy: {entropy}")

        # 10. Signal-to-Noise Ratio (SNR)
        # Assuming background is the area outside the beam
        background_mask = np.logical_not(ellipse_mask)
        background_intensities = image_norm[background_mask]
        noise_std = np.std(background_intensities)
        snr = mean_intensity / (noise_std + 1e-8)
        results['Signal-to-Noise Ratio'] = snr
        logger.info(f"Signal-to-Noise Ratio: {snr}")

        # 11. Contrast Ratio
        contrast_ratio = np.max(intensities) / (np.min(intensities) + 1e-8)
        results['Contrast Ratio'] = contrast_ratio
        logger.info(f"Contrast Ratio: {contrast_ratio}")

        # 12. Local Gradient Analysis
        # Compute gradients
        gy, gx = np.gradient(image_norm)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        # Mean gradient within the ellipse
        mean_gradient = np.mean(gradient_magnitude[ellipse_mask])
        results['Mean Gradient Magnitude'] = mean_gradient
        logger.info(f"Mean Gradient Magnitude: {mean_gradient}")

        # 13. Texture Metrics (GLCM)
        from skimage.feature import graycomatrix, graycoprops

        # Rescale intensities to 8-bit for GLCM
        intensities_8bit = exposure.rescale_intensity(intensities, out_range=(0, 255)).astype(np.uint8)
        glcm = graycomatrix(intensities_8bit.reshape(-1,1), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0,0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0,0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
        ASM = graycoprops(glcm, 'ASM')[0,0]
        energy = graycoprops(glcm, 'energy')[0,0]
        correlation = graycoprops(glcm, 'correlation')[0,0]

        results['GLCM Contrast'] = contrast
        logger.info(f"GLCM Contrast: {contrast}")
        results['GLCM Dissimilarity'] = dissimilarity
        logger.info(f"GLCM Dissimilarity: {dissimilarity}")
        results['GLCM Homogeneity'] = homogeneity
        logger.info(f"GLCM Homogeneity: {homogeneity}")
        results['GLCM ASM'] = ASM
        logger.info(f"GLCM ASM: {ASM}")
        results['GLCM Energy'] = energy
        logger.info(f"GLCM Energy: {energy}")
        results['GLCM Correlation'] = correlation
        logger.info(f"GLCM Correlation: {correlation}")

        # 14. Spatial Frequency Analysis (Fourier Transform)
        fft_image = np.fft.fft2(image_norm * ellipse_mask)
        fft_shifted = np.fft.fftshift(fft_image)
        magnitude_spectrum = np.abs(fft_shifted)
        # Compute mean of magnitude spectrum to represent high-frequency content
        mean_frequency_content = np.mean(magnitude_spectrum)
        results['Mean Frequency Content'] = mean_frequency_content
        logger.info(f"Mean Frequency Content: {mean_frequency_content}")

        # 15. Power Spectral Density (PSD) Analysis
        psd2D = np.abs(fft_shifted)**2
        psd_mean = np.mean(psd2D)
        results['Mean PSD'] = psd_mean
        logger.info(f"Mean PSD: {psd_mean}")

        # 16. Standard Deviation Map
        from scipy.ndimage import generic_filter

        logger.info("Computing local standard deviation map...")
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                compute_local_std,
                image_norm,
                ellipse_mask
            )
            try:
                std_map = future.result(timeout=30)  # 30 second timeout
                mean_std_within_ellipse = np.mean(std_map[ellipse_mask])
                results['Mean Local Standard Deviation'] = mean_std_within_ellipse
                logger.info(f"Mean Local Standard Deviation: {mean_std_within_ellipse}")
            except TimeoutError:
                logger.warning("Local standard deviation calculation timed out, skipping...")
                results['Mean Local Standard Deviation'] = np.nan

        # Start step-by-step execution from the point where it hangs
        logger.info("=== Starting step-by-step execution ===")

        # Step 17: Autocorrelation Function (SKIP FOR NOW)
        print("\nSkipping Autocorrelation (memory intensive)...")
        logger.info("Skipping Autocorrelation calculation")
        results['Autocorrelation Peak'] = np.nan

        # Step 18: Wavelet Transform Analysis
        print("\nPreparing to compute Wavelet Transform...")
        if not wait_for_input():
            return
        logger.info("Computing Wavelet Transform...")
        try:
            # Reduce memory usage by working with a smaller portion of the image
            masked_image = image_norm * ellipse_mask
            max_size = 1024  # Maximum size to process
            if (masked_image.shape[0] > max_size or masked_image.shape[1] > max_size):
                scale = max_size / max(masked_image.shape)
                from skimage.transform import resize
                masked_image = resize(masked_image, 
                                   (int(masked_image.shape[0] * scale), 
                                    int(masked_image.shape[1] * scale)))

            coeffs2 = pywt.dwt2(masked_image, 'haar')
            cA, (cH, cV, cD) = coeffs2
            wavelet_energy = np.sum(cH**2 + cV**2 + cD**2)
            results['Wavelet Energy'] = wavelet_energy
            logger.info(f"Wavelet Energy: {wavelet_energy}")
            
            # Clean up
            del masked_image, coeffs2, cA, cH, cV, cD
        except Exception as e:
            logger.error(f"Error in Wavelet Transform: {e}")
            results['Wavelet Energy'] = np.nan

        # Step 19: Zernike Polynomial Analysis (Removed)
        # print("\nPreparing to compute Zernike Polynomial Analysis...")
        # if not wait_for_input():
        #     return
        # logger.info("Starting Zernike Polynomial Analysis...")
        # try:
        #     # Create an affine transformation to map ellipse to circle
        #     scale_transform = AffineTransform(scale=(1, minor_axis_length / (major_axis_length + 1e-8)))
        #     shift_transform = AffineTransform(translation=(-cx, -cy))
        #     rotate_transform = AffineTransform(rotation=-orientation)
        #     transform = shift_transform + rotate_transform + scale_transform + rotate_transform.inverse + shift_transform.inverse
        #
        #     # Apply the transformation with error handling
        #     try:
        #         warped_image = warp(image_norm * ellipse_mask, transform.inverse, output_shape=image_norm.shape)
        #         logger.info("Affine transformation applied to map ellipse to circle.")
        #     except Exception as e:
        #         logger.error(f"Error in warping: {e}")
        #         results['Zernike Moment Sum'] = np.nan
        #         raise
        #
        #     # Compute Zernike moments
        #     n = 4  # Degree of Zernike polynomials
        #     zernike_moment_array = compute_zernike_moments(warped_image, degree=n)
        #     if zernike_moment_array.size > 0:
        #         zernike_moment_sum = np.sum(np.abs(zernike_moment_array))
        #         results['Zernike Moment Sum'] = zernike_moment_sum
        #         logger.info(f"Zernike Moment Sum: {zernike_moment_sum}")
        #     else:
        #         results['Zernike Moment Sum'] = np.nan
        #         logger.warning("Zernike moments could not be computed.")
        #         
        # except Exception as e:
        #     logger.error(f"Error in Zernike analysis: {e}")
        #     results['Zernike Moment Sum'] = np.nan
        #
        # # Clean up large arrays
        # del warped_image

        # Step 20: Beam Quality Factor
        print("\nPreparing to compute Beam Quality Factor...")
        if not wait_for_input():
            return
        logger.info("Computing Beam Quality Factor...")
        x = np.arange(image_norm.shape[1])
        y = np.arange(image_norm.shape[0])
        X, Y = np.meshgrid(x, y)
        X = X[ellipse_mask]
        Y = Y[ellipse_mask]
        I = intensities

        x_mean = np.sum(X * I) / np.sum(I)
        y_mean = np.sum(Y * I) / np.sum(I)

        x2_mean = np.sum((X - x_mean)**2 * I) / np.sum(I)
        y2_mean = np.sum((Y - y_mean)**2 * I) / np.sum(I)

        beam_radius = np.sqrt(x2_mean + y2_mean)
        results['Beam Radius (Second Moment)'] = beam_radius
        logger.info(f"Beam Radius (Second Moment): {beam_radius}")

        # Continue with remaining steps in the same pattern
        # Add wait_for_input() before each major computation step

        # 21. Smoothness Index
        print("\nPreparing to compute Smoothness Index...")
        if not wait_for_input():
            return
        # Compute gradients within the ellipse
        gradient_magnitude_within_ellipse = gradient_magnitude[ellipse_mask]
        smoothness_index = 1 / (np.sum(gradient_magnitude_within_ellipse**2) + 1e-8)
        results['Smoothness Index'] = smoothness_index
        logger.info(f"Smoothness Index: {smoothness_index}")

        # 22. Edge Slope Analysis
        print("\nPreparing to compute Edge Slope Analysis...")
        if not wait_for_input():
            return
        from skimage.filters import sobel

        edge_sobel = sobel(binary_image.astype(float))
        edge_profile = gradient_magnitude[edge_sobel > 0]
        edge_slope = np.mean(edge_profile)
        results['Edge Slope'] = edge_slope
        logger.info(f"Edge Slope: {edge_slope}")

        # 23. Edge Uniformity Index
        print("\nPreparing to compute Edge Uniformity Index...")
        if not wait_for_input():
            return
        edge_variance = np.var(edge_profile)
        results['Edge Variance'] = edge_variance
        logger.info(f"Edge Variance: {edge_variance}")

        # 24. Histogram Analysis
        print("\nPreparing to compute Histogram Analysis...")
        if not wait_for_input():
            return
        hist_counts, _ = np.histogram(intensities, bins=256)
        hist_peak = np.max(hist_counts)
        results['Histogram Peak'] = hist_peak
        logger.info(f"Histogram Peak: {hist_peak}")

        # 25. Line Profiles
        print("\nPreparing to compute Line Profiles...")
        if not wait_for_input():
            return
        center_x = int(cx)
        center_y = int(cy)
        horizontal_profile = image_norm[center_y, :]
        vertical_profile = image_norm[:, center_x]
        # Compute deviations from mean
        horizontal_deviation = np.std(horizontal_profile - mean_intensity)
        vertical_deviation = np.std(vertical_profile - mean_intensity)
        results['Horizontal Profile Deviation'] = horizontal_deviation
        logger.info(f"Horizontal Profile Deviation: {horizontal_deviation}")
        results['Vertical Profile Deviation'] = vertical_deviation
        logger.info(f"Vertical Profile Deviation: {vertical_deviation}")

        # 26. Local Peak and Trough Detection
        print("\nPreparing to compute Local Peak and Trough Detection...")
        if not wait_for_input():
            return
        from scipy.signal import find_peaks

        intensities_flat = intensities.flatten()
        peaks, _ = find_peaks(intensities_flat)
        troughs, _ = find_peaks(-intensities_flat)
        num_peaks = len(peaks)
        num_troughs = len(troughs)
        results['Number of Peaks'] = num_peaks
        logger.info(f"Number of Peaks: {num_peaks}")
        results['Number of Troughs'] = num_troughs
        logger.info(f"Number of Troughs: {num_troughs}")

        # 27. Heatmap of Regional Intensity Averages
        print("\nPreparing to compute Heatmap of Regional Intensity Averages...")
        if not wait_for_input():
            return
        # Divide ellipse into grid
        grid_size = 10
        height, width = image_norm.shape
        grid_height = height // grid_size
        grid_width = width // grid_size

        logger.info("Starting Heatmap of Regional Intensity Averages.")
        intensity_grid = compute_intensity_grid(grid_size, grid_height, grid_width, ellipse_mask, image_norm)
        logger.info("Intensity grid computed.")

        # Generate and save heatmap
        heatmap_filepath = os.path.join(metadata_output_folder, f"heatmap_{timestamp}.png")
        plt.figure(figsize=(8, 6))
        plt.imshow(intensity_grid, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Mean Intensity')
        plt.title('Heatmap of Regional Intensity Averages')
        plt.savefig(heatmap_filepath)
        plt.close()
        logger.info(f"Heatmap saved to {heatmap_filepath}")

        # Compute variance across grid cells
        grid_variance = np.nanvar(intensity_grid)
        results['Grid Intensity Variance'] = grid_variance
        logger.info(f"Grid Intensity Variance: {grid_variance}")

        # --- New Changes Start Here ---
        
        # Configuration for finer grid and radial slices
        finer_grid_size = 20  # Increased grid size for finer resolution
        finer_grid_size = 20  # You can adjust this value as needed
        num_radial_slices = 360  # Number of radial slices up to 360 degrees
        max_radius = min(image_norm.shape[0], image_norm.shape[1]) // 2  # Maximum radius for slices

        # Update grid computation with finer grid
        print("\nPreparing to compute Finer Heatmap of Regional Intensity Averages...")
        if not wait_for_input():
            return
        logger.info("Starting finer intensity grid computation.")
        intensity_grid = compute_intensity_grid(finer_grid_size, grid_height, grid_width, ellipse_mask, image_norm)
        logger.info("Finer intensity grid computed.")

        # Generate and save finer heatmap
        finer_heatmap_filepath = os.path.join(metadata_output_folder, f"finer_heatmap_{timestamp}.png")
        plt.figure(figsize=(10, 8))
        plt.imshow(intensity_grid, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Mean Intensity')
        plt.title('Finer Heatmap of Regional Intensity Averages')
        plt.savefig(finer_heatmap_filepath)
        plt.close()
        logger.info(f"Finer Heatmap saved to {finer_heatmap_filepath}")

        # --- Radial Cross-Section Slices ---
        print("\nPreparing to compute Radial Cross-Section Slices...")
        if not wait_for_input():
            return
        logger.info("Computing radial cross-section slices.")
        
        # Define center and radius
        center = (cx, cy)
        radius = max_radius

        # Compute radial slices
        radial_slices = compute_radial_slices(image_norm, center, radius, num_radial_slices)
        
        # Save radial slices to CSV
        radial_csv_filepath = os.path.join(metadata_output_folder, f"radial_slices_{timestamp}.csv")
        df_radial = pd.DataFrame(radial_slices)
        df_radial.to_csv(radial_csv_filepath, index=False)
        logger.info(f"Radial cross-section slices saved to {radial_csv_filepath}")
        
        # --- Save Cross-Sections Metadata ---
        # Ensure metadata is not empty
        if not metadata:
            logger.warning("No metadata found in TIFF. Attempting to extract basic metadata.")
            metadata = {
                "Image Shape": image.shape,
                "Normalized Range": f"{min_val} to {max_val}",
                "Grid Size": finer_grid_size,
                "Radial Slices": num_radial_slices
            }
            with open(metadata_filepath, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            logger.info("Basic metadata extracted and saved.")
        else:
            # Append additional metadata
            with open(metadata_filepath, 'a') as f:
                f.write(f"Grid Size: {finer_grid_size}\n")
                f.write(f"Number of Radial Slices: {num_radial_slices}\n")
            logger.info("Additional metadata appended.")

        # Continue with remaining steps...

        # 28. Region-to-Region Intensity Difference
        print("\nPreparing to compute Region-to-Region Intensity Difference...")
        if not wait_for_input():
            return
        differences = []
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                if not np.isnan(intensity_grid[i, j]) and not np.isnan(intensity_grid[i+1, j]):
                    differences.append(abs(intensity_grid[i, j] - intensity_grid[i+1, j]))
                if not np.isnan(intensity_grid[i, j]) and not np.isnan(intensity_grid[i, j+1]):
                    differences.append(abs(intensity_grid[i, j] - intensity_grid[i, j+1]))
        if differences:
            mean_region_difference = np.mean(differences)
            results['Mean Region-to-Region Difference'] = mean_region_difference
            logger.info(f"Mean Region-to-Region Difference: {mean_region_difference}")
        else:
            results['Mean Region-to-Region Difference'] = np.nan
            logger.warning("No valid differences found between regions.")

        # Final steps
        print("\nPreparing to save results...")
        if not wait_for_input():
            return
        logger.info("Saving results...")
        # Save results to CSV
        csv_filename = f"results_{timestamp}.csv"
        csv_filepath = os.path.join(metadata_output_folder, csv_filename)

        df_results = pd.DataFrame([results])
        df_results.to_csv(csv_filepath, index=False)
        logger.info(f"Results saved to {csv_filepath}")
        logger.info(f"Metadata saved to {metadata_filepath}")
        logger.info("Analysis completed.")

        print(f"Analysis complete. Results saved to {csv_filepath}")
        print(f"Metadata saved to {metadata_filepath}")

    except Exception as e:
        logger.exception("An error occurred during analysis.")
        print(f"An error occurred: {e}")
        return 1
    finally:
        # Stop both listeners if they were started
        if 'initial_listener' in locals() and isinstance(initial_listener, QueueListener):
            initial_listener.stop()
        if 'output_listener' in locals() and isinstance(output_listener, QueueListener):
            output_listener.stop()
        if root:
            root.destroy()
        manager.shutdown()
    
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
