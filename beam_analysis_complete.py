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

def setup_logging(log_queue, log_filepath):
    """Improved logging setup with both file and console output"""
    logger = logging.getLogger()
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

def compute_zernike_moments(warped_image, n):
    from skimage.measure import moments_zernike
    logger = logging.getLogger()
    moments = []
    for n_order in range(n+1):
        for m_order in range(-n_order, n_order+1, 2):
            moment = moments_zernike(warped_image, n_order, m_order)
            moments.append(moment)
    return moments

def main():
    """Improved main function with proper resource and error management"""
    manager = Manager()
    log_queue = manager.Queue()
    root = None
    
    try:
        # Initialize Tkinter
        root = tk.Tk()
        root.withdraw()
        
        # Setup logging first
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filepath = f"analysis_{timestamp}.log"
        listener, logger = setup_logging(log_queue, log_filepath)
        listener.start()
        
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
        
        # Save metadata and logging setup
        metadata_output_folder = os.path.join(os.path.dirname(file_path), 'output')
        os.makedirs(metadata_output_folder, exist_ok=True)
        metadata_filename = os.path.splitext(os.path.basename(file_path))[0] + '_metadata.txt'
        metadata_filepath = os.path.join(metadata_output_folder, metadata_filename)
        log_filepath = os.path.join(metadata_output_folder, 'analysis.log')

        # Initialize logging
        listener = setup_logging(log_queue, log_filepath)
        
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

        std_map = generic_filter(image_norm, np.std, size=5)
        mean_std_within_ellipse = np.mean(std_map[ellipse_mask])
        results['Mean Local Standard Deviation'] = mean_std_within_ellipse
        logger.info(f"Mean Local Standard Deviation: {mean_std_within_ellipse}")

        # 17. Autocorrelation Function
        from scipy.signal import correlate2d

        autocorr = correlate2d(image_norm * ellipse_mask, image_norm * ellipse_mask, mode='same')
        # Measure the central peak value
        autocorr_peak = autocorr[autocorr.shape[0]//2, autocorr.shape[1]//2]
        results['Autocorrelation Peak'] = autocorr_peak
        logger.info(f"Autocorrelation Peak: {autocorr_peak}")

        # 18. Wavelet Transform Analysis
        coeffs2 = pywt.dwt2(image_norm * ellipse_mask, 'haar')
        cA, (cH, cV, cD) = coeffs2
        # Use the energy of the detail coefficients as a metric
        wavelet_energy = np.sum(cH**2 + cV**2 + cD**2)
        results['Wavelet Energy'] = wavelet_energy
        logger.info(f"Wavelet Energy: {wavelet_energy}")

        # 19. Zernike Polynomial Analysis
        from skimage.transform import warp, AffineTransform

        logger.info("Starting Zernike Polynomial Analysis.")
        # Create an affine transformation to map ellipse to circle
        scale_transform = AffineTransform(scale=(1, minor_axis_length / (major_axis_length + 1e-8)))
        shift_transform = AffineTransform(translation=(-cx, -cy))
        rotate_transform = AffineTransform(rotation=-orientation)
        transform = shift_transform + rotate_transform + scale_transform + rotate_transform.inverse + shift_transform.inverse

        # Apply the transformation
        warped_image = warp(image_norm * ellipse_mask, transform.inverse, output_shape=image_norm.shape)
        logger.info("Affine transformation applied to map ellipse to circle.")

        # Generate polar coordinates
        y_indices, x_indices = np.indices(warped_image.shape)
        x_center = warped_image.shape[1] / 2
        y_center = warped_image.shape[0] / 2
        r = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2) / (min(warped_image.shape) / 2 + 1e-8)
        theta = np.arctan2(y_indices - y_center, x_indices - x_center)

        # Limit to unit circle
        inside_circle = r <= 1
        zernike_intensities = warped_image[inside_circle]

        # Compute Zernike moments (up to order n)
        n = 4  # Order of Zernike polynomials
        zernike_moments = compute_zernike_moments(warped_image, n)
        logger.info("Zernike moments computed.")

        # For simplicity, we'll just compute the sum of absolute values of moments
        zernike_moment_sum = np.sum(np.abs(zernike_moments))
        results['Zernike Moment Sum'] = zernike_moment_sum
        logger.info(f"Zernike Moment Sum: {zernike_moment_sum}")

        # 20. Beam Quality Factor (M^2)
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

        # 21. Smoothness Index
        # Compute gradients within the ellipse
        gradient_magnitude_within_ellipse = gradient_magnitude[ellipse_mask]
        smoothness_index = 1 / (np.sum(gradient_magnitude_within_ellipse**2) + 1e-8)
        results['Smoothness Index'] = smoothness_index
        logger.info(f"Smoothness Index: {smoothness_index}")

        # 22. Edge Slope Analysis
        from skimage.filters import sobel

        edge_sobel = sobel(binary_image.astype(float))
        edge_profile = gradient_magnitude[edge_sobel > 0]
        edge_slope = np.mean(edge_profile)
        results['Edge Slope'] = edge_slope
        logger.info(f"Edge Slope: {edge_slope}")

        # 23. Edge Uniformity Index
        edge_variance = np.var(edge_profile)
        results['Edge Variance'] = edge_variance
        logger.info(f"Edge Variance: {edge_variance}")

        # 24. Histogram Analysis
        hist_counts, _ = np.histogram(intensities, bins=256)
        hist_peak = np.max(hist_counts)
        results['Histogram Peak'] = hist_peak
        logger.info(f"Histogram Peak: {hist_peak}")

        # 25. Line Profiles
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
        # Divide ellipse into grid
        grid_size = 10
        height, width = image_norm.shape
        grid_height = height // grid_size
        grid_width = width // grid_size

        logger.info("Starting Heatmap of Regional Intensity Averages.")
        intensity_grid = compute_intensity_grid(grid_size, grid_height, grid_width, ellipse_mask, image_norm)
        logger.info("Intensity grid computed.")

        # Compute variance across grid cells
        grid_variance = np.nanvar(intensity_grid)
        results['Grid Intensity Variance'] = grid_variance
        logger.info(f"Grid Intensity Variance: {grid_variance}")

        # 28. Region-to-Region Intensity Difference
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

        # Save results to CSV
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
        logging.exception("An error occurred during analysis.")
        print(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        listener.stop()
        if root:
            root.destroy()
        manager.shutdown()

if __name__ == "__main__":
    sys.exit(main() or 0)
