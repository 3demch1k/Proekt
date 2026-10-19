"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    n = matrices[0].shape[0]
    total_sum = np.zeros((n, 1), dtype=float)
    
    for A, x in zip(matrices, vectors):
        total_sum += A @ x
        
    return total_sum


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    result = np.zeros_like(matrix)
    result[matrix > threshold] = 1
    return result


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    result = []
    for row in matrix:
        uniques = np.unique(row).tolist()
        result.append(uniques)
    return result


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    result = []
    for col_idx in range(matrix.shape[1]):
        col = matrix[:, col_idx]
        uniques = np.unique(col).tolist()
        result.append(uniques)
    return result


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    
    np.random.seed(seed)
    
    matrix = np.random.normal(loc=mean, scale=std, size=(rows, columns))
    
    row_means = np.mean(matrix, axis=1)
    col_means = np.mean(matrix, axis=0)
    row_vars = np.var(matrix, axis=1)
    col_vars = np.var(matrix, axis=0)
    
    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=col_means,
        row_variances=row_vars,
        column_variances=col_vars,
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    matrix = np.zeros((rows, columns), dtype=type(first))
    
    for i in range(rows):
        for j in range(columns):
            if (i + j) % 2 == 0:
                matrix[i, j] = first
            else:
                matrix[i, j] = second
                
    return matrix


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    
    y_center = image_height // 2
    x_center = image_width // 2
    
    y_start = y_center - height // 2
    y_end = y_start + height
    x_start = x_center - width // 2
    x_end = x_start + width
    
    image[y_start:y_end, x_start:x_end] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    
    y0 = image_height // 2
    x0 = image_width // 2
    
    y_coords, x_coords = np.ogrid[:image_height, :image_width]
    
    ellipse_mask = ((x_coords - x0) ** 2 / (semi_axis_x ** 2) + 
                    (y_coords - y0) ** 2 / (semi_axis_y ** 2)) <= 1.0
    
    image[ellipse_mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    
    mean_val = float(np.mean(values))
    var_val = float(np.var(values))
    std_val = float(np.std(values))
    
    maxima_indices = []
    minima_indices = []
    
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            maxima_indices.append(i)
        elif values[i] < values[i - 1] and values[i] < values[i + 1]:
            minima_indices.append(i)
            
    moving_avg = []
    for i in range(len(values) - window + 1):
        window_slice = values[i : i + window]
        moving_avg.append(float(np.mean(window_slice)))
        
    return TimeSeriesStatistics(
        mean=mean_val,
        variance=var_val,
        std=std_val,
        local_maxima=np.array(maxima_indices),
        local_minima=np.array(minima_indices),
        moving_average=np.array(moving_avg),
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    
    if class_count is None:
        class_count = int(np.max(labels)) + 1
        
    encoded = np.eye(class_count, dtype=int)[labels]
    return encoded