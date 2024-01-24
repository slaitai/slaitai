import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

 


def get_video_info(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    duration_seconds = frame_count / frame_rate

    # Release the video capture object
    cap.release()

    return {
        "frame_count": frame_count,
        "frame_rate": frame_rate,
        "duration_seconds": duration_seconds
    }


def display_even_pieces(frame, new_height, pieces_per_row = 5, pieces_per_col = 5  ):
    print(frame.shape)
    print(new_height)
    frame = frame[new_height // 3:, :]
    print(frame.shape)
    rows, cols, _ = frame.shape


    piece_height = rows // pieces_per_row
    piece_width = cols // pieces_per_col

    patch_engery = {}
    patches_dict = {}

    for i in range(pieces_per_row):
        for j in range(pieces_per_col):
            y_start = i * piece_height
            y_end = (i + 1) * piece_height
            x_start = j * piece_width
            x_end = (j + 1) * piece_width

            piece = frame[y_start:y_end, x_start:x_end, :]
            patch_sum = np.sum(np.square(piece.astype(np.float32)))
            
            patch_x = i
            patch_y = j
            patches_dict[(patch_x, patch_y)] = piece
            patch_engery[(patch_x, patch_y)] = patch_sum

    return patches_dict, patch_engery


def get_patches_and_engery(video_path, pieces_per_row = 5, pieces_per_col = 5 ):
    # Read the video file
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_height = 250
    # Iterate through frames
    frame_num = 0
    frame_engery = []
    all_patches = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
    
        # Display even pieces for the current frame
        patches_dict, patch_engery = display_even_pieces(frame, new_height, pieces_per_row = 5, pieces_per_col = 5 )
        all_patches.append(patches_dict)
        frame_engery.append(patch_engery)
        frame_num += 1
    
    # Release the video capture object
    video_capture.release()
    return all_patches, frame_engery


def get_engery_diff_matrix(frame_engery):
    enger_diff_matrix = []
    for fr_en in frame_engery:
    
        input_engery_dict = fr_en
        
        indices = set(index for pair in input_engery_dict.keys() for index in pair)
        
        # Create an empty adjacency matrix
        matrix_size = max(indices) + 1
        adjacency_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        
        # Populate the adjacency matrix
        for (i, j), value in input_engery_dict.items():
            adjacency_matrix[i, j] = value
        
        # Print the adjacency matrix
        # print("Adjacency Matrix:")
        # print(adjacency_matrix)
        
        enger_diff_dict = {}
        max_diff_val = 0
        max_diff_val_id = (0,0)
        for i in range(len(adjacency_matrix)):
            for j in range(len(adjacency_matrix[0])):
        
                absolute_difference_matrix = np.abs(np.subtract.outer(adjacency_matrix[i, j], adjacency_matrix))
                #print(i, j, adjacency_matrix[i, j], np.mean(absolute_difference_matrix))
                norm_diff_mean = np.mean(absolute_difference_matrix)
                enger_diff_dict[i, j] = norm_diff_mean
                if norm_diff_mean > max_diff_val:
                    max_diff_val = norm_diff_mean
                    max_diff_val_id = (i, j)
                
        enger_diff_matrix.append([max_diff_val_id, max_diff_val])
    return enger_diff_matrix

def compute_euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))


def get_interest_patch(all_patches, enger_diff_matrix):
    frame_engery_patches = []
    
    for i in range(len(enger_diff_matrix)):
        key = enger_diff_matrix[i][0]
        patch_data = all_patches[i][key]
        frame_engery_patches.append(patch_data.flatten())
    
    return frame_engery_patches


def symbolise_video_energy_states(norm_data, ngram):
    
    def build_word(vector):
        concatenated_strings = []
        vector_length = len(vector)
        
        for i in range(vector_length - (ngram-1)):
            interval = vector[i:i + (ngram)]
            concatenated_string = "".join(map(str, interval))
            concatenated_strings.append(concatenated_string)
        
        return concatenated_strings
    
    norm_mean = np.mean(norm_data)
    norm_std = np.std(norm_data)
    symbol = []
    for val in norm_data:
        if (1+norm_std)*norm_mean < val: 
            symbol.append('A')
        elif norm_mean < val <= (1+norm_std)*norm_mean:
            symbol.append('B')
        elif (1-norm_std)*norm_mean < val <= norm_mean:
            symbol.append('C')
        else:
            symbol.append('D')

    # if ngram == 1:
    #     output = symbol
    # elif ngram == 2:
    output = build_word(symbol)
    return output