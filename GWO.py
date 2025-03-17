import numpy as np
import cv2
import skimage 
from skimage import filters, io,  color, feature, measure
from skimage.filters import threshold_otsu
import copy
import random
from array import *
import math
################################################################

# initialization 
FCR = 0 # Fitness Computation Rate
FCR_max = 100 # Maximum Fitness Computation Rate
fitness_global_best = -math.inf 
X_best = [0, 0, 0, 0] #[a, b, c, k]
POP = 100 # population size
n = 3 # the size of the neighborhood 
MSE = 0 # Mean Square Error
lb_a = 2   # limitations for our prameters
ub_a = 2.5 # limitations for our prameters
lb_b = 0.3 # limitations for our prameters
ub_b = 0.5 # limitations for our prameters
lb_c = 0   # limitations for our prameters
ub_c = 3   # limitations for our prameters
lb_k = 3   # limitations for our prameters
ub_k = 4   # limitations for our prameters
FI = 0 # the number of foreground pixels φg 

X = np.zeros((100, 4), dtype="float") # structure of X or our random parameters

fitness = np.zeros((100, 1), dtype="float")

# GWO parameters

D_alpha = np.zeros((1, 4), dtype="float")
D_beta = np.zeros((1, 4), dtype="float")
D_delta = np.zeros((1, 4), dtype="float")

X_1 = np.zeros((1, 4), dtype="float")  # the value of wolves that should be averaged
X_2 = np.zeros((1, 4), dtype="float")  # the value of wolves that should be averaged
X_3 = np.zeros((1, 4), dtype="float")  # the value of wolves that should be averaged

X_alpha = np.zeros((1, 4), dtype="float")  # the value of alpha wolf
X_beta = np.zeros((1, 4), dtype="float")   # the value of beta wolf
X_delta = np.zeros((1, 4), dtype="float")  # the value of delta wolf

E_alpha = 0  # fitness of each wolf
E_beta = 0   # fitness of each wolf
E_delta = 0  # fitness of each wolf

A = 0 # A = a(2r_1 - 1)
a = 0 # 2 - ((2 * FCR)/(FRC_max))
C = 0 # C = 2 * R_2
r_1 = 0 # random parameter between 0 and 1
r_2 = 0 # random parameter between 0 and 1

# importing images
image_1 = cv2.imread("E:\\ali_project\\synpic15935_1.jpg", cv2.IMREAD_GRAYSCALE)/25.6
image_2 = cv2.imread("E:\\ali_project\\synpic28644_1.jpg", cv2.IMREAD_GRAYSCALE)/25.6
image_3 = cv2.imread("E:\\ali_project\\synpic51882_1.jpg", cv2.IMREAD_GRAYSCALE)/25.6

# Glaobal Mean
M_1 = np.mean(image_1) # global mean for image_1
M_2 = np.mean(image_2) # global mean for image_2
M_3 = np.mean(image_3) # global mean for image_3


# computing the gray level mean
def compute_mean(image, i, j):
    half_n = 3 // 2
    window = image[max(0, i-half_n):min(image.shape[0], i+half_n+1), 
                   max(0, j-half_n):min(image.shape[1], j+half_n+1)]
    return float(np.mean(window))#/27.5

# Computing the standard devation 
def compute_sigma(image, i, j):
    half_n = 3 // 2
    window = image[max(0, i-half_n):min(image.shape[0], i+half_n+1), 
                   max(0, j-half_n):min(image.shape[1], j+half_n+1)]
    mean = np.mean(window)
    return np.sqrt(np.mean((window - mean) ** 2))

# computing the transformed image
def transform_image(image, a, b, c, k, M):
    transformed_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            m_ij = compute_mean(image, i, j)
            sigma_ij = compute_sigma(image, i, j)
            f_ij = image[i, j]#/26.5
            g_ij = k * (M / (sigma_ij + b)) * (f_ij - c * m_ij) + m_ij ** a
            transformed_image[i, j] = g_ij
    # transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
    return transformed_image

# calculating the Entropy (Beta_g)
def calculate_entropy(image):
    # Calculate the histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Normalize the histogram to get probabilities
    hist = hist / hist.sum()
    # Calculate the entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Adding a small value to avoid log(0)
    return entropy

#################################################################### create the enhanced image
# Enhanced_image = transform_image(image_1, 2.50, 3.0, 0.9891, 4.0, M_1 * 25.5)#/25.5
# Enhanced_image = np.clip(Enhanced_image, 0, 255).astype(np.uint8)
# cv2.imwrite("E:\\ali_project\\Enhanced_image.jpg", Enhanced_image)
# cv2.imshow ("E:\\ali_project\\Enhanced_image.jpg", Enhanced_image)

# calculating the MSE
def MSE_calculation(image, Enhanced_image):
    MSE = 0
    for i in range(Enhanced_image.shape[0]):
        for j in range(Enhanced_image.shape[1]):
            MSE += math.sqrt(abs(image[i, j] - Enhanced_image[i, j]))
    MSE /= (256 * 256)
    return MSE
        
####### PSNR value
# ro = 10 * math.log10(((L - 1) ** 2)/MSE) # ro is PSNR
# L = Enhanced_image.max() # max pixel intensity value in g(i, j)
def max_intensity(image):
    L = image.max()
    return L

########## calculating the number of edge pixels 
def edge_pixels(Enhanced_image):
    sobel_x = cv2.Sobel(Enhanced_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(Enhanced_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    S_f = np.mean(sobel_magnitude)
    _, edge_pixels = cv2.threshold(sobel_magnitude, S_f, 255, cv2.THRESH_BINARY)
    n_edges = np.sum(edge_pixels > 0)
    return n_edges

########### n_edges = edge_pixels(enhanced_image)
# print("the number of edges are: ", n_edges)
def foreground_pixels(enhanced_image):
    threshold_value = threshold_otsu(enhanced_image)
    FI = np.sum(enhanced_image > threshold_value)
    return FI
    
# Evaluation Function or fitness function
def evaluation_function(Enhanced_image, original_image):
    number_of_edges = edge_pixels(Enhanced_image)                                           # calculating the number of edges
    # print(f'number_of_edges is {number_of_edges}')
    FI = foreground_pixels(Enhanced_image)                                                  # calculating the number of pixels belonging to the foreground object
    # print(f'FI is {FI}')
    mse = MSE_calculation(original_image, Enhanced_image)                                   # calculating the Mean Square Error
    # print(f'mse is {mse}')
    L = max_intensity(Enhanced_image)                                                       # calculating the maximum intensity valeu
    # print(f'L is {L}')
    RO = 10 * math.log10(((L - 1) ** 2)/mse) # ro is PSNR                                   # peak signal to noise ratio
    # print(f'RO is {RO}')
    Beta_g = calculate_entropy(Enhanced_image)                                              # calculating the entropic measure  
    # print(f'Beta_g is {Beta_g}')
    
    e_f = 1 - math.exp(-RO/100) + ((number_of_edges + FI) / (256 * 256)) + (Beta_g / 8)     # calculating the fitness which is between 0 to 4
    # print(f'fitness is : {e_f}')
    return e_f

while FCR < 200: # back before FCR_max was 200 but now its 20
    for p in range(POP):
        # creating random parameters
        X[p,0] = random.uniform(lb_a, ub_a) # creating random variables a
        X[p,1] = random.uniform(lb_b, ub_b) # creating random variables b
        X[p,2] = random.uniform(lb_c, ub_c) # creating random variables c
        X[p,3] = random.uniform(lb_k, ub_k) # creating random variables k
        
        # creating enhanced image and calculating the fitness
        G_ij = transform_image(image_1, X[p,0], X[p,1], X[p,2], X[p,3], M_1)
        G_ij = np.clip(G_ij, 0, 255).astype(np.uint8)
        fitness[p] = evaluation_function(G_ij, image_1)
        
        if fitness[p] > fitness_global_best:
            fitness_global_best = fitness[p]
            X_best = X[p]
    FCR += 1
    print("FCR IS:", FCR)
            
best_random_image_GWO = transform_image(image_1, X_best[0], X_best[1], X_best[2], X_best[3], M_1)
best_random_image_GWO = np.clip(best_random_image_GWO, 0, 255).astype(np.uint8)
cv2.imwrite("E:\\ali_project\\best_random_image_GWO.jpg", best_random_image_GWO)
cv2.imshow ("E:\\ali_project\\best_random_image_GWO.jpg", best_random_image_GWO)
print(f'\nBest random fitness is: {fitness_global_best}\nand best parameter valeus are: {X_best}\nPhase one has finished successfully')


FCR = 0
# X's shape is "100 x 4" and global best's shape is "1 x 4" 
global_best = X_best # the best set of parameters 

# GWO algorithm goes here
E_alpha = fitness_global_best # assigning the best fitness
X_alpha = global_best         # assigning the best values

# main loop
while FCR <= FCR_max:
    print("FCR is: " ,FCR)
    a = 2 - ((2 * FCR) / FCR_max)
    for p in range(POP):
        print("\titeration number is: ", p)
        if FCR > 1:
            G_ij = transform_image(image_1, X[p,0], X[p,1], X[p,2], X[p,3], M_1)
            G_ij = np.clip(G_ij, 0, 255).astype(np.uint8)
            fitness[p] = evaluation_function(G_ij, image_1)

        # updating X_alpha X_beta X_delta
        if fitness[p] > E_alpha:
            X_delta = X_beta
            E_delta = E_beta
            X_beta = X_alpha
            E_beta = E_alpha
            X_alpha = X[p]
            E_alpha = fitness[p]
        elif fitness[p] > E_beta:
            X_delta = X_beta
            E_delta = E_beta
            X_beta = X[p]
            E_beta = fitness[p]
        elif fitness[p] > E_delta:
            E_delta = fitness[p]
            X_delta = X[p]

        # updating X_1 X_2 X_3
        if FCR > 1:
            X_1 = X_alpha - np.dot((2 * a * random.random() - a), abs(np.dot((2 * random.random()), X_alpha) - X[p]))
            X_2 = X_beta  - np.dot((2 * a * random.random() - a), abs(np.dot((2 * random.random()), X_beta ) - X[p]))
            X_3 = X_delta - np.dot((2 * a * random.random() - a), abs(np.dot((2 * random.random()), X_delta) - X[p]))

            # updating X[p]
            X[p] = np.average([X_1, X_2, X_3])
            
            # limitiing the parameters in X 
            X[p, 0] = np.clip(X[p, 0], lb_a, ub_a)
            X[p, 1] = np.clip(X[p, 1], lb_b, ub_b)
            X[p, 2] = np.clip(X[p, 2], lb_c, ub_c)
            X[p, 3] = np.clip(X[p, 3], lb_k, ub_k)
    FCR += 1

# keep in mind that best results are X_alpha and E_alpha

# visualization the results
if X_alpha[2] > 0.96:
    result_image_GWO = transform_image(image_1, X_alpha[0], X_alpha[1], X_alpha[2], X_alpha[3], M_1 * 1.9)
elif X_alpha[2] < 0.96 and X_alpha > 0.93:
    result_image_GWO = transform_image(image_1, X_alpha[0], X_alpha[1], X_alpha[2], X_alpha[3], M_1 * 1.2)
elif X_alpha[2] < 0.93 and X_alpha > 0.89:
    result_image_GWO = transform_image(image_1, X_alpha[0], X_alpha[1], X_alpha[2], X_alpha[3], M_1 * 0.9)
else :
    result_image_GWO = transform_image(image_1, X_alpha[0], X_alpha[1], X_alpha[2], X_alpha[3], M_1)#/25.5

result_image_GWO = np.clip(result_image_GWO, 0, 255).astype(np.uint8)
cv2.imwrite("E:\\ali_project\\result_image_GWO.jpg", result_image_GWO)
cv2.imshow ("E:\\ali_project\\result_image_GWO.jpg", result_image_GWO)
print(f'\nBest random fitness was: {fitness_global_best}\nand best parameter valeus were: {X_best}\n\n\n')
print(f'\nFinal result --->>>\nBest fitness is: {E_alpha}\nand best parameters are: {X_alpha}') 
