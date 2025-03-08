import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

#Loading data
data = pd.read_csv(r'C:\Users\ch\PycharmProjects\Linear_regression_Assignement\Lab2 (1)\assignment\assignment1dataset.csv')
print("The Data is : ")
print("-------------------------------")
print(data)

feature_1 = data['NCustomersPerDay']
feature_2 = data['AverageOrderValue']
feature_3 = data['WorkingHoursPerDay']
feature_4 = data['NEmployees']
feature_5 = data['MarketingSpendPerDay']
feature_6 = data['LocationFootTraffic']
Target_column = data['RevenuePerDay']

# Convertin to numpy array : 
    
featureN1 = np.array(feature_1)
featureN2 = np.array(feature_2)
featureN3 = np.array(feature_3)
featureN4 = np.array(feature_4)
featureN5 = np.array(feature_5)
featureN6 = np.array(feature_6)

features_arr = [featureN1,featureN2,featureN3,featureN4,featureN5,featureN6]
feature_normalized_arr_of_arr = [] 
# Normalization of Data: 
for feature in features_arr:
    feature_normalized_arr = (feature - feature.min()) / (feature.max() - feature.min())
    feature_normalized_arr_of_arr.append(feature_normalized_arr)


mse_lr = [] 
mse_ep = []

LR = [0.1,0.01,0.001,0.0001]
for normalized_feature in feature_normalized_arr_of_arr:
    features_lists = []
    epoch = 100
    m = 0
    c = 0
    n = float(len(normalized_feature))
    for L in LR:
        for i in range(epoch):
            Y_pred = m*normalized_feature +c # current value of Y_predict
            D_m = (-2/n) * sum((Target_column - Y_pred) * normalized_feature)
            D_c = (-2/n) * sum(Target_column - Y_pred)
            m = m - L * D_m
            c = c - L * D_c 
            
        prediction = m*normalized_feature +c
# =============================================================================
#         plt.plot(normalized_feature, prediction)
#         plt.scatter(normalized_feature, Target_column, color = 'red')
#         plt.show()
# =============================================================================
        features_lists.append(((prediction - Target_column)**2).mean())
    mse_lr.append(features_lists) 
    
  
# Epochs : 
epochs = [100, 500, 1000, 2500]
for normalized_feature in feature_normalized_arr_of_arr:
    features_lists = []
    L = 0.01
    
    m = 0
    c = 0
    n = float(len(normalized_feature))
    for epoch in epochs:
        for i in range(epoch):
            Y_pred = m*normalized_feature +c # current value of Y_predict
            D_m = (-2/n) * sum((Target_column - Y_pred) * normalized_feature)
            D_c = (-2/n) * sum(Target_column - Y_pred)
            m = m - L * D_m
            c = c - L * D_c 
            
        prediction = m*normalized_feature +c
        features_lists.append(((prediction - Target_column)**2).mean())
    mse_ep.append(features_lists) 
    
best_mse = 1e18
best_feature = None

for i in range(6):
    if min(mse_ep[i]) < best_mse:
        best_mse = min(mse_ep[i])
        best_feature = i
    if min(mse_lr[i]) < best_mse:
        best_mse = min(mse_lr[i])
        best_feature = i

fig, axes = plt.subplots(2, 3, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    ax.plot(LR,mse_lr[i])  # Example: different scaled sine waves
    ax.set_title(f"feature {i+1}")
    ax.set_xlabel("Learning Rate (lr)")  
    ax.set_ylabel("mean squared error (MSE)")  
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    ax.plot(epochs,mse_ep[i])  # Example: different scaled sine waves
    ax.set_title(f"feature {i+1}")
    ax.set_xlabel("Epochs")  
    ax.set_ylabel("mean squared error (MSE)")  
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

for i in range(6):
    
    print(f"Best mse for feature {i + 1} is : {min(mse_ep)} with epochs = {epochs[mse_ep.index(min(mse_ep))]}")
    print(f"Best mse for feature {i + 1} is : {min(mse_lr)} with lr = {LR[mse_lr.index(min(mse_lr))]}")

          

print(f"the best feature is feature {best_feature + 1} with mse : {best_mse}")
# =============================================================================
# for i in range(6):
#     plt.plot(LR,mse_lr[i])
#     plt.show()
#     plt.plot(epochs, mse_ep[i])
#     plt.show()
# =============================================================================
    
