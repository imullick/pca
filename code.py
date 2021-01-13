#Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

#----------------------------------------------------------------------------------------------------------

# Loading dataset into Pandas DataFrame and converting to suitable form

df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Separating out the features data as an array (150 entries x 4 features)
x=np.array(df.iloc[:,0:-1])


# Separating the targets from dataframe and creating an array of targets (150 x 1)
y=[]
for str in df["target"]:
    y.append(str)
    
#----------------------------------------------------------------------------------------------------------

# Standardizing the features

# Generating array of means
mean = np.zeros((4,1))
for j in range(4):
    sum=0
    for i in range(len(x)):
        sum = sum + x[i,j]
    mean[j] = sum/i
    
# Generating array of variances, further modified to create array of standard deviations
# var = np.zeros((4,1))
# for j in range(4):
#     sum=0
#     for i in range(len(x)):
#         sum = sum + (x[i,j] - mean[j])**2
#     var[j] = sum/i
   
# Generating array of standard deviations 
    
std = np.zeros((4,1))
for j in range(4):
    sum=0
    for i in range(len(x)):
        sum = sum + (x[i,j] - mean[j])**2
    std[j] = (sum/i)**(1/2)

# Calculating z score (z = (xi - mean)/std)
    
for j in range(4):
    for i in range(len(x)):
        x[i,j] = (x[i,j] - mean[j])/std[j]

#----------------------------------------------------------------------------------------------------------

# COVARIANCE MATRIX

# df_cols -> array of data of dataframe column wise i.e. data of each feature
df_cols = []
for j in range(4):
    for i in df.columns[:-1]:
        df_cols.append(np.array(df[i]))

cov_mat = np.zeros((4,4)) # 4 features are present

# Function to calculate covariance using - 
# E(Xi - X_mean)(Yi - Y_mean)

def my_cov1(i,j):                           
    vect1=df_cols[i]
    vect2=df_cols[j]
    mean1=mean[i]
    mean2=mean[j]
    sum=0
    for k in range(len(vect1)):
        sum = sum + float(vect1[k]-mean1)*float(vect2[k]-mean2)
    cov=sum/len(vect1)
    return cov

for i in range(4):
    for j in range(4):
        cov_mat[i,j]=my_cov1(i,j)
    
#----------------------------------------------------------------------------------------------------------
      
        
# Calculation of eigenvalues and eigenvectors using inbuilt function
eigval, eigvec = np.linalg.eig(cov_mat)

# Sorting eigenvalues (and switching corresponding eigenvectors) using bubble sort
for i in range(len(eigval)-1):
    print(i)
    for j in range(i+1, len(eigval)):
        print(j)
        if(eigval[i] > eigval[j]):
            temp = eigval[i]
            eigval[i]=eigval[j]
            eigval[j]=temp
            array = eigvec[i].copy()
            eigvec[i]=eigvec[j]
            eigvec[j]=array

# Choosing top 2 eigenvectors (since they have high eigenvalues,
# we know that they tell us about our dataset the most) -
            
print("As percentages: ")
sum = 0
for i in eigval:
    sum = sum + i
for i in range(len(eigval)):
    print(f'Eigenvalue {i+1}: {eigval[i]/sum *100}')

# It can be seen that the 4th eigenvector has the most impact on the dataset
# followed by eigenvector 3. Eigenvector 1 & 2 have %ages nearing 0.
    
eigvec_mat = eigvec[:-2] # Selecting top 2 eigenvectors

# Transpose of eigenvector matrix
mat = np.zeros((4,2))
for i in range(2):
   for j in range(4):
       mat[j][i] = eigvec_mat[i][j]

mat = np.asmatrix(mat) 

# Transformed matrix
final_mat = x.dot(mat)

#----------------------------------------------------------------------------------------------------------

# Plotting transformed matrices

# Converting targets to numbers so that they can be plotted
for i in range(len(y)):
    if y[i]=="Iris-setosa":
        y[i]=0
    elif y[i]=="Iris-versicolor":
        y[i]=1
    else:
        y[i]=2
y=np.asarray(y)

# Reshaping of matrices so that they can be plotted
matrix1=np.asarray(final_mat[:,0].reshape(150,1)) # First column of transformed matrix
matrix1 = matrix1.reshape(matrix1.shape[0])
matrix2=np.asarray(final_mat[:,1].reshape(150,1)) # Second column of transformed matrix
matrix2 = matrix2.reshape(matrix1.shape[0])

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(
    matrix1,
    matrix2,
    c=y, # Division by target so that the plotted points can be differentiated by colour
    cmap='rainbow',
    edgecolors='b'
)

#----------------------------------------------------------------------------------------------------------
