## 2. Matrix Vector Multiplication ##

import numpy as np

matrix_a = np.asarray( [[0.7,3,9],[1.7,2,9],[0.7,9,2] ], dtype = np.float32)
vector_b = np.asarray( [[1],[2],[1]] , dtype = np.float32)
                       
ab_product  = matrix_a.dot(vector_b)
                       
print(ab_product)

## 3. Matrix Multiplication ##

matrix_a = np.asarray([
    [0.7, 3],
    [1.7, 2],
    [0.7, 9]
], dtype=np.float32)

matrix_b = np.asarray([
    [113, 3, 10],
    [1, 0, 1],
], dtype=np.float32)

product_ab = np.dot(matrix_a, matrix_b)
product_ba = np.dot(matrix_b, matrix_a)

## 4. Matrix Transpose ##

matrix_a = np.asarray([
    [0.7, 3],
    [1.7, 2],
    [0.7, 9]
], dtype=np.float32)

matrix_b = np.asarray([
    [113, 3, 10],
    [1, 0, 1],
], dtype=np.float32)

transpose_a = np.transpose(matrix_a)
print (transpose_a)

matrix_a = np.transpose( transpose_a)

print(matrix_a)

trans_ba =  np.transpose(matrix_b).dot( np.transpose(matrix_a))

trans_ab =  np.transpose(matrix_a).dot( np.transpose(matrix_b))
print(trans_ba)
product_ab = np.dot(matrix_a , matrix_b)

print(product_ab )





## 5. Identity Matrix ##

import numpy as np
i_2 = np.identity(2)
i_3 = np.identity(3)
matrix_33 = np.asarray( [[1,2,3],[4,5,6],[7,8,9] ] ,dtype = np.float32)
matrix_23 = np.asarray( [[9,8,7],[1,2,3] ] ,dtype = np.float32)

identity_33 = i_3.dot(matrix_33)

identity_23 = i_2.dot(matrix_23 )



## 6. Matrix Inverse ##

matrix_a = np.asarray([
    [1.5, 3],
    [1, 4]
])


def matrix_inverse_two(matrix_a) :
    
    determinant =  ( matrix_a[0][0] * matrix_a[1][1]) - (matrix_a[0][1] * matrix_a[1][0])
    
    if determinant  == 0 :
        return "Error"
    else :
        matrix_ai = (1/abs(determinant)) * np.asarray([[matrix_a[1][1], -1 *matrix_a[0][1] ] ,[ -1 * matrix_a[1][0] ,  matrix_a[0][0]  ] 
                                                 
                                                 ] , dtype =np.float32)
        return matrix_ai
        
inverse_a = matrix_inverse_two(matrix_a)
    
i_2 = np.dot( matrix_a ,inverse_a )
    

## 7. Solving The Matrix Equation ##

matrix_a = np.asarray([
    [30, -1],
    [50, -1]
])

vector_b = np.asarray([
    [-1000],
    [-100]
])
matrix_a_inverse = np.linalg.inv(matrix_a)
solution_x = np.dot(matrix_a_inverse, vector_b)
print(solution_x)

## 8. Determinant For Higher Dimensions ##

matrix_22 = np.asarray([
    [8, 4],
    [4, 2]
])

matrix_33 = np.asarray([
    [1, 1, 1],
    [1, 1, 6],
    [7, 8, 9]
])

det_22 = np.linalg.det(matrix_22 )
det_33 = np.linalg.det(matrix_33)

