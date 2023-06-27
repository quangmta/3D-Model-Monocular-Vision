import sympy
import numpy as np
from matplotlib import pyplot as plt

# Define the variables
# x, y = sympy.symbols('x y')

# # Define the linear equation
# eq1 = sympy.Eq(2.43*x + y, 2.03)
# eq2 = sympy.Eq(2.73*x + y, 2.26)

# # Solve for x and y
# sol = sympy.solve((eq1, eq2), (x, y))

# print(sol)


x = [2.43,2.73,3.46,1.1,2.00,2.46,1.28,1.87]
y = [2.03,2.26,3.30,0.6,1.60,2.30,0.70,1.00]
# coeff = [ 1.17846871, -0.82910784]
# x = [2.40,1.15,2.50,2.35,2.67,3.06,1.42,2.10,2.03,3.30,1.94,1.28]
# y = [1.50,0.60,1.83,2.03,2.23,2.50,0.50,1.30,1.75,3.30,1.00,0.70]
# coeff = [ 1.19410183, -1.003789  ]
data = np.array([x,y])
data = data[:, np.argsort(data[0])]
coeff = np.polyfit(data[0],data[1],1)
print(coeff)
y_new = np.polyval(coeff,data[0])
dis = abs(y_new - data[1])


print(dis)
print(data[1]-data[0])
print(y_new-data[0])
print(dis.mean(),dis.max(),dis.min())
# plt.scatter(data[0],abs(data[1]-data[0]))
# plt.scatter(data[0],abs(y_new-data[0]))
plt.scatter(data[0],data[1])
plt.plot(data[0],y_new)
plt.show()
