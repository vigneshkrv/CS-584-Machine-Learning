import matplotlib.pyplot as plt
import numpy
import random
import sklearn.ensemble as ensemble
import sklearn.linear_model as lm

nSample = 1001

# Generate (x,y) where y = a + b*x
random.seed(a = 20181128)

x = numpy.zeros((nSample,1))
y = numpy.zeros((nSample,1))

for i in range(nSample):
    x[i] = (i - 200.0) / 100.0
    y[i] = random.gauss((50.0 + 0.1 * x[i]), 5.0)

# Descriptives
print('Mean of x = ', x.mean(), 'Mean of y = ', y.mean())

# Plot the data
plt.figure(figsize=(10,6))
plt.scatter(x = x, y = y)
plt.grid(axis='both')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Build an Intercept-only regression model (i.e., y-hat = mean of y) on the training sample
all0 = numpy.zeros((nSample,1))
reg0 = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
fit0 = reg0.fit(all0,y)
predY0 = reg0.predict(all0)
RSq0 = reg0.score(all0,y)
print('Model 0:', 'R-Squared = ', RSq0, '\n Intercept = ', reg0.intercept_, '\n Coefficients = ', reg0.coef_)

# Calculate the residuals of the model 0
residY0 = y - predY0

# Build a regression model on the residuals
reg1 = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
fit1 = reg1.fit(x, residY0)
predY1 = reg1.predict(x)
RSq1 = reg1.score(x, residY0)
print('Model 1:', 'R-Squared = ', RSq1, '\n Intercept = ', reg1.intercept_, '\n Coefficients = ', reg1.coef_)

# Calculate the residuals of the model 1
residY1 = residY0 - predY1

# Build a regression model on the residuals (there should be no additional information)
reg2 = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
fit2 = reg2.fit(x, residY1)
predY2 = reg2.predict(x)
RSq2 = reg2.score(x, residY1)
print('Model 2:', 'R-Squared = ', RSq2, '\n Intercept = ', reg2.intercept_, '\n Coefficients = ', reg2.coef_)

# The typical regression model
regC = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
fitC = regC.fit(x, y)
predYC = regC.predict(x)
RSqC = regC.score(x, y)
print('Complete Model :', 'R-Squared = ', RSqC, '\n Intercept = ', regC.intercept_, '\n Coefficients = ', regC.coef_)

# Plot the data and the original regression line
plt.figure(figsize=(10,6))
plt.scatter(x, y)
plt.scatter(x, predYC, c = 'red')
plt.grid(axis='both')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
