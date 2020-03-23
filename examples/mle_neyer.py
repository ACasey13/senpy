# the following adds the location of the senpy package (the top repo directory)
# to the python path. Is only needed if the package has not already been made
# accessible for import.
import sys
import os
senpy_package_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if senpy_package_location not in sys.path:
    sys.path.append(senpy_package_location)

#########################################
### begin execution of example script ###
#########################################
import numpy as np
from senpy.neyer import Neyer

# this is the example data provided in the Neyer paper
# X_data must be of shape [n_pts, 1]
x_data = np.array([1, 1.2, 1.4, 1.8, 2.6, 4.2, 3.4, 3.8, 4, 4.1, 4.28,
                   4.52, 5.55, 5.24, 6.37, 6.08, 7.38, 7.09, 6.89, 6.74]).reshape((-1,1))
y_data = np.array([0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,1,1,1,1])

# instantiate a Neyer object. In this case, we wish to assume a log-logistic
# latent distribution
est = Neyer(latent='log-logistic')

# call fit method on the Neyer estimator
est.fit(x_data, y_data)

# plot the results
# show=True simply call matplotlib.pyplot.show() which may be necessary
# depending on your IDE
est.plot_probability(confidence='likelihood-ratio', 
                     xlabel='Drop Height (m)', ylabel='Predicted Probability',
                     show=True)
est.plot_confidence_region([4, 7, 1, 20], [100], show=True)

# print the parameter estimates to console
est.print_estimators()
# printed to console:
# alpha: 5.317603889844265
# beta: 8.27725905945797


