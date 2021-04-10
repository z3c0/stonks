import os
import random
import numpy as np
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
os.environ['PYTHONHASHSEED'] = '0'

x = np.linspace(0, 10)


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)


set_seeds()

y = x + np.random.standard_normal(len(x))

regression = np.polyfit(x, y, deg=1)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo', label='observation')

x_predict = np.linspace(0, 20)
plt.plot(x_predict, np.polyval(regression, x_predict), 'r',
         lw=2.5, label='linear regression')
plt.legend(loc=0)
plt.show()
