import numpy as np
import matplotlib.pyplot as plt

D = 1000
df = np.linspace(0, 1, 500)

idf = 1 / df
idf_log = np.log(1 / df)
idf_smooth = np.log(1 / (df + 1/D)) + 1

plt.plot(df, idf,        label='$\dfrac{1}{df}$')
plt.plot(df, idf_log,    label='$\log \dfrac{1}{df}$')
plt.plot(df, idf_smooth, label='$\log \dfrac{1}{df+\epsilon} + 1$')
plt.xlim(0, 1)
plt.ylim(0, 10)
plt.xlabel('document frequency ($df$)')
plt.ylabel('inverse document frequency ($idf$)')
plt.legend()
plt.grid()