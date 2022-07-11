import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

data = sio.loadmat(r" ")
data = data['data']   #COIL用data,HAPT_no_Bayes用HAPT
data = np.array(data)

data = {
    'M1': data[:,0],
    'M2': data[:,1],
    'M3': data[:,2],
    "M4": data[:,3],
    "M5": data[:,4],
    "M6": data[:,5],
    "M7": data[:,6],
    "M8": data[:,7],
    "M9": data[:,8],
    "M10": data[:,9]
}

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


df = pd.DataFrame(data)
df.plot.box()
plt.title("(i) θ = 0.9",fontsize = 20)
plt.grid(linestyle="--", alpha=0.3)
# plt.savefig('E:\重邮\大数据中心\image/pic-BOX-end-{}.tiff'.format(0.9),dpi = 300, bbox_inches='tight')
plt.show()