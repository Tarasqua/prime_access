from matplotlib import pyplot as plt
import numpy as np
import time


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def get_mov_avrgs(data: np.array):
    pass


start_from = 0
end_ = 254
window = 5
data = np.load('in_centr.npy')
x, y = data[:, 0], data[:, 1]
m_x, m_y = moving_average(x, window), moving_average(y, window)
x = np.arange(0, len(m_x))
plt.plot(x, m_x, label='x')
plt.plot(x, m_y, label='y')
# x1, y1, x2, y2 = data[start_from:, 0], data[start_from:, 1], data[start_from:, 2], data[start_from:, 3]
# start_time = time.time()
# m_x1 = moving_average(x1, window)
# m_y1 = moving_average(y1, window)
# m_x2 = moving_average(x2, window)
# m_y2 = moving_average(y2, window)
# print("--- %s seconds ---" % (time.time() - start_time))
# x = np.arange(0, len(m_x1))
# plt.plot(x, m_x1, label='x1')
# plt.plot(x, m_x2, label='x2')
# plt.plot(x, m_y1, label='y1')
# plt.plot(x, m_y2, label='y2')
plt.legend()
plt.show()
