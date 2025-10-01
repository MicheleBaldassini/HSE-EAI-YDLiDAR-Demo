from random import sample

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, timedelta

# Worker 1
m_a = np.array(sample(range(360), 14))
x_a = datetime(2022, 11, 10, 9, 0, 0) + m_a * timedelta(minutes=1)

m = np.setdiff1d(range(360), m_a)
print(type(m))
m_0 = np.random.choice(m, size=144, replace=False, p=None)
x_0 = datetime(2022, 11, 10, 9, 0, 0) + m_0 * timedelta(minutes=1)

m = np.union1d(m_a, m_0)
m = np.setdiff1d(range(360), m)
m_1 = np.random.choice(m, size=37, replace=False, p=None)
x_1 = datetime(2022, 11, 10, 9, 0, 0) + m_1 * timedelta(minutes=1)

m = np.union1d(m_a, np.union1d(m_0, m_1))
m = np.setdiff1d(range(360), m)
m_2 = np.random.choice(m, size=44, replace=False, p=None)
x_2 = datetime(2022, 11, 10, 9, 0, 0) + m_2 * timedelta(minutes=1)

m = np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, m_2)))
m = np.setdiff1d(range(360), m)
m_3 = np.random.choice(m, size=44, replace=False, p=None)
x_3 = datetime(2022, 11, 10, 9, 0, 0) + m_3 * timedelta(minutes=1)

m = np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, m_3))))
m = np.setdiff1d(range(360), m)
m_4 = np.random.choice(m, size=26, replace=False, p=None)
x_4 = datetime(2022, 11, 10, 9, 0, 0) + m_4 * timedelta(minutes=1)

m = np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, np.union1d(m_3, m_4)))))
m = np.setdiff1d(range(360), m)
m_5 = np.random.choice(m, size=26, replace=False, p=None)
x_5 = datetime(2022, 11, 10, 9, 0, 0) + m_5 * timedelta(minutes=1)

m = np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, np.union1d(m_3, np.union1d(m_4, m_5))))))
m = np.setdiff1d(range(360), m)
m_6 = np.random.choice(m, size=24, replace=False, p=None)
x_6 = datetime(2022, 11, 10, 9, 0, 0) + m_6 * timedelta(minutes=1)


fig, ax = plt.subplots(figsize=(16, 3))
ax.plot(x_a, 1*np.ones(x_a.shape), 'd', markerfacecolor='#D95319', markeredgecolor='#D95319')
ax.plot(x_0, 2*np.ones(x_0.shape), 'd', markerfacecolor='#008400', markeredgecolor='#008400')
ax.plot(x_1, 3*np.ones(x_1.shape), 'd', markerfacecolor='#229954', markeredgecolor='#229954')
ax.plot(x_2, 4*np.ones(x_2.shape), 'd', markerfacecolor='#1A5276', markeredgecolor='#1A5276')
ax.plot(x_3, 5*np.ones(x_3.shape), 'd', markerfacecolor='#5B2C6F', markeredgecolor='#5B2C6F')
ax.plot(x_4, 6*np.ones(x_4.shape), 'd', markerfacecolor='#DE3163', markeredgecolor='#DE3163')
ax.plot(x_5, 7*np.ones(x_5.shape), 'd', markerfacecolor='#68A810', markeredgecolor='#68A810')
ax.plot(x_6, 8*np.ones(x_6.shape), 'd', markerfacecolor='#9D3F5E', markeredgecolor='#9D3F5E')

#ax.set_ylim([0.75, 8.25])
ax.set_yticks(range(1, 9))
ax.set_yticklabels(['-1', '0', '1', '2', '3', '5', '8', '11'])
ax.set_xlim([datetime(2022, 11, 10, 9, 0, 0), datetime(2022, 11, 10, 15, 0, 0)])
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.show()
# datetick('x', 'HH:MM')
# set(gca, 'YTick', 1:8, 'YTickLabel', {'11', '8', '5', '3', '2', '1', '0', '-1'})
# set(gcf, 'color', 'w')

print(np.intersect1d(m_a, m_0))
print(np.intersect1d(np.union1d(m_a, m_0), m_1))
print(np.intersect1d(np.union1d(m_a, np.union1d(m_0, m_1)), m_2))
print(np.intersect1d(np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, m_2))), m_3))
print(np.intersect1d(np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, m_3)))), m_4))
print(np.intersect1d(np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, np.union1d(m_3, m_4))))), m_5))
print(np.intersect1d(np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, np.union1d(m_3, np.union1d(m_4, m_5)))))), m_6))


# # Worker 2
# m_a = sample(range(360), 14)
# x_a = datetime(2022, 11, 10, 9, 0, 0) + minutes(m_a)

# m = np.setdiff1d(range(360), m_a)
# m_0 = sample(m, 144)
# x_0 = datetime(2022, 11, 10, 9, 0, 0) + minutes(m_0)

# m = np.union1d(m_a, m_0)
# m = np.setdiff1d(range(360), m)
# m_1 = sample(m, 37)
# x_1 = datetime(2022, 11, 10, 9, 0, 0) + minutes(m_1)

# m = np.union1d(m_a, np.union1d(m_0, m_1))
# m = np.setdiff1d(range(360), m)
# m_2 = sample(m, 44)
# x_2 = datetime(2022, 11, 10, 9, 0, 0) + minutes(m_2)

# m = np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, m_2)))
# m = np.setdiff1d(range(360), m)
# m_3 = sample(m, 44)
# x_3 = datetime(2022, 11, 10, 9, 0, 0) + minutes(m_3)

# m = np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, m_3))))
# m = np.setdiff1d(range(360), m)
# m_4 = sample(m, 26)
# x_4 = datetime(2022, 11, 10, 9, 0, 0) + minutes(m_4)

# m = np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, np.union1d(m_3, m_4)))))
# m = np.setdiff1d(range(360), m)
# m_5 = sample(m, 26)
# x_5 = datetime(2022, 11, 10, 9, 0, 0) + minutes(m_5)

# m = np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, np.union1d(m_3, np.union1d(m_4, m_5))))))
# m = np.setdiff1d(range(360), m)
# m_6 = sample(m, 24)
# x_6 = datetime(2022, 11, 10, 9, 0, 0) + minutes(m_6)


# plt.plot(x_a, 8*ones(size(x_a)), 'o', 'MarkerFaceColor', '#77AC30', 'MarkerEdgeColor', '#77AC30')
# plt.plot(x_0  7*ones(size(x_0)), 'o', 'MarkerFaceColor', '#77AC30', 'MarkerEdgeColor', '#77AC30')
# plt.plot(x_1, 6*ones(size(x_1)), 'o', 'MarkerFaceColor', '#77AC30', 'MarkerEdgeColor', '#77AC30')
# plt.plot(x_2, 5*ones(size(x_2)), 'o', 'MarkerFaceColor', '#77AC30', 'MarkerEdgeColor', '#77AC30')
# plt.plot(x_3, 4*ones(size(x_3)), 'o', 'MarkerFaceColor', '#77AC30', 'MarkerEdgeColor', '#77AC30')
# plt.plot(x_4, 3*ones(size(x_4)), 'o', 'MarkerFaceColor', '#77AC30', 'MarkerEdgeColor', '#77AC30')
# plt.plot(x_5, 2*ones(size(x_5)), 'o', 'MarkerFaceColor', '#77AC30', 'MarkerEdgeColor', '#77AC30')
# plt.plot(x_6, 1*ones(size(x_6)), 'o', 'MarkerFaceColor', '#77AC30', 'MarkerEdgeColor', '#77AC30')

# hold off
# ylim([0 9])
# datetick('x', 'HH:MM')
# set(gca, 'YTick', 1:8, 'YTickLabel', {'11', '8', '5', '3', '2', '1', '0', '-1'})
# set(gcf, 'color', 'w')

# print(np.intersect1d(m_a, m_0))
# print(np.intersect1d(np.union1d(m_a, m_0), m_1))
# print(np.intersect1d(np.union1d(m_a, np.union1d(m_0, m_1)), m_2))
# print(np.intersect1d(np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, m_2))), m_3))
# print(np.intersect1d(np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, m_3)))), m_4))
# print(np.intersect1d(np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, np.union1d(m_3, m_4))))), m_5))
# print(np.intersect1d(np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, np.union1d(m_3, np.union1d(m_4, m_5)))))), m_6))


# # Worker 3
# m_a = sample(range(360), 14)
# x_a = datetime(2022, 11, 10, 9, 0, 0) + minutes(m_a)

# m = np.setdiff1d(range(360), m_a)
# m_0 = sample(m, 144)
# x_0 = datetime(2022, 11, 10, 9, 0, 0) + minutes(m_0)

# m = np.union1d(m_a, m_0)
# m = np.setdiff1d(range(360), m)
# m_1 = sample(m, 37)
# x_1 = datetime(2022, 11, 10, 9, 0, 0) + minutes(m_1)

# m = np.union1d(m_a, np.union1d(m_0, m_1))
# m = np.setdiff1d(range(360), m)
# m_2 = sample(m, 44)
# x_2 = datetime(2022, 11, 10, 9, 0, 0) + minutes(m_2)

# m = np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, m_2)))
# m = np.setdiff1d(range(360), m)
# m_3 = sample(m, 44)
# x_3 = datetime(2022, 11, 10, 9, 0, 0) + minutes(m_3)

# m = np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, m_3))))
# m = np.setdiff1d(range(360), m)
# m_4 = sample(m, 26)
# x_4 = datetime(2022, 11, 10, 9, 0, 0) + minutes(m_4)

# m = np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, np.union1d(m_3, m_4)))))
# m = np.setdiff1d(range(360), m)
# m_5 = sample(m, 26)
# x_5 = datetime(2022, 11, 10, 9, 0, 0) + minutes(m_5)

# m = np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, np.union1d(m_3, np.union1d(m_4, m_5))))))
# m = np.setdiff1d(range(360), m)
# m_6 = sample(m, 24)
# x_6 = datetime(2022, 11, 10, 9, 0, 0) + minutes(m_6)


# figure
# plt.plot(x_a, 8*ones(size(x_a)), 'o', 'MarkerFaceColor', '#0072BD', 'MarkerEdgeColor', '#0072BD')
# hold on
# plt.plot(x_0,  7*ones(size(x_0)), 'o', 'MarkerFaceColor', '#0072BD', 'MarkerEdgeColor', '#0072BD')
# plt.plot(x_1, 6*ones(size(x_1)), 'o', 'MarkerFaceColor', '#0072BD', 'MarkerEdgeColor', '#0072BD')
# plt.plot(x_2, 5*ones(size(x_2)), 'o', 'MarkerFaceColor', '#0072BD', 'MarkerEdgeColor', '#0072BD')
# plt.plot(x_3, 4*ones(size(x_3)), 'o', 'MarkerFaceColor', '#0072BD', 'MarkerEdgeColor', '#0072BD')
# plt.plot(x_4, 3*ones(size(x_4)), 'o', 'MarkerFaceColor', '#0072BD', 'MarkerEdgeColor', '#0072BD')
# plt.plot(x_5, 2*ones(size(x_5)), 'o', 'MarkerFaceColor', '#0072BD', 'MarkerEdgeColor', '#0072BD')
# plt.plot(x_6, 1*ones(size(x_6)), 'o', 'MarkerFaceColor', '#0072BD', 'MarkerEdgeColor', '#0072BD')

# hold off
# ylim([0 9])
# datetick('x', 'HH:MM')
# set(gca, 'YTick', 1:8, 'YTickLabel', {'11', '8', '5', '3', '2', '1', '0', '-1'})
# set(gcf, 'color', 'w')

# print(np.intersect1d(m_a, m_0))
# print(np.intersect1d(np.union1d(m_a, m_0), m_1))
# print(np.intersect1d(np.union1d(m_a, np.union1d(m_0, m_1)), m_2))
# print(np.intersect1d(np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, m_2))), m_3))
# print(np.intersect1d(np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, m_3)))), m_4))
# print(np.intersect1d(np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, np.union1d(m_3, m_4))))), m_5))
# print(np.intersect1d(np.union1d(m_a, np.union1d(m_0, np.union1d(m_1, np.union1d(m_2, np.union1d(m_3, np.union1d(m_4, m_5)))))), m_6))