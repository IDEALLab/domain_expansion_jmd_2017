from matplotlib import rcParams
rcParams.update({'font.size': 20})

plt.figure(figsize=(12, 7))
line1, = plt.plot(f1s1, '-', label='Length scale = 0.5')
line2, = plt.plot(f1s2, '--', label='Length scale = 1.0')
plt.legend(handles=[line1, line2], loc=4)
plt.xlabel('Number of queries')
plt.ylabel('F1 score')
plt.tight_layout()

plt.show()