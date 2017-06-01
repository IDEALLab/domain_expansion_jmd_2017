from matplotlib import rcParams
rcParams.update({'font.size': 20})

plt.figure(figsize=(12, 7))
line1, = plt.plot(f1s1, '-', label=r'$\epsilon$-margin sampling', linewidth=2)
line2, = plt.plot(f1s2, '--', label='Uncertainty sampling', linewidth=2)
line3, = plt.plot(f1s3, ':', label='Random sampling', linewidth=2)
plt.legend(handles=[line1, line2, line3], loc=2)
plt.xlabel('Number of queries')
plt.ylabel('F1 score')
plt.tight_layout()

plt.show()