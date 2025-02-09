import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression

df = pd.read_csv('records_train_full.csv') # llama

# Handle NaN values in correct column
df['correct'] = df['correct'].fillna(False)

correct_df = df[df.correct]
incorrect_df = df[~df.correct]

print(correct_df.p_true.mean())
print(incorrect_df.p_true.mean())

df[df.correct].p_true.hist(bins=np.linspace(0,1,51),alpha=0.5,density=True,label="Confidence when Correct")
df[~df.correct].p_true.hist(bins=np.linspace(0,1,51),alpha=0.5,density=True,label="Confidence when Incorrect")
plt.xlim(0.0, 1)
plt.legend()
plt.savefig('train_confidence_hist_llama.png')

bins = np.linspace(0,1,41)
bin_values = []
for i in range(len(bins) - 1):
    bin_values.append(df[(df.p_true >= bins[i]) & (df.p_true < bins[i+1])].correct.mean())

iso_reg = IsotonicRegression().fit((bins[:-1] + bins[1:]) / 2, bin_values)
plt.scatter((bins[:-1] + bins[1:]) / 2, bin_values)
plt.plot(iso_reg.predict((bins[:-1] + bins[1:]) / 2), (bins[:-1] + bins[1:]) / 2)
plt.grid('both')
plt.title('Train')
plt.savefig('train_isotonic_llama.png')

df = pd.read_csv('records_test_full_llama.csv')
df['correct'] = df['correct'].fillna(False)

bins = np.linspace(0,1,51)
bin_values = []
for i in range(len(bins) - 1):
    bin_values.append(df[(df.p_true >= bins[i]) & (df.p_true < bins[i+1])].correct.mean())
plt.scatter((bins[:-1] + bins[1:]) / 2, bin_values)
plt.grid('both')
plt.title('Test Uncorrected')
plt.savefig('test_uncorrected_llama.png')

plt.scatter(iso_reg.predict((bins[:-1] + bins[1:]) / 2), bin_values)
plt.grid('both')
plt.title('Test Corrected')
plt.savefig('test_corrected_llama.png')




points = []
for thresh in list(df.p_true.unique()):
    # pred_correct = df[df.confidence >= thresh]
    # tpr = pred_correct.correct.mean()
    # fpr = sum(~pred_correct.correct) / len(df[~df.correct])
    tp = len(df[(df.p_true >= thresh) & df.correct])
    fp = len(df[(df.p_true >= thresh) & ~df.correct])
    tpr = tp / len(df[df.correct])
    fpr = fp / len(df[~df.correct])

    points.append((fpr, tpr))

points.append((0, 0))
points = sorted(points, key=lambda p: p[0])
auroc = 0
for p1, p2 in zip(points[:-1], points[1:]):
    auroc += (p2[0] - p1[0]) * (p1[1] + p2[1]) / 2
print(f'AUROC: {auroc}')
fprs = [p[0] for p in points]
tprs = [p[1] for p in points]
plt.plot(fprs, tprs)
plt.plot([0,1], [0,1], linestyle='--', c='orange')
plt.grid()
plt.savefig('test_auroc_llama.png')