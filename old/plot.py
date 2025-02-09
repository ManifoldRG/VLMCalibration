import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier

#df = pd.read_csv('results-llama-8B.csv')
df = pd.read_csv('results-llama-70B.csv')
#df = pd.read_csv('results-gpt-4o-mini.csv')

#df = pd.read_csv('results-test-llama-8B.csv')
#df = pd.read_csv('results-test-llama-70B.csv')
#df = pd.read_csv('results-test-gpt-4o-mini.csv')


print(f'Acc: {df.correct.mean()}')
print(f'Avg Conf when Correct: {df[df.correct].confidence.mean()}')
print(f'Avg Conf when Inorrect: {df[~df.correct].confidence.mean()}')
df[df.correct].confidence.plot.kde(bw_method=0.4,label="Confidence when Corect")
df[~df.correct].confidence.plot.kde(bw_method=0.4,label="Confidence when Incorrect")
plt.xlim(0.0, 1)
plt.ylim(0, None)
plt.legend()
plt.show()

#plt.scatter(df.correct, df.confidence)
df[df.correct].confidence.hist(bins=np.linspace(0,1,201),alpha=0.5,density=True,label="Confidence when Corect")
df[~df.correct].confidence.hist(bins=np.linspace(0,1,201),alpha=0.5,density=True,label="Confidence when Incorrect")
plt.xlim(0.0, 1)
plt.legend()
plt.show()

# Prepare the feature matrix and target vector
X = df[['confidence']]
y = df['correct']

# Train a DecisionTreeClassifier with criterion='entropy' to use information gain
tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)
tree_clf.fit(X, y)

# The threshold of the best split
best_split = tree_clf.tree_.threshold[0]

# Extract the information gain (reduction in entropy)
info_gain = tree_clf.tree_.impurity[0] - (
    tree_clf.tree_.weighted_n_node_samples[1] / tree_clf.tree_.weighted_n_node_samples[0] * tree_clf.tree_.impurity[1] +
    tree_clf.tree_.weighted_n_node_samples[2] / tree_clf.tree_.weighted_n_node_samples[0] * tree_clf.tree_.impurity[2]
)

print(f"Best split at feature value: {best_split}, Information gain: {info_gain}")