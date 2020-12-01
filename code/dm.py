import pandas as pd
import matplotlib.pyplot as plt
import statistics

data = pd.read_csv("agreed.csv", sep=",")
a = data['s1tos2'].mean()
b = data['s2tos1'].mean()

print(f'Mean for s1tos2:  {a:.4f}')
print(f'Mean for s2tos1: {b:.4f}')
am = (a+b)/2
print(f'Overall Mean for Agreed: {am:.4f}')

s1tos2 = data['s1tos2'].tolist()

res = statistics.pstdev(s1tos2)
print(f'Standard Deviation for Agreed: {res:.4f}')

plt.figure('Agree')
plt.title("Distribution of Agree")
plt.hist(s1tos2, 5)
plt.show()

print("-----------------------")

data = pd.read_csv("disagreed.csv", sep=",")
a = data['s1tos2'].mean()
b = data['s2tos1'].mean()

print(f'Mean for s1tos2:  {a:.4f}')
print(f'Mean for s2tos1: {b:.4f}')
am = (a+b)/2
print(f'Overall Mean for Disagreed: {am:.4f}')

s1tos2 = data['s1tos2'].tolist()

res = statistics.pstdev(s1tos2)
print(f'Standard Deviation for Disagreed: {res:.4f}')

plt.figure('Disagree')
plt.title("Distribution of Disagree")
plt.hist(s1tos2, 5)
plt.show()
