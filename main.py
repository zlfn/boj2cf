import pickle
import torch
import torch.optim as optim


with open('new_user_infos.pkl', 'rb') as f:
    user_infos = pickle.load(f)

boj_cf = [[], []]
boj_at = [[], []]
cf_at = [[], []]

boj_train = []
cf_train = []
at_train = []

count = 0

for u in user_infos:
    if 'cf_rating' in u:
        boj_cf[0].append(u['rating'])
        boj_cf[1].append(u['cf_rating'])
    if 'atcoder_rating' in u:
        boj_at[0].append(u['rating'])
        boj_at[1].append(u['atcoder_rating'])
    if 'atcoder_rating' in u and 'cf_rating' in u:
        cf_at[0].append(u['cf_rating'])
        cf_at[1].append(u['atcoder_rating'])
        boj_train.append(u['rating'])
        cf_train.append(u['cf_rating'])
        at_train.append([u['atcoder_rating']])

x_train = [[boj_train[i], cf_train[i]] for i in range(len(boj_train))]
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(at_train)
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad = True)

optimizer = optim.SGD([W, b], lr=1e-10)
nb_epochs = 3000
for epochs in range(nb_epochs + 1):
    hypothesis = x_train.matmul(W) + b
    cost = torch.mean((hypothesis - y_train) ** 2)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

print(W[0].item())
print(W[1].item())
print(b[0].item())

at_pre = [[], []]
for u in user_infos:
    if 'cf_rating' in u:
        at_pre[0].append(u['rating'])
        prediction = W[0].item() * u['rating'] + W[1].item() * u['cf_rating'] + b[0].item()
        at_pre[1].append(prediction)

import matplotlib.pyplot as plt

plt.scatter(boj_at[0], boj_at[1], color='r', s=4, zorder=2)
plt.scatter(at_pre[0], at_pre[1], color='orange', s=1, zorder=1)
plt.scatter(boj_cf[0], boj_cf[1], color='b', s=1, zorder=0)
plt.legend(['atcoder', 'atcoder(prediction)', 'codeforces'])
plt.xlabel('solved.ac')
plt.ylabel('cf/at')
plt.title('solved.ac to CF / AT ')
plt.show()
