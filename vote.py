import numpy as np

def vote(a,b,c):
    if a == b:
        return a
    if b == c:
        return b
    if a == c:
        return a
    return a

def getTestLabel():
    f = open('./out.txt')
    data = f.read().split('\n')
    ans = []
    dicts = {}
    for i in data:
        array = i.split(' ')
        if len(array) != 2:
            continue
        dicts[array[0]] = array[1]
    for i in range(1, len(dicts)+1):
        ans.append(dicts[str(i)])
    return ans

svm = np.load('./vote/output_svm.npy')
ada = np.load('./vote/output_adaboost.npy')
pro = np.load('./vote/protonet.npy')

ans = []

for i in range(0, len(svm)):
    ans.append(vote(svm[i], ada[i], pro[i]))

test_labels = getTestLabel()
test_ans = []
for i in test_labels:
    test_ans.append(int(i))

total = 0
correct = 0
for i in range(0, len(ans)):
    total += 1
    if ans[i] == test_ans[i]:
        correct += 1

print(correct / total)


np.save("./vote/ans.npy",ans)

