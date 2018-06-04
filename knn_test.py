import numpy as np    
from sklearn import neighbors   
  
knn = neighbors.KNeighborsClassifier(n_neighbors=6,weights='distance') #取得knn分类器 
knn2 = neighbors.KNeighborsClassifier(n_neighbors=3,weights='distance')
#n_neighbors每次对比的时候跟多少个点比，我们应该取那个最小的值，或者64？个人偏向64
#weights='distance'这个应该就是按照欧式距离算点之间的距离
#p default=2 欧式距离
#n_jobs 多线程

data_vector = [[3,104],[2,100],[1,81],[101,10],[99,5],[98,2],[48,25],[38,29],[42,22]]
labels_vector = [1,1,1,2,2,2,3,3,3]
data = np.array(data_vector)
labels = np.array(labels_vector)
# data2 = np.array([[1,102],[49,30]])
# labels2 = np.array([1,3])
knn.fit(data, labels) #导入数据进行训练  
# knn.fit(data2, labels2) 
knn2.fit(data, labels)

#直接预测是那个label
print(knn.predict([[100,8], [11,20]])) 

#预测每个label的可能性
print(knn.predict_proba([[101,9],[11,20]]))

#显示谁离得最近，甚至还可以显示距离
test_place = [[3,103], [11,20]]
print(knn.kneighbors(test_place, return_distance=False))

#显示每个点所对应的那些neighbors
array_test = knn.kneighbors_graph(test_place)
print(array_test.toarray())

test_data = [[11,24], [48,23],[88,16],[4,50]]
test_label = [1,3,2,1]
result1 = knn.score(test_data, test_label)
result2 = knn2.score(test_data, test_label)
#这里可以看出3并不一定是最好的选择，最后可能会涉及一个调参
print(knn.predict(test_data))
print(knn2.predict(test_data))
print(result1)
print(result2)