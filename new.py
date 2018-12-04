import pandas as pd
import math
import random
from sklearn.metrics import confusion_matrix
from datetime import datetime
from random import seed
from random import randrange
from csv import reader



from numpy import array
from sklearn.model_selection import KFold




def merge_sort(alist, start, end):
    
    if(end-start > 1):
        mid = (start + end)//2
        merge_sort(alist, start, mid)
        merge_sort(alist, mid, end)
        merge_list(alist, start, mid, end)
 
def merge_list(alist, start, mid, end):
    left=alist[start:mid]
    right=alist[mid:end]
    k=start
    i=0
    j=0
    while (start + i < mid and mid + j < end):
        if (left[i].distance <= right[j].distance):
            alist[k]= left[i]
            i = i+1
        else:
            alist[k]=right[j]
            j = j + 1
        k = k + 1
    if(start + i < mid):
        while(k < end):
            alist[k]=left[i]
            i = i+1
            k = k+1
    else:
        while(k < end):
            alist[k]=right[j]
            j =j+1
            k =k+1

class point:
    
    def __init__(self,points,x):
        self.p=[]
        for i in range(0,8,1):
        
            self.p.append(points[i])
        
        self.o=points[8]
        self.distance=0
        
        
              
              
              
        for i in range(0,8,1):
            self.distance=self.distance+ (self.p[i]-x[i])**2

        

        
        

        
        

x=pd.read_csv("dataset_new_cleaned.csv",header=None)
x0=x[0].tolist()
x1=x[1].tolist()
x2=x[2].tolist()
x3=x[3].tolist()
x4=x[4].tolist()
x5=x[5].tolist()
x6=x[6].tolist()
x7=x[7].tolist()
x8=x[8].tolist()


for i in range(1,len(x0),1):
    x0[i]=int(x0[i])
for i in range(1,len(x1),1):
    x1[i]=int(x1[i])
for i in range(1,len(x2),1):
    x2[i]=int(x2[i])
for i in range(1,len(x3),1):
    x3[i]=int(x3[i])
for i in range(1,len(x4),1):
    x4[i]=int(x4[i])
for i in range(1,len(x5),1):
    x5[i]=int(x5[i])
for i in range(1,len(x6),1):
    x6[i]=int(x6[i])
for i in range(1,len(x7),1):
    x7[i]=int(x7[i])
for i in range(1,len(x8),1):
    x8[i]=int(x8[i])


points=[]
accuracy_for_knn=[]
accuracy_for_decision_tree=[]

for i in range(1,len(x0),1):
    z=[]
    z.append(x0[i])
    z.append(x1[i])
    z.append(x2[i])
    z.append(x3[i])
    z.append(x4[i])
    z.append(x5[i])
    z.append(x6[i])
    z.append(x7[i])
    z.append(x8[i])
  
    
    points.append(z)

    
p=[]
random.shuffle(points)
random.shuffle(points)

for i in range(1,15000,1):
    p.append(points[i])


points=p





	
	              



def knn(training,testing,k_n):
 start_time = datetime.now()   

 result=[]


 y_predicted=[]
 y_actual=[]


 for v in testing:
  test=[]   
  for vv in range(0,8,1):
     test.append(v[vv])
    
  k=[]



  for i in training:
     p= point(i,test)
     k.append(p)


  merge_sort(k,0,len(k)-1)


        


  knn=k_n
  count=[]
  for i in range(0,knn,1):
    count.append(str(k[i].o))

 
  if(count.count("0")>count.count("1")):
    r=0
  else:
    r=1


  
    
  
  
  
  
 
  
  y_actual.append(v[8])
  y_predicted.append(r)
  if(v[8]==r):
     result.append("1")
  else:
     result.append("0")
  


                 
 

 

 print("confusion matrix:")
 print(confusion_matrix(y_actual, y_predicted))
 accuracy_for_knn.append((result.count("1")/len(result))*100)
 end_time = datetime.now()
 print('Duration: {}'.format(end_time - start_time))












 



def test_split(ind, value, dataset):
	ll=list()
	rr =list()
	for x_r in dataset:
		if x_r[ind] < value:
			ll.append(x_r)
		else:
			rr.append(x_r)
	return(ll, rr)
 

def gini_ind(groups, classes):
	
	n_instances=float(sum([len(group) for group in groups]))
	
	gini=0.0
	for group in groups:
		size=float(len(group))
		
		if(size==0):
			continue
		score=0.0
		
		for class_val in classes:
			p=[x_r[-1] for x_r in group].count(class_val)/size
			score += p * p
		
		gini += (1.0 - score) * (size / n_instances)
	return(gini)
 

def get_split(dataset):
	class_values=list(set(x_r[-1] for x_r in dataset))
	b_ind, b_value, b_score, b_groups = 999, 999, 999, None
	for ind in range(len(dataset[0])-1):
		for x_r in dataset:
			groups=test_split(ind, x_r[ind], dataset)
			gini=gini_ind(groups, class_values)
			if(gini < b_score):
				b_ind,b_value,b_score,b_groups = ind, x_r[ind], gini, groups
	return {'ind':b_ind, 'value':b_value, 'groups':b_groups}

def to_terminal(group):
	outcomes = [x_r[-1] for x_r in group]
	return(max(set(outcomes), key=outcomes.count))
 

def split(node,max_depth,min_size,depth):
	ll, rr = node['groups']
	del(node['groups'])
	
	if not ll or not rr:
		node['ll'] = node['rr'] = to_terminal(ll + rr)
		return
	
	if (depth >= max_depth):
		node['ll'], node['rr'] = to_terminal(ll), to_terminal(rr)
		return
	
	if (len(ll) <= min_size):
		node['ll'] = to_terminal(ll)
	else:
		node['ll'] = get_split(ll)
		split(node['ll'], max_depth, min_size, depth+1)
	
	if(len(rr) <= min_size):
		node['rr'] = to_terminal(rr)
	else:
		node['rr'] = get_split(rr)
		split(node['rr'], max_depth, min_size, depth+1)
 

def build_decision_tree(train,max_depth,min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return(root)
 

def predict(node, x_r):
	if(x_r[node['ind']] < node['value']):
		if(isinstance(node['ll'], dict)):
			return(predict(node['ll'], x_r))
		else:
			return(node['ll'])
	else:
		if(isinstance(node['rr'], dict)):
			return(predict(node['rr'], x_r))
		else:
			return(node['rr'])



def print_tree(node, depth=0):
	if(isinstance(node, dict)):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['ind']+1), node['value'])))
		print_tree(node['ll'], depth+1)
		print_tree(node['rr'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))

def decision_tree(train,test,max_depth,min_size):
        start_time = datetime.now()
        y_actual=[]
        y_predicted=[]
        correct=[]
        
        
        tree = build_decision_tree(train, max_depth, min_size)
        print_tree(tree)
        
        predictions = list()
        for x_r in test:
            prediction=predict(tree, x_r)
            predictions.append(prediction)
            if(x_r[-1]==prediction):
                    correct.append("1")
            else:
                correct.append("0")
            
            
            y_actual.append(x_r[-1])
            y_predicted.append(prediction)
            
        
        accuracy_for_decision_tree.append((correct.count("1")/len(correct))*100)
        
        print("confusion matrix:")
        print(confusion_matrix(y_actual, y_predicted))
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))
        return(predictions)
        
        
        
        
 

data = points

kfold = KFold(2, True, 1)

kfold.split(data)
t_no=1
for train, test in kfold.split(data):
        print("test number:")
        print(t_no)
        training=[]
        testing=[]
        for t in train:
            training.append(points[t])
        for t in test:
            testing.append(points[t])
        print("applying knn")
        knn(training,testing,20)
        print("applying decision tree")
        decision_tree(training,testing,5,5)
        t_no=t_no+1
        print("-----------------------------------")
        print("-----------------------------------")

print("accuracy of knn")
print(accuracy_for_knn)
print("accuracy of decision tree")
print(accuracy_for_decision_tree)








        



