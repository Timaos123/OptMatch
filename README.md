# 1.基于线性规划解决两两配对问题

大致原理见知乎：[【模型工程】线性规划解决两两配对问题](https://zhuanlan.zhihu.com/p/348554373)

## 1.1.读取数据


```python
import  pandas as pd

teacherDf=pd.read_csv("data/teacher_info.csv")
teacherDf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>teacher_id</th>
      <th>major</th>
      <th>grade</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>t1</td>
      <td>电包</td>
      <td>3</td>
      <td>1班</td>
    </tr>
    <tr>
      <th>1</th>
      <td>t2</td>
      <td>电包</td>
      <td>3</td>
      <td>2班</td>
    </tr>
    <tr>
      <th>2</th>
      <td>t3</td>
      <td>电包</td>
      <td>3</td>
      <td>3班</td>
    </tr>
    <tr>
      <th>3</th>
      <td>t4</td>
      <td>电商</td>
      <td>3</td>
      <td>4班</td>
    </tr>
    <tr>
      <th>4</th>
      <td>t5</td>
      <td>电商</td>
      <td>4</td>
      <td>5班</td>
    </tr>
  </tbody>
</table>
</div>




```python
studentDf=pd.read_csv("data/student_info.csv")
studentDf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>student_id</th>
      <th>major</th>
      <th>grade</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s1</td>
      <td>电包</td>
      <td>1</td>
      <td>1班</td>
    </tr>
    <tr>
      <th>1</th>
      <td>s2</td>
      <td>电包</td>
      <td>1</td>
      <td>2班</td>
    </tr>
    <tr>
      <th>2</th>
      <td>s3</td>
      <td>电包</td>
      <td>3</td>
      <td>3班</td>
    </tr>
    <tr>
      <th>3</th>
      <td>s4</td>
      <td>电商</td>
      <td>3</td>
      <td>4班</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s5</td>
      <td>电商</td>
      <td>2</td>
      <td>5班</td>
    </tr>
  </tbody>
</table>
</div>



## 1.2.构建带权邻接矩阵


```python
import numpy as np
from OptMatch import matchLinearAdj

studentInfoList=studentDf.values.tolist()
teacherInfoList=teacherDf.values.tolist()
adjMat=np.matrix([[abs(teacherInfoList[teacherI][2]-studentInfoList[studentI][2]) for teacherI in range(len(teacherInfoList))] for studentI in range(len(studentInfoList))])
rx,result,status=matchLinearAdj(adjMat)
```

## 1.3.输出结果


```python
pairList=[]
for xItem in rx:
    rowI=int(xItem.split("_")[1])
    colI=int(xItem.split("_")[2])
    pairList.append([teacherInfoList[colI][0],studentInfoList[rowI][0]])
pairList=[eval(pairItem) for pairItem in list(set([str(pairItem) for pairItem in pairList]))]
print("决策变量（前5条）：\n",pairList[:5])
print("最终结果：",result)
print("解答状态：",status)
```

    决策变量（前5条）：
     [['t3', 's15'], ['t8', 's20'], ['t9', 's10'], ['t6', 's19'], ['t19', 's7']]
    最终结果： 44.0
    解答状态： 1
    

# 2.多人组匹配问题

## 2.1.构建带权邻接矩阵

### 2.1.1.人员名单构建


```python
studentNameList=[row[0] for row in studentInfoList]
teacherNameList=[row[0] for row in teacherInfoList]
peopleNameList=studentNameList+teacherNameList
peopleInfoList=[row for row in studentInfoList]+[row for row in teacherInfoList]
```

### 2.1.2.关联权值构建（需要依据自己的业务逻辑进行构建），并过滤掉无效关联（权值为0的关联）


```python
def matchOK(peopleItem1,peopleItem2):
    # 条件过滤

    # 两人不同约束
    if peopleItem1[0]==peopleItem2[0]:
        return 0

    # 二者都是学生
    if peopleItem1[0] in studentNameList and peopleItem2[0] in studentNameList:
        if peopleItem1[1]==peopleItem2[1]:# 学生同专业约束
            return 1
    
    # 二者都是教师
    if peopleItem1[0] in teacherNameList and peopleItem2[0] in teacherNameList:
        return 1
    
    # peopleItem1是学生，peopleItem2是教师
    if peopleItem1[0] in studentNameList and peopleItem2[0] in teacherNameList:# 教师与考生尽可能同年级
        return 4-abs(peopleItem2[2]-peopleItem1[2])

    # 筛除peopleItem1是教师，peopleItem2是学生的情况
    if peopleItem1[0] in teacherNameList and peopleItem2[0] in studentNameList:
        return 0

    return 0
```


```python
pairList=[[peopleItem1[0],peopleItem2[0],matchOK(peopleItem1,peopleItem2)] for peopleItem1 in peopleInfoList for peopleItem2 in peopleInfoList]    
pairList=[row for row in pairList if row[2]>0]
myAdjMat=np.zeros([len(peopleNameList),len(peopleNameList)])
for pairItem in pairList:
    myAdjMat[peopleNameList.index(pairItem[0]),peopleNameList.index(pairItem[1])]=pairItem[2]
```

## 2.2.基于极大团进行组匹配


```python
from OptMatch import matchCliqueAdj

studentMaxNum=10
teaMinNum=3
meetingList,remainStuList,remainTeaList=matchCliqueAdj(myAdjMat,studentNameList,studentMaxNum,teacherNameList,teaMinNum,teaMaxTime=1)
```

## 2.3.输出结果


```python
for meetingItem in meetingList:
    print(meetingItem)
```

    [['t3', 't17', 't2'], ['s11', 's16', 's12', 's23', 's18', 's22', 's10', 's4', 's6', 's5']]
    [['t1', 't4', 't18'], ['s3', 's9', 's21', 's15', 's1', 's20', 's13', 's7', 's14', 's19']]
    [['t20', 't16', 't23'], ['s17', 's24']]
    [['t6', 't7', 't5'], ['s2', 's8']]
    

其中，每一行list中，左侧为该组中的教师名单，右侧为该组中的学生名单
