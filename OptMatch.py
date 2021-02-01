import pulp
import numpy as np
from pulp.utilities import value
import  pandas as pd

def matchLinearAdj(adjMat,rowMax=1,colMax=3,fullX=False):
    '''
    输入参数：
    adjMat:带权邻接矩阵
    rowMax:学生参与的最多场次
    colMax:教师参与的最多场次
    fullX:是否要求全员参与
    =====================
    输出结果：
    xvDict:决策变量结果{"x_i_j":0|1,......}（i,j为决策变量的行和列，结果为1或0）
    result:规划结果
    status:规划解状态（{0: 'Not Solved', 1: 'Optimal', -1: 'Infeasible', -2: 'Unbounded', -3: 'Undefined'}）
    '''
    # 目标函数
    obj=pulp.LpProblem("myPro", pulp.LpMaximize)
    # 自变量
    X=[[pulp.LpVariable("X_{}_{}".format(studentI,teacherI),lowBound=0,upBound=1,cat=pulp.LpInteger) for teacherI in range(adjMat.shape[0])] for studentI in range(adjMat.shape[1])]

    # 目标函数构建
    obj+=sum([adjMat[rowI,colI]*X[rowI][colI] for rowI in range(adjMat.shape[0]) for colI in range(adjMat.shape[1])])

    # 约束构建
    if fullX==False:
        for rowI in range(adjMat.shape[0]):
            obj+=sum([X[rowI][colI] for colI in range(adjMat.shape[1])])<=rowMax
    else:
        for rowI in range(adjMat.shape[0]):
            obj+=sum([X[rowI][colI] for colI in range(adjMat.shape[1])])==rowMax

    for colI in range(adjMat.shape[1]):
        obj+=sum([X[rowI][colI] for rowI in range(adjMat.shape[0])])<=colMax

    status=obj.solve()
    result=pulp.value(obj.objective)
    xvDict=dict((rx.name,rx.varValue) for rx in obj.variables())

    return xvDict,result,status

def packagePairs(xMat,packageSize=4):
    pass

if __name__=="__main__":
    print("1.双人匹配：")
    print("1.1.读取数据：")
    teacherDf=pd.read_csv("data/teacher_info.csv")
    studentDf=pd.read_csv("data/student_info.csv")
    studentInfoList=studentDf.values.tolist()
    teacherInfoList=teacherDf.values.tolist()
    adjMat=np.matrix([[abs(teacherInfoList[teacherI][2]-studentInfoList[studentI][2]) for teacherI in range(len(teacherInfoList))] for studentI in range(len(studentInfoList))])
    rx,result,status=matchLinearAdj(adjMat)
    print("1.2.输出结果")
    print("决策变量：\n",rx)
    print("最终结果：",result)
    print("解答状态：",status)
