import pulp
import numpy as np
from pulp.utilities import value
import  pandas as pd
import networkx as nx
import numpy as np

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

def matchCliqueAdj(myAdjMat,studentNameList,studentMaxNum,teacherNameList,teaMinNum,teaMaxTime=3,meetingMaybeNum=25):
    '''
    输入参数:
    myAdjMat:带权邻接矩阵
    studentNameList:学生名列表
    studentMaxNum:学生人数
    teacherNameList:教师名列表
    teaMinNum:教师最少人数
    teaMaxTime:教师能参与的最高场次
    meetingMaybeNum:可能的会议室数目
    ===============================
    输出参数：
    meetingList：组列表[教师列表,学生列表]
    remainStuList：剩余学生名单
    remainTeaList：剩余教师名单
    '''
    peopleNameList=studentNameList+teacherNameList
    weightedAdjMat=myAdjMat
    zeroOneAdjMat=weightedAdjMat>0
    pairList=[(peopleNameList[hI],peopleNameList[tI]) for hI in range(zeroOneAdjMat.shape[0]) for tI in range(zeroOneAdjMat.shape[1]) if zeroOneAdjMat[hI,tI]==1]
    myGraph=nx.Graph()
    myGraph.add_edges_from(pairList)
    cliqueGenerator=nx.find_cliques(myGraph)
    cliqueList=[]
    for cliqueItem in cliqueGenerator:
        if len(cliqueList)<meetingMaybeNum:
            if 0<len([staffItem for staffItem in cliqueItem if staffItem in studentNameList])<len(cliqueItem):
                cliqueList.append(cliqueItem)
        else:
            break
    if len(cliqueList)>0:
        TFMax=max([len([staffItem for staffItem in cliqueItem if staffItem in studentNameList])>=studentMaxNum 
                    and
                    len([staffItem for staffItem in cliqueItem if staffItem in teacherNameList])>=teaMinNum 
                    for cliqueItem in cliqueList])
        if TFMax==True:
            cliqueList=[cliqueItem 
                            for cliqueItem in cliqueList 
                            if len([staffItem for staffItem in cliqueItem if staffItem in studentNameList])>=studentMaxNum
                            and len([staffItem for staffItem in cliqueItem if staffItem in teacherNameList])>=teaMinNum]

    #面试分组
    stuteaStrList=[str(pairItem) for pairItem in pairList]
    meetingList=[]
    totalJointeaList=[]
    totalJoinstuList=[]
    cliqueList=[cliqueItem for cliqueItem in cliqueList if len([staffItem for staffItem in cliqueItem if staffItem in teacherNameList])>=teaMinNum]
    cliqueList=sorted(cliqueList,key=lambda row:[len([staffItem for staffItem in row if staffItem in teacherNameList]),\
                                                    -len([staffItem for staffItem in row if staffItem not in teacherNameList])])
    if len(cliqueList)>0:
        TFMax=max([len([staffItem for staffItem in cliqueItem if staffItem in studentNameList])>=studentMaxNum 
                    and
                    len([staffItem for staffItem in cliqueItem if staffItem in teacherNameList])>=teaMinNum 
                    for cliqueItem in cliqueList])
        if TFMax==True:
            cliqueList=[cliqueItem 
                            for cliqueItem in cliqueList 
                            if len([staffItem for staffItem in cliqueItem if staffItem in studentNameList])>=studentMaxNum
                            and len([staffItem for staffItem in cliqueItem if staffItem in teacherNameList])>=teaMinNum]

    while len(cliqueList)>0:
        cliStuList=[staffItem for staffItem in cliqueList[0] if staffItem in studentNameList]
        cliStuWeightList=[np.mean(weightedAdjMat[peopleNameList.index(stuItem),:]) for stuItem in cliStuList]
        cliStuList=sorted(cliStuList,key=lambda clistuItem:cliStuWeightList[cliStuList.index(clistuItem)],reverse=True)

        cliTeaList=[staffItem for staffItem in cliqueList[0] if staffItem in teacherNameList]
        cliTeaWeightList=[np.mean(weightedAdjMat[peopleNameList.index(teaItem),:]) for teaItem in cliTeaList]
        cliTeaList=sorted(cliTeaList,key=lambda cliteaItem:cliTeaWeightList[cliTeaList.index(cliteaItem)],reverse=True)
        
        if len(cliStuList)==0 or len(cliTeaList)==0:
            break
        while len(cliStuList)>0 and len(cliTeaList)>0:
            if TFMax==True:#存在评委和候选人人数正好满足要求的团
                meetingTeaList=[]
                cliteaI=0
                #装填教师
                while cliteaI<len(cliTeaList) and len(cliTeaList)>0 and len(meetingTeaList)<teaMinNum:#团教师未遍历完成 且 团教师数量大于0 且 会议室内的教师数量小于最低要求
                    if cliTeaList[cliteaI] not in meetingTeaList:
                        if totalJointeaList.count(cliTeaList[cliteaI])>=teaMaxTime:
                            cliTeaList.pop(cliteaI)
                            cliteaI-=1
                        else:
                            meetingTeaList.append(cliTeaList[cliteaI])#增加会议室评委
                            totalJointeaList.append(cliTeaList[cliteaI])#评委参与历史构建
                    cliteaI+=1
                #装填候选人
                meetingStuList=[]
                while len(cliStuList)>0 and len(meetingStuList)<studentMaxNum:
                    if cliStuList[0] not in totalJoinstuList:
                        meetingStuList.append(cliStuList[0])#增加会议室候选人
                        totalJoinstuList.append(cliStuList[0])#候选人参与历史构建
                    cliStuList.pop(0)

                if len(meetingStuList)>=studentMaxNum:
                    meetingList.append([meetingTeaList,meetingStuList])
                else:
                    totalJointeaList=totalJointeaList[:-len(meetingTeaList)]
                    totalJoinstuList=totalJoinstuList[:-len(meetingStuList)]

            else:#存在候选人人数较少的团
                meetingTeaList=[]
                cliteaI=0
                while cliteaI<len(cliTeaList) and len(cliTeaList)>0 and len(meetingTeaList)<teaMinNum:#团评委未遍历完成 且 团评委数量大于0 且 会议室内的评委小于最低要求
                    if cliTeaList[cliteaI] not in meetingTeaList:
                        if totalJointeaList.count(cliTeaList[cliteaI])>=teaMaxTime:
                            cliTeaList.pop(cliteaI)
                            cliteaI-=1
                        else:
                            meetingTeaList.append(cliTeaList[cliteaI])#增加会议室评委
                            totalJointeaList.append(cliTeaList[cliteaI])#评委参与历史构建
                    cliteaI+=1
                #装填候选人
                meetingStuList=[]
                while len(cliStuList)>0 and len(meetingStuList)<studentMaxNum:
                    if cliStuList[0] not in totalJoinstuList:
                        meetingStuList.append(cliStuList[0])#增加会议室候选人
                        totalJoinstuList.append(cliStuList[0])#候选人参与历史构建
                    cliStuList.pop(0)
                #填充会议人员
                if len(meetingTeaList)>=teaMinNum and len(meetingStuList)<=studentMaxNum:
                    meetingList.append([meetingTeaList,meetingStuList])
                else:
                    totalJointeaList=totalJointeaList[:-len(meetingTeaList)]
                    totalJoinstuList=totalJoinstuList[:-len(meetingStuList)]

        #剩余团处理
        cliStaffList=list(set([stuItem for stuItem in studentNameList if stuItem not in totalJoinstuList]+
                                [teaItem for teaItem in teacherNameList if totalJointeaList.count(teaItem)<teaMaxTime]))
        restructPairList=[[staffItem1,staffItem2] 
                            for staffItem1 in cliStaffList 
                            for staffItem2 in cliStaffList 
                            if str((staffItem1,staffItem2)) in stuteaStrList or str((staffItem2,staffItem1)) in stuteaStrList]
        restructGraph=nx.Graph()
        restructGraph.add_edges_from(restructPairList)
        restructCliqueGenerator=nx.find_cliques(restructGraph)
        
        cliqueList=[]
        for cliqueItem in restructCliqueGenerator:
            if len(cliqueList)<meetingMaybeNum:
                if 0<len([staffItem for staffItem in cliqueItem if staffItem in studentNameList])<len(cliqueItem):
                    cliqueList.append(cliqueItem)
            else:
                break
        if len(cliqueList)>0:
            TFMax=max([len([staffItem for staffItem in cliqueItem if staffItem in studentNameList])>=studentMaxNum 
                        and
                        len([staffItem for staffItem in cliqueItem if staffItem in teacherNameList])>=teaMinNum 
                        for cliqueItem in cliqueList])
            if TFMax==True:
                cliqueList=[cliqueItem 
                            for cliqueItem in cliqueList 
                            if len([staffItem for staffItem in cliqueItem if staffItem in studentNameList])>=studentMaxNum
                                and len([staffItem for staffItem in cliqueItem if staffItem in teacherNameList])>=teaMinNum]
        if len(cliqueList)>0:
            cliqueList=sorted(cliqueList,key=lambda row:[len([staffItem for staffItem in row if staffItem in teacherNameList]),\
                                                        -len([staffItem for staffItem in row if staffItem not in teacherNameList])])

    meetingList=[meetingItem for meetingItem in meetingList if len(meetingItem[0])>=teaMinNum and len(meetingItem[1])>0]
    totalJoinStuList=[stuItem for meetingItem in meetingList for stuItem in meetingItem[1] if stuItem in studentNameList]
    remainStuList=[canItem for canItem in studentNameList if canItem not in totalJoinStuList]
    totalJoinTeaList=[teaItem for meetingItem in meetingList for teaItem in meetingItem[0] if teaItem in teacherNameList]
    remainTeaList=[teaItem for teaItem in teacherNameList if teaItem not in totalJoinTeaList]

    return meetingList,remainStuList,remainTeaList
    

if __name__=="__main__":
    print("1.双人匹配：")
    print("要求：")
    print("在安排一场考试地过程中，我们希望与考生同年级的教师尽可能地与考生分配到一起，同时：")
    print("·每个老师出场次数最多为3次")
    print("1.1.读取数据：")
    teacherDf=pd.read_csv("data/teacher_info.csv")
    studentDf=pd.read_csv("data/student_info.csv")
    studentInfoList=studentDf.values.tolist()
    teacherInfoList=teacherDf.values.tolist()

    print("1.2.构造带权邻接矩阵")
    adjMat=np.matrix([[abs(teacherInfoList[teacherI][2]-studentInfoList[studentI][2]) for teacherI in range(len(teacherInfoList))] for studentI in range(len(studentInfoList))])
    rx,result,status=matchLinearAdj(adjMat)
    print("1.3.输出结果")
    pairList=[]
    for xItem in rx:
        rowI=int(xItem.split("_")[1])
        colI=int(xItem.split("_")[2])
        pairList.append([teacherInfoList[colI][0],studentInfoList[rowI][0]])
    pairList=[eval(pairItem) for pairItem in list(set([str(pairItem) for pairItem in pairList]))]
    print("决策变量：\n",pairList)
    print("最终结果：",result)
    print("解答状态：",status)

    print("2.组匹配：")
    print("2.1.构造带权邻接边")
    print("要求：")
    print("在一个分配考场（或者称为：组）的案例中，我们希望能够适当的把教师与学生分配到相应考场中，同时：")
    print("·每个老师出场次数最多为3次")
    print("·每个组内最多10个学生")
    print("·每个组内最少3个教师")
    print("·每个组内最多3个教师")
    print("·每个组内的学生尽可能属于同一专业")
    print("·与考生同年级的教师尽可能地与考生分配到一起")

    studentNameList=[row[0] for row in studentInfoList]
    teacherNameList=[row[0] for row in teacherInfoList]
    peopleNameList=studentNameList+teacherNameList
    peopleInfoList=[row for row in studentInfoList]+[row for row in teacherInfoList]
    
    #构建规则筛选函数
    def matchOK(peopleItem1,peopleItem2):
        # 条件过滤
        # 二者都是学生

        if peopleItem1[0]==peopleItem2[0]:
            return 0

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


    pairList=[[peopleItem1[0],peopleItem2[0],matchOK(peopleItem1,peopleItem2)] for peopleItem1 in peopleInfoList for peopleItem2 in peopleInfoList]    
    pairList=[row for row in pairList if row[2]>0]
    myAdjMat=np.zeros([len(peopleNameList),len(peopleNameList)])
    for pairItem in pairList:
        myAdjMat[peopleNameList.index(pairItem[0]),peopleNameList.index(pairItem[1])]=pairItem[2]

    studentMaxNum=15
    teaMinNum=1
    meetingList,remainStuList,remainTeaList=matchCliqueAdj(myAdjMat,studentNameList,studentMaxNum,teacherNameList,teaMinNum)

    for meetingItem in meetingList:
        print(meetingItem)
