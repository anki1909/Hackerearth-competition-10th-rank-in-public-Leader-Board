import numpy as np
import pandas as pd
from sklearn import ensemble, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans





def break_down_skills(x):
    #print x
    l= x.tolist()[0]
    try:
        total_skills = l.split('|')
    except:
        return
    length =  len(total_skills)
    a = np.zeros(26)
       
    dct = {'Befunge':0, 'C':1, 'C#':2, 'C++':3, 'C++ (g++ 4.8.1)':4, 'Clojure':5, 'Go':6,'Haskell':7, 
    'Java':8, 'Java (openjdk 1.7.0_09)':9, 'JavaScript':10,
       'JavaScript(Node.js)':11, 'JavaScript(Rhino)':12, 'Lisp':13, 'Objective-C':14,
       'PHP':15, 'Pascal':16, 'Perl':17, 'Python':18, 'Python 3':19, 'R(RScript)':20, 'Ruby':21,
       'Rust':22, 'Scala':23, 'Text':24, 'Whenever':25 }

    for i in range(length):
        #print i,length
        a[dct[total_skills[i]]] = a[dct[total_skills[i]]] + 1
    
    #print a[0]
    return pd.Series({'R0':a[0],'R1':a[1],'R2':a[2],'R3':a[3],'R4':a[4],'R5':a[5],'R6':a[6],'R7':a[7],'R8':a[8],'R9':a[9],'R10':a[10],
    'R11':a[11],'R12':a[12],'R13':a[13],'R14':a[14],'R15':a[15],'R16':a[16],'R17':a[17],'R18':a[18],'R19':a[19],'R20':a[20],'R21':a[21],
    'R22':a[22],'R23':a[23],'R24':a[24],'R25':a[25],'R26':length})
    

    
def all_tags_convert(x):
    dict_tag = {'Ad-Hoc':0, 'Ad-hoc':0, 'Algorithms':1, 'BFS':2, 'BIT':3, 'Basic Programming':4,
       'Basic-Programming':4, 'Bellman Ford':5, 'Binary Search':6,
       'Binary Search Tree':7, 'Binary Tree':7, 'Bipartite Graph':8,
       'Bit manipulation':9, 'Bitmask':10, 'Brute Force':11, 'Combinatorics':12,
       'Completed':13, 'DFS':14, 'Data Structures':15, 'Data-Structures':15,
       'Dijkstra':16, 'Disjoint Set':17, 'Divide And Conquer':18,
       'Dynamic Programming':19, 'Easy-medium':20, 'Expectation':21,
       'Extended Euclid':22, 'FFT':23, 'Fenwick Tree':24, 'Flow':25, 'Floyd Warshall':26,
       'GCD':27, 'Game Theory':28, 'Geometry':29, 'Graph Theory':28, 'Greedy':30,
       'HashMap':31, 'Hashing':32, 'Heap':33, 'Heavy light decomposition':34,
       'Implementation':35, 'KMP':36, 'Kruskal':36, 'Line-sweep':37, 'Maps':38,
       'Matching':39, 'Math':40, 'Matrix Exponentiation':41, 'Memoization':42,
       'Minimum Spanning Tree':43, 'Modular arithmetic':44,
       'Modular exponentiation':45, 'Number Theory':46, 'Prime Factorization':47,
       'Priority Queue':48, 'Priority-Queue':48, 'Probability':49, 'Queue':50,
       'Recursion':51,  'Segment Trees':52, 'Set':53,
       'Shortest-path':54, 'Sieve':55, 'Simple-math':56, 'Simulation':57, 'Sorting':58,
       'Sqrt-Decomposition':59, 'Stack':60, 'String Algorithms':61,
       'String-Manipulation':62, 'Suffix Arrays':63, 'Trees':64, 'Trie':65,
       'Two-pointer':66, 'Very Easy':67, 'adhoc':0, 'cake-walk':68}
    

    x = x.tolist()
    #print x
    a = list(np.zeros(69))
    number_of_tag =0
    for i in range(5):
        try:
            if np.isnan(x[i]):
                continue
        except:
            number_of_tag = number_of_tag +1
            try:
                a[dict_tag[x[i]]] = a[dict_tag[x[i]]] +1
            except:
                continue
    a.append(number_of_tag)
    #print a[0]
    
    return pd.Series(a)    

   


if __name__ == '__main__':
    
    train_submission  = pd.read_csv('../input3/will_bill_solve_it/train/submissions.csv')#, nrows =1000)
    train_users  = pd.read_csv('../input3/will_bill_solve_it/train/users.csv')
    train_problems = pd.read_csv('../input3/will_bill_solve_it/train/problems.csv')
    
    #freq =  pd.read_csv('../input/sortedfreq.csv')
    
    

    
    c = train_submission.groupby(['user_id', 'problem_id', 'solved_status']).agg('count').reset_index()
    train_submission = c[['user_id', 'problem_id', 'solved_status']] 
    
    
    test_submission  = pd.read_csv('../input3/will_bill_solve_it/test/test.csv')#, nrows =100)
    test_users  = pd.read_csv('../input3/will_bill_solve_it/test/users.csv')
    test_problems = pd.read_csv('../input3/will_bill_solve_it/test/problems.csv')
    
    train_problems.rename(columns ={'solved_count':'solved_count_problems'},inplace = True )
    test_problems.rename(columns ={'solved_count':'solved_count_problems'},inplace = True )
    train_users.rename(columns ={'solved_count':'solved_count_users'},inplace = True )
    test_users.rename(columns ={'solved_count':'solved_count_users'},inplace = True )
    
   
   
    

    tags_convert_train = train_problems[['tag1','tag2','tag3','tag4','tag5']].apply(all_tags_convert ,axis =1)
    tags_convert_test = test_problems[['tag1','tag2','tag3','tag4','tag5']].apply(all_tags_convert ,axis =1)
    c = np.sum(tags_convert_train,axis = 0)
    c = c.reset_index()
    c.rename(columns={0:'freq'},inplace = True)
    
    train_skills = train_users[['skills']].apply(break_down_skills, axis =1)
    test_skills = test_users[['skills']].apply(break_down_skills, axis =1)
    
    """
    d = np.sum(train_skills,axis = 0)
    d = d.reset_index()
    d.rename(columns={0:'freq'},inplace = True)
    columns_freq = d.loc[d.freq > 100, 'index'].values
    
    train_skills = train_skills[columns_freq]
    test_skills = test_skills[columns_freq]
    """
    train_users  = train_users.drop('skills',axis = 1)
    test_users = test_users.drop('skills', axis =1)
    
    test_users = pd.concat((test_users,  test_skills), axis=1)
    train_users = pd.concat((train_users, train_skills), axis=1)    
    train = train_submission.merge(train_users , on = 'user_id' , how = 'left')
    test = test_submission.merge(test_users , on = 'user_id' , how = 'left')
    train.loc[train['solved_status'] == 'SO' , 'solved_status'] = 1
    train.loc[train['solved_status'] == 'AT' , 'solved_status'] = 0
    train.loc[train['solved_status'] == 'UK' , 'solved_status'] = 1
   
    y = train['solved_status'].values
    Ids = test['Id'].values
    
    train = train.drop('solved_status', axis = 1)
    test = test.drop('Id',axis = 1)
    
    
    
    columns_freq = c.loc[c.freq > 10, 'index'].values
    
    tags_convert_train1 = tags_convert_train[columns_freq]
    tags_convert_test1 = tags_convert_test[columns_freq]
    
    train_problems2 = pd.concat((train_problems,tags_convert_train1), axis =1)
    test_problems2 = pd.concat((test_problems,tags_convert_test1), axis =1)
        
        
    
    
    
    train1 = train.merge(train_problems2 ,on = 'problem_id', how = 'left')
    test1 = test.merge(test_problems2 ,on = 'problem_id', how = 'left')
        
   
        
    text_columns = []
    for f in train1.columns:
        if train1[f].dtype== 'object':
                #print f
            if f != 'll':   
                print f
                text_columns.append(f)            
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(train1[f].values) + list(test1[f].values))
                train1[f] = lbl.transform(list(train1[f].values))
                test1[f] = lbl.transform(list(test1[f].values))    
                
    df_t = pd.concat((train1[['solved_count_users','user_id', 'level']] , test1[['solved_count_users','user_id', 'level']] ) , axis =0 )
    
    #u_l_acc_count = df_t[['accuracy','user_id', 'level']].groupby(['user_id','level']).agg('count').reset_index()
    
    u_l_acc_mean = df_t[['solved_count_users','user_id', 'level']].groupby(['user_id','level']).agg('mean').reset_index()
    u_l_acc_mean.rename(columns = {'solved_count_users':'levels_count_by_users_solved'}, inplace = True)    
    train1 = train1.merge(u_l_acc_mean , on = ['level','user_id'] , how = 'left')
    test1 = test1.merge(u_l_acc_mean , on = ['level','user_id' ] , how = 'left')
    
    
    
    
    
    
    
    
    
    df_t = pd.concat((train1[['error_count','user_id', 'level']] , test1[['error_count','user_id', 'level']] ) , axis =0 )
    u_l_error_mean = df_t[['error_count','user_id', 'level']].groupby(['user_id','level']).agg('mean').reset_index()
    u_l_error_mean.rename(columns = {'error_count':'levels_count_by_users_error'}, inplace = True)    
    train1 = train1.merge(u_l_error_mean , on = ['level','user_id'] , how = 'left')
    test1 = test1.merge(u_l_error_mean , on = ['level','user_id' ] , how = 'left')
   
    """
    c = train1[['accuracy','level']].groupby(['level']).agg('mean').reset_index()
    c.rename(columns = {'accuracy':'new_cal_acc'}, inplace = True)    
    
    
    
    train1 = train1.merge(c , on = ['level'] , how = 'left')
    test1 = test1.merge(c , on = ['level' ] , how = 'left')
    """
    
    
    train1 = train1.drop(['tag1','tag2','tag3','tag4','tag5','user_id'],axis = 1)
    test1 = test1.drop(['tag1','tag2','tag3','tag4','tag5','user_id'],axis = 1)
    print 'start',train1.shape
    train1 = train1.replace(np.nan , -1)
    test1 = test1.replace(np.nan , -1)
    
    
    
    scl = StandardScaler()
    X = scl.fit_transform(train1)
    X_test = scl.transform(test1)
    #reduced_data = PCA(n_components=2).fit_transform(X)
    kmean = KMeans(n_clusters=  10,max_iter =3000, random_state=2)
    cluster_id_train = kmean.fit_predict(X)
    cluster_id_test = kmean.predict(X_test)
    
    train1['cluster_id']  = cluster_id_train
    test1['cluster_id']  = cluster_id_test
   
    print 'over'
   
    #c = train1[['solved_count_users', 'attempts', 'user_type','level','accuracy','solved_count_problems', 'error_count' , 'rating']]
    gbm1 = ensemble.GradientBoostingClassifier(random_state = 42, learning_rate = 0.08,subsample = 1,  min_samples_split = 30 , max_features = 'sqrt' ,  n_estimators = 500 , max_depth =  6)
    #gbm1 = XGBClassifier(max_depth=6, learning_rate=0.08,objective= 'binary:logistic', n_estimators= 500, subsample= 1, colsample_bytree= 0.65, seed=0) 
    gbm1.fit(train1,y)
    pred1 = gbm1.predict_proba(test1)[:,1]
        
    pred3 = np.zeros(len(pred1))
    pred3[pred1>0.56] = 1
    pred3 = pred3.astype(int)
    submission = pd.DataFrame({'Id':Ids,'solved_status':pred3})
    submission['solved_status'] = submission['solved_status'].astype(int)
    submission.to_csv('finalsubmission_csv_use_this_for_evaluation.csv',index = False)
