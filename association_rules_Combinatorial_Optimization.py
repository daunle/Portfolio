import argparse

import pandas as pd
import numpy as np
# Association rule mining library
from mlxtend.frequent_patterns import apriori, association_rules
# Gurobi optimization library
import gurobipy as gp
from gurobipy import GRB
import json
import matplotlib.pyplot as plt
import seaborn as sns
# Evaluation & Visualization libraries
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.evaluate import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('-a','--alpha',dest='alpha',action='store',default=1,type = float)
parser.add_argument('-b','--beta',dest='beta',action='store',default=1,type = float)
parser.add_argument('-g','--gamma',dest='gamma',action='store',default=1,type = float)
parser.add_argument('-d','--delta',dest='delta',action='store',default=1,type = float)
parser.add_argument('-m1','--m1',dest='m1',action='store',default=1,type = float)
parser.add_argument('-m2','--m2',dest='m2',action='store',default=1,type = float)
parser.add_argument('-m3','--m3',dest='m3',action='store',default=1,type = float)
parser.add_argument('-p','--path',dest='path',action='store',default="C://Users//dlekd//OneDrive//바탕 화면//lab_lee//chemical//pcm//dara_product",type= str)
parser.add_argument('-c','--confidence',dest='confidence',action='store',default=0.03,type=float)
parser.add_argument('-s','--support',dest='support',action='store',default=0.03,type=float)
parser.add_argument('-ct','--ChoiceTrue',dest = 'ChoiceTrue',action='store_true')
parser.add_argument('-pt','--PathTrue',dest = 'PathTrue',action='store_true')
arg = parser.parse_args()

def coverage(rule, klass):
    """ Function used to calculate coverage of a rule"""
    return (sum(df.loc[df['Class'] == klass][rule].apply(lambda x: all(x), axis=1)))


def check_rules(num, rules):
    """ Function used to create matrix C"""
    lis = []
    for i in rules.index:
        lis.append(int(all(df[rules['itemsets'][i]].loc[num])))
    return lis


def check_rules_features():
    itemsets_unraveled = arules.explode('itemsets')['itemsets']
    rules_features = pd.DataFrame(data=0, index=df.columns[:-1], columns=arules.index)

    for i in itemsets_unraveled.iteritems():
        rules_features.loc[i[1]][i[0]] = 1

    return rules_features

if arg.PathTrue == False:
    se = pd.read_csv(arg.path+"//df_se.csv")
    ki = pd.read_csv(arg.path+"//df_ki.csv")
    se = se.set_index('Unnamed: 0',drop=True)
    ki = ki.set_index('Unnamed: 0',drop=True)
    df = pd.concat([se,ki],axis=1)
    df = df.fillna(0)
    li = ki.columns
    df = df.transpose()
    df['label'] = 0
    for v in li:
        df['label'][v] = 1
    df = df.reset_index()
    df = df.drop_duplicates(subset='index')
    df = df.drop(['index'],axis=1)
    df = df.reset_index()
    df = df.drop(['index'],axis=1)

    #df = pd.read_csv("C://Users//dlekd//OneDrive//바탕 화면//lab_lee//chemical//pcm//dara_product//tot_ki_killsm.csv",encoding = 'cp949')
    #df = d
    # Rename Class variable
    df.rename(columns={'label': 'Class'},inplace=True)

    # Remove columns not required
    #df.drop(columns = ['System3','TRIAGE'], inplace=True)


    frequent_itemsets = apriori(df, min_support=arg.support, use_colnames = True)
    arules = association_rules(frequent_itemsets, metric="confidence", min_threshold=arg.confidence)
    arules = arules[arules['consequents'] == frozenset({'Class'})]
    arules = arules[['antecedents','support','confidence','lift']]
    arules['size'] = arules['antecedents'].apply(lambda x: len(x))
    arules['Tcovered'] = arules['antecedents'].apply(lambda x: coverage(x,1))
    arules['Ncovered'] = arules['antecedents'].apply(lambda x: coverage(x,0))
    arules.rename(columns = {'antecedents': 'itemsets'}, inplace=True)
    arules.reset_index(drop=True,inplace=True)
    arules['itemsets'] = arules['itemsets'].apply(lambda x: list(x))

    #Matrix B
    rules_features = check_rules_features()
    rules_features = rules_features[rules_features.sum(axis=1)!=0]

    #Matrix C
    df_rules_data = [check_rules(x,arules) for x in df.index]
    df_rules = pd.DataFrame(data = df_rules_data, index = df.index, columns = arules.index)
    df_rules['Class'] = df['Class']

elif arg.PathTrue == True:
    with open("C://Users//Public//Desktop//df_rules.json", "r") as f:
        df_rules = json.load(f)
    with open("C://Users//Public//Desktop//rules_features.json", "r") as f:
        rules_features = json.load(f)
    with open("C://Users//Public//Desktop//df.json", "r") as f:
        df = json.load(f)

gp.disposeDefaultEnv()

J,K,I_P,I=len(rules_features.index),len(rules_features.columns),df['Class'].sum(),len(df.index)

c_P_df = df_rules.loc[df_rules['Class'] == 1, df_rules.columns !='Class'].reset_index(drop=True).transpose()

c_N_df = df_rules.loc[df_rules['Class'] == 0, df_rules.columns !='Class'].reset_index(drop=True).transpose()

bf = rules_features.reset_index(drop = True).transpose()

try:
# Create a new model
    A = gp.Model("mip1")

    # Create variables
    x = A.addVars(I, vtype=GRB.BINARY, name="x")
    # xneg = A.addVars(10, vtype=GRB.BINARY, name="xneg")
    y = A.addVars(J, vtype=GRB.BINARY, name="y")
    z = A.addVars(K, vtype=GRB.BINARY, name="z")

    # Set objective

    A.setObjective(arg.alpha*(1 / J) * gp.quicksum(y[j] for j in range(J)) +
                   arg.beta * (1 / K) * gp.quicksum(z[k] for k in range(K)) +
                   arg.gamma * (1 / (I - I_P)) * gp.quicksum(x[i] for i in range(I_P, I)) -
                   3*arg.gamma * (1 / I_P) * gp.quicksum(x[i] for i in range(I_P)),
                   GRB.MINIMIZE)

    # Add constraint: sigma{c_i_k+ * z_k} >= x_i
    for i in range(I_P):
        A.addConstr(gp.quicksum(c_P_df[i][k] * z[k] for k in range(K)) >= x[i], name='con1')

    # Add constraint: sigma{c_i_k- * z_k} <= M_1 * x_i
    for i in range(I_P, I):
        i1 = i - I_P
        A.addConstr(gp.quicksum(c_N_df[i1][k] * z[k] for k in range(K)) <= arg.m1 * x[i], name='con2')

    # Add constraint: sigma{b_j_k * z_k} <= M_2 * y_j
    if arg.ChoiceTrue == True:
        for j in range(J):
            A.addConstr(gp.quicksum(bf[j][k] * z[k] for k in range(K)) >= arg.m2 * y[j], name='con3')

    if arg.ChoiceTrue == False:
        for j in range(J):
            A.addConstr(gp.quicksum(bf[j][k] * z[k] for k in range(K)) <= arg.m3 * y[j], name='con4')
    # Optimize model
    A.optimize()
    dk = []
    z_rules = []
    for v in A.getVars():
        if 'z' in v.varName:
            z_rules.append(bool(v.x))
        print('%s %g' % (v.varName, v.x))
        dk.append(v.x)
    print('Obj: %g' % A.objVal)

    # rules_final = [i for i in arules[z_rules].sort_values(by = 'support', ascending = False)['itemsets']]
    rules_final = [i for i in arules[z_rules]['itemsets']]

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')

x_i = dk[:I]
y_j = dk[I:I+J]
z_k = dk[I+J:]
obje = A.objVal

x_i_a = np.array([x_i])
y_j_a = np.array([y_j])
z_k_a = np.array([z_k])

da = x_i_a*y_j_a
b = y_j_a.T*z_k_a
c = x_i_a.T*z_k_a

plt.figure(figsize=(40,400))
ax = sns.heatmap(rules_features - c,cmap = 'Blues_r',linewidths=.5)
plt.title('Heatmap of Diffrence C_ik', fontsize=15)
plt.title('Heatmap of Diffrence C_ik', fontsize=15)
plt.show()