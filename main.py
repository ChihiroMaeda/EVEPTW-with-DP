#列生成法でモデルを解くプログラム．
#翌日の電力価格を考慮したモデル．min a*(y_0-y_YDend) + b*(Y_i-y_i) - c*y_end
#aは夜間の電力価格，bは当日日中の電力価格，cは翌日日中の電力価格
#前日からの繰越のバッテリー残量をy_YDendとし，翌日の電力価格に応じて電力を持って帰ってくるモデル

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import pulp
from itertools import product
import csv
from concurrent import futures

#車両の数
K = 10

#顧客の数
n_1 = 10
n_2 = 10
n_3 = 10
n_4 = 10
n_5 = 10
n_6 = 10
n_7 = 10
n_8 = 10
n_9 = 10
n_10 = 10

n = [n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, n_9, n_10]

#充電ステーションの数
n_CS = 1

#時間枠の数
n_TW = 10

#エリア範囲(一辺)
area_length = 10

#ノードの総数
n_total = sum(n[k] for k in range(K)) + n_TW + 2

#業務時間(hour)
deli_hour = 5

#JEPXで公開されている取引価格のCSVファイルから，alpha, beta, gammaのリストを生成する関数
def price_data_create(start_month, file_name):
    month = start_month
    alpha = {month:[]}
    beta = {month:[]}
    gamma = {month:[]}
    a = []
    b = []
    with open(file_name, 'r',encoding='shift_jis') as f:
        dataReader = csv.reader(f)
        for row in dataReader:
            if row[0] == '受渡日':
                index_num = row.index('エリアプライス関西(円/kWh)')
                continue
            today = row[0].split('/')
            if month != int(today[1]):
                month = int(today[1])
                alpha[month] = []
                beta[month] = []
                gamma[month] = []
            if int(row[1]) == 31:
                alpha[month].append(sum(a)/len(a))
                #alpha.append(sum(a)/len(a))
                #print(alpha)
                a = []
                beta[month].append(b)
                #beta.append(b)
                #print(beta)
                b = []
            if (int(row[1]) > 30) or (int(row[1]) < 21):
                a.append(float(row[index_num]))
            else:
                b.append(float(row[index_num]))
    
    for i,v in enumerate(gamma):
        for j in range(1,len(beta[v])):
            g = sum(beta[v][j])/len(beta[v][j])
            gamma[v].append(g)
        
    return(alpha,beta,gamma)

#電力価格
alpha_list,beta_list,gamma_list = price_data_create(4, 'spot_summary_2023.csv')
alpha = alpha_list[5]   #業務時間外
beta = beta_list[5]   #業務時間内
gamma = gamma_list[5]   #翌日業務時間内の電力価格の平均
gamma[-1] = 0   #月が31日までのときは，30日の月に合わせる

#バッテリーの上限
Q = 40

#車両の走行可能距離(km)
mile = 100

#走行速度(km/h)
S = 40

#単位時間あたりの充電量
r = 50

#MTZ制約のパラメータ
M = 10000

#時間枠下限
elist = [deli_hour/n_TW*i for i in range(n_TW)]

#時間枠上限
llist = [deli_hour/n_TW*i for i in range(1,n_TW+1)]

#単位距離あたりのバッテリー消費量(Q/走行可能距離)
o = Q / 100

#キロあたりの電力消費量
o = Q/mile

#充電ステーション一回訪問あたりの充電量
q = deli_hour/n_TW*r

#file name
#バッテリー残量の推移以外の結果を記録するファイル
name = 'result/result231114_Monthly_January_N{0}CS1TW{1}_CG_Q{2}q{3}_MinCost.csv'.format(n_1,n_TW,Q,int(q))
#バッテリー残量の推移を記録するファイル
name2 = "result/optimalY231114_Monthly_January_N{0}CS1TW{1}_CG_Q{2}q{3}_MinCost.csv".format(n_1,n_TW,Q,int(q))

#主問題を定義
#c_list[k][i]: 車両kの時間枠の組合せiを選んだときの最適ルートのコスト.Dim = K*W[k]
#a_list[k][i][j]: 車両kの時間枠の組合せiに充電ステーションjが含まれているなら１. Dim = K*W[k]*n_TW
def solve_main(c_list, a_list):
  W = [len(a_list[k]) for k in range(K)]   #それぞれの車両についての時間枠の組合せの候補数
  print(W)
  prob = pulp.LpProblem('EVRP_main', pulp.LpMinimize)

  #変数
  #x_list[k][i]: 車両kが時間枠の組合せiを使うかどうか. Dim = K*W[k]
  x_list = [[pulp.LpVariable(f"x_{k}{i}", lowBound=0, cat='Binary') for i in range(W[k])] for k in range(K)]
  
  #目的関数
  prob += pulp.lpSum(pulp.lpDot(c_list[k], x_list[k]) for k in range(K))

  #制約式
  #時間枠の組合せを必ず一つ選ぶ．
  for k in range(K):
    prob += pulp.lpSum(x_list[k][i] for i in range(W[k])) == 1

  #ある時間帯fの充電ステーションを使えるのは一台以下.
  for f in range(n_TW):
    prob += pulp.lpSum(a_list[k][i][f]*x_list[k][i] for k in range(K) for i in range(W[k])) <= 1

  #求解
  prob.solve(pulp.PULP_CBC_CMD(msg = 0))

  xv_list = [[0]*W[k] for k in range(K)]   #xの最適解を格納するリスト. Dim = K*W[k]

  #xの最適解を格納
  for k in range(K):
    for i in range(W[k]):
      if x_list[k][i].value() > 0:
        xv_list[k][i] = 1

  return(prob, xv_list)

#双対問題を定義
def solve_dual(c_list, a_list):
  prob = pulp.LpProblem('EVRP_dual', pulp.LpMaximize)
  W = [len(a_list[k]) for k in range(K)]   #それぞれの車両についての時間枠の組合せの候補数

  #変数
  #主問題の一つ目の制約式に対応する双対変数．キャリーオーバー考慮の場合は定義域なし．Dim = K
  u_list = [pulp.LpVariable(f"u_{k}", cat="Continuous") for k in range(K)]
  #主問題の二つ目の制約式に対応する双対変数．Dim = n_TW
  v_list = [pulp.LpVariable(f"v_{f}", lowBound=0, cat="Continuous") for f in range(n_TW)]

  #目的関数
  prob += pulp.lpSum(u_list) - pulp.lpSum(v_list)

  #制約式
  for k in range(K):
    for i in range(W[k]):
      prob += c_list[k][i] - u_list[k] + pulp.lpSum(v_list[j]*a_list[k][i][j] for j in range(n_TW)) >= 0

  #求解
  prob.solve(pulp.PULP_CBC_CMD(msg = 0))

  #最適解を格納
  uv_list = [u.value() for u in u_list]
  vv_list = [v.value() for v in v_list]

  return(prob, uv_list, vv_list)

#違反制約式を見つける問題（子問題）
def solve_child(k, Q, t, M, r, elist, llist, o, D, vv_list, flag, ytd, day):
  #すでに違反制約がない場合（flag=1）は問題を解かない
  if flag==1:
    return(0)

  prob = pulp.LpProblem("EVRP_child", pulp.LpMinimize)

  #変数
  #z_list[i][j]: i, j間を通るなら１，通らないなら０
  z_list = [[None]*(n[k]+n_TW+2) for _ in range(n[k]+n_TW+2)]
  for i in range(n[k]+n_TW+2):
    for j in range(n[k]+n_TW+2):
      #if i == j: continue
      z_list[i][j] = pulp.LpVariable(f"z_{i}_{j}", cat="Binary")
  
  #y_list[i]: ノードiに到着したときのバッテリー残量．Dim = n[k]+n_TW+2
  y_list = [None]*(n[k]+n_TW+2)
  for i in range(n[k]+n_TW+2):
    y_list[i] = pulp.LpVariable(f"y_{i}", lowBound=0, upBound=Q, cat="Continuous")
  
  #Y_list[i]: 充電ステーションノードiを出発するときのバッテリー残量．Dim = n_TW
  Y_list = [None]*n_TW
  for i in range(n_TW):
    Y_list[i] = pulp.LpVariable(f"Y_{i}", lowBound = 0, upBound = Q, cat = 'Continuous')
  
  #T_list[i]: ノードiに到着したときの時間．Dim = n[k]+n_TW+2
  T_list = [None]*(n[k]+n_TW+2)
  for i in range(n[k]+n_TW+2):
    T_list[i] = pulp.LpVariable(f"T_{i}", lowBound = 0, cat = 'Continuous')

  #Dist: 走行距離を記録する変数．定式化とは関係なし．
  Dist = pulp.LpVariable("Dist")

  #y_total: 総充電量を記録する変数．定式化とは関係なし．
  y_total = pulp.LpVariable("y_total", lowBound = 0)

  #目的関数
  prob += pulp.lpSum(q*beta[day][j]*z_list[i][n[k]+1+j] for i in range(n[k]+n_TW+1) for j in range(n_TW)) + alpha[day]*(y_list[0]-ytd[k]) - gamma[day]*y_list[n[k]+n_TW+1] + pulp.lpSum(vv_list[j]*z_list[i][n[k]+1+j] for i in range(n[k]+n_TW+1) for j in range(n_TW))
  #制約式
  #Distの定義．定式化とは関係なし．
  prob += pulp.lpSum(D[k][i][j]*z_list[i][j] for i in range(n[k]+n_TW+1) for j in range(1,n[k]+n_TW+2)) == Dist

  #y_totalの定義．定式化には関係なし．
  prob += y_list[0] - ytd[k] + pulp.lpSum((Y_list[i] - y_list[n[k]+1+i]) for i in range(n_TW)) == y_total

  #夜間充電量が負にならないようにする制約
  prob += y_list[0] - ytd[k] >= 0

  #始点，終点，充電ステーション以外のノードからでていく辺は一本
  for i in range(1, n[k] + 1):
    prob += pulp.lpSum(z_list[i][j] for j in range(1, n[k] + n_TW + 2)) == 1
  
  #始点からでていく辺は一本
  prob += pulp.lpSum(z_list[0][j] for j in range(1, n[k] + n_TW + 1)) == 1

  #終点に入る辺は一本
  prob += pulp.lpSum(z_list[j][n[k]+n_TW+1] for j in range(1, n[k] + n_TW + 1)) == 1

  for i in range(n[k] + n_TW + 2):
    prob += z_list[i][i] == 0

  #デポ以外で入る辺と出る辺の本数が等しい
  for j in range(1, n[k] + n_TW + 1):
    prob += pulp.lpSum(z_list[i][j] for i in range(n[k] + n_TW + 1)) - pulp.lpSum(z_list[j][i] for i in range(1, n[k] + n_TW + 2)) == 0

  #充電ステーションに入る辺は一本以下（必ずしも立ち寄らない）
  for s in range(n_TW):
    prob += pulp.lpSum(z_list[i][n[k] + 1 + s] for i in range(n[k] + n_TW +1)) <= 1

  #充電ステーション以外のノードの到着時間と任意のノードの到着時間の関係
  for i in range(n[k] + 1):
    for j in range(1, n[k] + n_TW + 1):
      prob += T_list[i] + t[k][i][j]*z_list[i][j] - M*(1 - z_list[i][j]) <= T_list[j]

  #終点の到着時間の関係
  for i in range(n[k]+1):
    prob += T_list[i] + t[k][i][n[k]+n_TW+1]*z_list[i][n[k]+n_TW+1] - M*(1 - z_list[i][n[k]+n_TW+1]) <= T_list[n[k]+n_TW+1]

  #充電ステーションノードの到着時間と任意のノードの到着時間の関係
  for i in range(n_TW):
    for j in range(1, n[k] + n_TW + 1):
      prob += T_list[n[k] + 1 + i] + t[k][n[k] + 1 + i][j]*z_list[n[k] + 1 + i][j] + (Y_list[i]-y_list[n[k]+1+i])/r - (M + Q/r)*(1 - z_list[n[k] + 1 + i][j]) <= T_list[j]

  #終点のノードは別で
  for i in range(n_TW):
      prob += T_list[n[k] + 1 + i] + t[k][n[k] + 1 + i][n[k]+n_TW+1]*z_list[n[k] + 1 + i][n[k]+n_TW+1] + (Y_list[i]-y_list[n[k]+1+i])/r - (M + Q/r)*(1 - z_list[n[k] + 1 + i][n[k]+n_TW+1]) <= T_list[n[k]+n_TW+1]

  #充電ステーションの時間枠下限
  for i in range(n_TW):
    prob += elist[i] <= T_list[n[k] + 1 + i]

  for i in range(n_TW):
    prob += T_list[n[k] + 1 + i] <= llist[i]    #充電ステーションの時間枠上限
  """
  #充電ステーション以外の時間枠
  for i in range(1,n[k]+1):
          prob += i%n_TW - 1 <= T_list[i]    #顧客の時間枠下限

  for i in range(1,n[k]+1):
          prob += T_list[i] <= i%n_TW    #顧客の時間枠上限
  """
  #充電ステーション以外の時間枠
  #時間枠の上下限を表すリストを生成
  start_time = [0,1,2,3,4]
  for i in range(5):
    start_time += [0,1,2,3,4]
  end_time = [1,2,3,4,5]
  for i in range(5):
    end_time += [1,2,3,4,5]
  #顧客の時間枠の制約
  for i in range(1,n[k]+1):
    prob += start_time[i-1] <= T_list[i]    #顧客の時間枠下限

  for i in range(1,n[k]+1):
    prob += T_list[i] <= end_time[i-1]  #顧客の時間枠上限

  #充電時間が時間枠の上限をこえない
  for i in range(n_TW):
    prob += T_list[n[k] + 1 + i] + (Y_list[i]-y_list[n[k]+1+i])/r <= llist[i]

  #バッテリー残量の関係
  for i in range(1, n[k] + 1): 
    for j in range(1, n[k] + n_TW + 1):
        prob += y_list[j] <= y_list[i] - o*D[k][i][j]*z_list[i][j] + Q*(1 - z_list[i][j])

  #終点のノードは別で．意味は一つ上の制約といっしょ
  for i in range(1, n[k] + 1):
    prob += y_list[n[k]+n_TW+1] <= y_list[i] - o*D[k][i][n[k]+n_TW+1]*z_list[i][n[k]+n_TW+1] + Q*(1 - z_list[i][n[k]+n_TW+1])

  #始点のノードも別で
  for j in range(1, n[k] + n_TW + 2):
    prob += y_list[j] <= y_list[0] - o*D[k][0][j]*z_list[0][j] + Q*(1 - z_list[0][j])

  #充電ステーション発のバッテリー残量の関係
  #最終項に充電ステーション同士を移動するループを防ぐための十分小さな項を追加
  for i in range(n_TW):
    for j in range(1, n[k] + n_TW + 1):
      prob += y_list[j] <= Y_list[i] - o*D[k][n[k] + 1 + i][j]*z_list[n[k] + 1 + i][j] + Q*(1 - z_list[n[k] + 1 + i][j]) - 0.00001

  #終点ノードは別
  for i in range(n_TW):            
    prob += y_list[n[k]+n_TW+1] <= Y_list[i] - o*D[k][n[k] + 1 + i][n[k]+n_TW+1]*z_list[n[k] + 1 + i][n[k]+n_TW+1] + Q*(1 - z_list[n[k] + 1 + i][n[k]+n_TW+1])

  for i in range(n_TW):
    prob += y_list[n[k] + 1 + i] <= Y_list[i]

  # ステーションノード間の渡りを禁止
  #for i in range(n[k] +1, n[k] + n_TW + 1):
  #  for j in range(n[k] + 1, n[k] + n_TW + 1):
  #    prob += z_list[i][j] == 0

  #求解
  child_start = time.perf_counter()
  prob.solve(pulp.PULP_CBC_CMD(msg=0))
  child_end = time.perf_counter()

  #新しい時間枠の組合せaと電力コストc
  a = [0]*n_TW
  for i in range(n[k]+n_TW+2):
    for j in range(n_TW):
      if pulp.value(z_list[i][n[k]+1+j])==1:
        a[j] = 1
  c = sum(q*beta[day][j]*z_list[i][n[k]+1+j].value() for i in range(n[k]+n_TW+1) for j in range(n_TW)) + alpha[day]*(y_list[0].value() - ytd[k]) - gamma[day]*y_list[n[k]+n_TW+1].value()
  print("result of subproblem {}".format(k), (pulp.value(prob.objective),c,a))
  print("Runtime: ", child_end-child_start)

  #解の出力
  arc_list = []
  for i in range(n[k]+n_TW+2):
    for j in range(n[k]+n_TW+2):
      if pulp.value(z_list[i][j]) == 1:
         print('z({0},{1})='.format(i, j),pulp.value(z_list[i][j]))
         arc_list.append((i,j))
  
  for i in range(n[k] + n_TW + 2):
      print('T({})='.format(i), pulp.value(T_list[i]))
        
  for i in range(n[k] + n_TW + 2):
      print('y({})='.format(i), pulp.value(y_list[i]))

  for i in range(n_TW):
      print('Y({})='.format(i+n[k]+1), pulp.value(Y_list[i]))  

  cost = c + gamma[day]*y_list[n[k]+n_TW+1].value()

  #訪問ノードの順番示すリストを作成
  current_node = arc_list[0][1]
  order = [0, arc_list[0][1]]
  for i in range(len(arc_list)-1):
    for l in arc_list:
      if l[0] == current_node:
        current_node =l[1]
        order.append(l[1])

  #訪問順に並べたyのリスト
  y_list_in_order = []
  for i in order:
    y_list_in_order.append(y_list[i].value())
    if i > n[k] and i != n[k]+n_TW+1:
      y_list_in_order.append(Y_list[i-n[k]-1].value())

  return(pulp.value(prob.objective),c,a, pulp.value(y_list[n[k]+n_TW+1]),Dist.value(),y_total.value(),cost,y_list_in_order,y_list[0].value()-ytd[k])

#初期解を生成する問題(充電ステーションに立ち寄らない場合の電力コスト最小化．つまり，TSPとほぼいっしょ)
#充電ステーションに立ち寄らなくても電欠起こさないつもりで考えている．そうじゃない場合は別で初期解の生成の仕方を考える必要がある．
def solve_first(k, Q, t, M, r, elist, llist, o, D, ytd, day):
  prob = pulp.LpProblem("EVRP_child", pulp.LpMinimize)

  #変数
  #z_list[i][j]: i, j間を通るなら１，通らないなら０
  z_list = [[None]*(n[k]+n_TW+2) for _ in range(n[k]+n_TW+2)]
  for i in range(n[k]+n_TW+2):
    for j in range(n[k]+n_TW+2):
      #if i == j: continue
      z_list[i][j] = pulp.LpVariable(f"z_{i}_{j}", cat="Binary")
  
  #y_list[i]: ノードiに到着したときのバッテリー残量．Dim = n[k]+n_TW+2
  y_list = [None]*(n[k]+n_TW+2)
  for i in range(n[k]+n_TW+2):
    y_list[i] = pulp.LpVariable(f"y_{i}", lowBound=0, upBound=Q, cat="Continuous")
  
  #Y_list[i]: 充電ステーションノードiを出発するときのバッテリー残量．Dim = n_TW
  Y_list = [None]*n_TW
  for i in range(n_TW):
    Y_list[i] = pulp.LpVariable(f"Y_{i}", lowBound = 0, upBound = Q, cat = 'Continuous')
  
  #T_list[i]: ノードiに到着したときの時間．Dim = n[k]+n_TW+2
  T_list = [None]*(n[k]+n_TW+2)
  for i in range(n[k]+n_TW+2):
    T_list[i] = pulp.LpVariable(f"T_{i}", lowBound = 0, cat = 'Continuous')

  #Dist: 走行距離を記録する変数．定式化とは関係なし．
  Dist = pulp.LpVariable("Dist")

  #y_total: 総充電量を記録する変数．定式化とは関係なし．
  y_total = pulp.LpVariable("y_total", lowBound = 0)

  #目的関数
  #prob += gamma[day]*o*pulp.lpSum(D[i][j] * z_list[i][j] for i in range(n[k] + n_TW + 1) for j in range(1,n[k] + n_TW + 2)) + (alpha-gamma[day])*y_list[0] - alpha*ytd
  prob += alpha[day]*(y_list[0]-ytd[k]) - gamma[day]*y_list[n[k]+n_TW+1]

  #制約式
  #Distの定義．定式化とは関係なし．
  prob += pulp.lpSum(D[i][j]*z_list[i][j] for i in range(n[k]+n_TW+1) for j in range(1,n[k]+n_TW+2)) == Dist

  #y_totalの定義．定式化には関係なし．
  prob += y_list[0] -ytd[k] == y_total

  #夜間充電量が負にならないようにする制約
  prob += y_list[0] - ytd[k] >= 0

  #始点，終点，充電ステーション以外のノードからでていく辺は一本
  for i in range(1, n[k] + 1):
    prob += pulp.lpSum(z_list[i][j] for j in range(1, n[k] + n_TW + 2)) == 1
  
  #始点からでていく辺は一本
  prob += pulp.lpSum(z_list[0][j] for j in range(1, n[k] + n_TW + 1)) == 1

  #終点に入る辺は一本
  prob += pulp.lpSum(z_list[j][n[k]+n_TW+1] for j in range(1, n[k] + n_TW + 1)) == 1

  for i in range(n[k] + n_TW + 2):
    prob += z_list[i][i] == 0

  #デポ以外で入る辺と出る辺の本数が等しい
  for j in range(1, n[k] + n_TW + 1):
    prob += pulp.lpSum(z_list[i][j] for i in range(n[k] + n_TW + 1)) - pulp.lpSum(z_list[j][i] for i in range(1, n[k] + n_TW + 2)) == 0

  #充電ステーションに入る辺は0本（立ち寄らない）
  for s in range(n_TW):
    prob += pulp.lpSum(z_list[i][n[k] + 1 + s] for i in range(n[k] + n_TW +1)) == 0

  #充電ステーション以外のノードの到着時間と任意のノードの到着時間の関係
  for i in range(n[k] + 1):
    for j in range(1, n[k] + n_TW + 1):
      prob += T_list[i] + t[i][j]*z_list[i][j] - M*(1 - z_list[i][j]) <= T_list[j]

  #終点の到着時間の関係
  for i in range(n[k]+1):
    prob += T_list[i] + t[i][n[k]+n_TW+1]*z_list[i][n[k]+n_TW+1] - M*(1 - z_list[i][n[k]+n_TW+1]) <= T_list[n[k]+n_TW+1]

  #充電ステーションノードの到着時間と任意のノードの到着時間の関係
  for i in range(n_TW):
    for j in range(1, n[k] + n_TW + 1):
      prob += T_list[n[k] + 1 + i] + t[n[k] + 1 + i][j]*z_list[n[k] + 1 + i][j] + (Y_list[i] - y_list[n[k] + 1 + i])/r - (M + Q/r)*(1 - z_list[n[k] + 1 + i][j]) <= T_list[j]

  #終点のノードは別で
  for i in range(n_TW):
      prob += T_list[n[k] + 1 + i] + t[n[k] + 1 + i][n[k]+n_TW+1]*z_list[n[k] + 1 + i][n[k]+n_TW+1] + (Y_list[i] - y_list[n[k] + 1 + i])/r - (M + Q/r)*(1 - z_list[n[k] + 1 + i][n[k]+n_TW+1]) <= T_list[n[k]+n_TW+1]

  #充電ステーションの時間枠下限
  for i in range(n_TW):
    prob += elist[i] <= T_list[n[k] + 1 + i]

  for i in range(n_TW):
    prob += T_list[n[k] + 1 + i] <= llist[i]    #充電ステーションの時間枠上限
  """
  #充電ステーション以外の時間枠
  for i in range(1,n[k]+1):
    prob += i%n_TW - 1 <= T_list[i]    #顧客の時間枠下限

  for i in range(1,n[k]+1):
    prob += T_list[i] <= i%n_TW    #顧客の時間枠上限
  """
  
  start_time = [0,1,2,3,4]
  for i in range(5):
    start_time += [0,1,2,3,4]
  end_time = [1,2,3,4,5]
  for i in range(5):
    end_time += [1,2,3,4,5]
  
  for i in range(1,n[k]+1):
    prob += start_time[i-1] <= T_list[i]    #顧客の時間枠下限

  for i in range(1,n[k]+1):
    prob += T_list[i] <= end_time[i-1]  #顧客の時間枠上限

  #充電時間が時間枠の上限をこえない
  for i in range(n_TW):
    prob += T_list[n[k] + 1 + i] + (Y_list[i] - y_list[n[k] + 1 + i])/r <= llist[i]

  #バッテリー残量の関係
  for i in range(1, n[k] + 1): 
    for j in range(1, n[k] + n_TW + 1):
        prob += y_list[j] <= y_list[i] - o*D[i][j]*z_list[i][j] + Q*(1 - z_list[i][j])

  #終点のノードは別で．意味は一つ上の制約といっしょ
  for i in range(1, n[k] + 1):
    prob += y_list[n[k]+n_TW+1] <= y_list[i] - o*D[i][n[k]+n_TW+1]*z_list[i][n[k]+n_TW+1] + Q*(1 - z_list[i][n[k]+n_TW+1])

  #始点のノードも別で
  for j in range(1, n[k] + n_TW + 2):
    prob += y_list[j] <= y_list[0] - o*D[0][j]*z_list[0][j] + Q*(1 - z_list[0][j])

  #充電ステーション発のバッテリー残量の関係
  #最終項に充電ステーション同士を移動するループを防ぐための十分小さな項を追加
  for i in range(n_TW):
    for j in range(1, n[k] + n_TW + 1):
      prob += y_list[j] <= Y_list[i] - o*D[n[k] + 1 + i][j]*z_list[n[k] + 1 + i][j] + Q*(1 - z_list[n[k] + 1 + i][j]) - 0.00001

  #終点ノードは別
  for i in range(n_TW):            
    prob += y_list[n[k]+n_TW+1] <= Y_list[i] - o*D[n[k] + 1 + i][n[k]+n_TW+1]*z_list[n[k] + 1 + i][n[k]+n_TW+1] + Q*(1 - z_list[n[k] + 1 + i][n[k]+n_TW+1])

  for i in range(n_TW):
    prob += y_list[n[k] + 1 + i] <= Y_list[i]

  for i in range(n_TW):
    prob += Y_list[i] <= Q

  # ステーションノード間の渡りを禁止
  #for i in range(n[k] +1, n[k] + n_TW + 1):
  #  for j in range(n[k] + 1, n[k] + n_TW + 1):
  #    prob += z_list[i][j] == 0

  #求解
  prob.solve(pulp.PULP_CBC_CMD(msg=0))

  #新しい時間枠の組合せaと電力コストc
  a = [0]*n_TW
  for i in range(n[k]+1):
    for j in range(n_TW):
      if z_list[i][n[k]+1+j].value() > 0.5:
        a[j] = 1

  c = pulp.value(prob.objective)

  #解の出力
  arc_list = []
  for i in range(n[k]+n_TW+2):
    for j in range(n[k]+n_TW+2):
      if pulp.value(z_list[i][j]) == 1:
         print('z({0},{1})='.format(i, j),pulp.value(z_list[i][j]))
         arc_list.append((i,j))
  
  for i in range(n[k] + n_TW + 2):
      print('T({})='.format(i), pulp.value(T_list[i]))
        
  for i in range(n[k] + n_TW + 2):
      print('y({})='.format(i), pulp.value(y_list[i]))

  for i in range(n_TW):
      print('Y({})='.format(n[k]+1+i), pulp.value(Y_list[i]))

  cost = c + gamma[day]*y_list[n[k]+n_TW+1].value()

  #訪問ノードの順番示すリストを作成
  current_node = arc_list[0][1]
  order = [0, arc_list[0][1]]
  for i in range(len(arc_list)-1):
    for l in arc_list:
      if l[0] == current_node:
        current_node =l[1]
        order.append(l[1])

  #訪問順に並べたyのリスト
  y_list_in_order = []
  for i in order:
    y_list_in_order.append(y_list[i].value())
    if i > n[k] and i != n[k]+n_TW+1:
      y_list_in_order.append(Y_list[i-n[k]-1].value())

  return(prob,c,a,pulp.value(y_list[n[k]+n_TW+1]),Dist.value(),y_total.value(),cost,y_list_in_order,y_list[0].value()-ytd[k])


if __name__ == '__main__':

  with open(name, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(["day","status","obj","cost","dist","totalcharge","y_offhour","runtime","iteration"]+["TW of Vehicle{}".format(k) for k in range(1,K+1)]+["Obj of Vehicle{}".format(k) for k in range(1,K+1)]+["y_end of Vehicle {}".format(k) for k in range(1,K+1)])

  ytd = [0 for i in range(K)]

  for NumOfRepeat in range(30):
    print("Day ", NumOfRepeat+1)

    #データの作成
    np.random.seed(300+NumOfRepeat)    #シードの固定
    coordinate_CSX = area_length*np.random.rand()
    CSX = np.array([coordinate_CSX for i in range(n_TW)]) 
    coordinate_CSY = area_length*np.random.rand()
    CSY = np.array([coordinate_CSY for i in range(n_TW)]) 
    clientX = [area_length*np.random.rand(n[k]) for k in range(K)]   #顧客のX座標
    clientY = [area_length*np.random.rand(n[k]) for k in range(K)]   #顧客のY座標
    dataX = [np.concatenate([np.array([area_length/2+area_length%2]), clientX[k], CSX, np.array([area_length/2+area_length%2])]) for k in range(K)]
    dataY = [np.concatenate([np.array([area_length/2+area_length%2]), clientY[k], CSY, np.array([area_length/2+area_length%2])]) for k in range(K)]

    #距離行列
    D = [np.sqrt(abs(dataX[k].reshape(-1,1) - dataX[k])**2 + abs(dataY[k].reshape(-1,1) - dataY[k])**2) for k in range(K)]

    #ノード間移動にかかる時間
    t = [D[k]/S for k in range(K)]
    start = time.perf_counter()

    #列生成法の実行
    #初期解を生成
    a_list = [[] for k in range(K)]
    c_list = [[] for k in range(K)]
    y_end_list = [[] for k in range(K)]
    dist_list = [[] for k in range(K)]
    y_total_list = [[] for k in range(K)]
    cost_list = [[] for k in range(K)]
    y_in_order_list = [[] for k in range(K)]
    y_offhour_list = [[] for k in range(K)]
    for k in range(K):
      prob,c,a,y_end,dist,y_total,cost,y_in_order,y_offhour = solve_first(k, Q, t[k], M, r, elist, llist, o, D[k], ytd, NumOfRepeat)
      a_list[k].append(a)
      c_list[k].append(c)
      y_end_list[k].append(y_end)
      dist_list[k].append(dist)
      y_total_list[k].append(y_total)
      cost_list[k].append(cost)
      y_in_order_list[k].append(y_in_order)
      y_offhour_list[k].append(y_offhour)
    print("a_list: ",a_list)
    print("c_list: ",c_list)

    #列生成法
    flag = [0 for k in range(K)]   #列生成を終了するかを決めるフラグ
    for rep in range(100):
      print("iteration: ", rep+1)
      #列生成を終了するかを決めるフラグ
      #flag = [0 for k in range(K)]

      #双対問題を解く
      prob,uv_list,vv_list = solve_dual(c_list,a_list)
      print("Dual solution u, v: ", uv_list, vv_list)

      #子問題を車両ごと並列に解く
      result = []
      with futures.ProcessPoolExecutor(max_workers=5) as executor:
        future = [executor.submit(solve_child, k, Q, t, M, r, elist, llist, o, D, vv_list, flag[k],ytd,NumOfRepeat) for k in range(K)]
        for k in range(K):
          result.append(future[k].result())

      #違反制約式が存在するかの判定．存在するなら組合せ候補に追加．
      for k in range(K):
        if flag[k]==1:
          continue
        print(f"Violation {k}: ", result[k][0]-uv_list[k])
        if result[k][0] - uv_list[k] >= -0.1:
          flag[k] = 1
          continue
        c_list[k].append(result[k][1])
        a_list[k].append(result[k][2])
        y_end_list[k].append(result[k][3])
        dist_list[k].append(result[k][4])
        y_total_list[k].append(result[k][5])
        cost_list[k].append(result[k][6])
        y_in_order_list[k].append(result[k][7])
        y_offhour_list[k].append(result[k][8])
        print("a_list[{}]: ".format(k),a_list[k])
        print("c_list[{}]: ".format(k),c_list[k])

      #フラグの和がK（違反制約式が存在しない）なら列生成を終了．
      if sum(flag) == K:
        break

    prob,x_list = solve_main(c_list,a_list)
    print('main',pulp.LpStatus[prob.status],prob.objective.value())
    opt_a = []
    opt_c = []
    opt_y_end = []
    opt_dist = []
    opt_y_total = []
    opt_cost = []
    opt_y_in_order = []
    opt_y_offhour = []
    for k in range(K):
      for i in range(len(a_list[k])):
          if x_list[k][i] == 1:
              opt_a.append(a_list[k][i])
              opt_c.append(c_list[k][i])
              opt_y_end.append(y_end_list[k][i])
              opt_dist.append(dist_list[k][i])
              opt_y_total.append(y_total_list[k][i])
              opt_cost.append(cost_list[k][i])
              opt_y_in_order.append(y_in_order_list[k][i])
              opt_y_offhour.append(y_offhour_list[k][i])
              ytd[k] = y_end_list[k][i]   #ytdの更新
              print(a_list[k][i],c_list[k][i],y_end_list[k][i])

    end = time.perf_counter()

    Dist = sum(opt_dist)
    TotalCharge = sum(opt_y_total)
    Cost = sum(opt_cost)
    Y_OFFHOUR = sum(opt_y_offhour)

    with open(name, 'a') as f:
      writer = csv.writer(f)
      writer.writerow(["day{}".format(NumOfRepeat+1),pulp.LpStatus[prob.status],prob.objective.value(),Cost,Dist,TotalCharge,Y_OFFHOUR,end-start,rep+1] + opt_a + opt_c + opt_y_end) 

    with open(name2, "a") as f:
      writer = csv.writer(f)
      for k in range(K):
        writer.writerow(["day{}".format(NumOfRepeat+1), "Vehicle{}".format(k)]+ [opt_a[k]] + [i for i in opt_y_in_order[k]])

    print("実行時間：", end-start)
    print("列生成回数：", rep)
