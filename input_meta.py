# -*- coding:utf-8 -*-

import pdb
from time import time
import datetime as dt
import numpy as np
from scipy.special import gammaln
import matplotlib
#matplotlib.use('Agg') #DIPLAYの設定
import matplotlib.pyplot as plt
import resource
import codecs
import random
import csv
from collections import defaultdict
from collections import namedtuple
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import commands

#--variable--
action_dic = {}
team_dic = {}
player1_dic = {}
player2_dic = {}
t_dic = []
elem = namedtuple("elem", "t, re_t, team, p, a, x, y, s")
#絶対時刻, ハーフ相対時間，チームID，アクションID, x座標，y座標, success
Stoppage = [5, 7, 11, 12]

xmax = -10 ** 14
xmin = 10 ** 14
ymax = -10 ** 14
ymin = 10 ** 14
tmin = 0
tmax = 0

period1_start = 0
period1_end = 0
period2_start = 0
period2_end = 0
period3_start = 0
period3_end = 0
period4_start = 0
period4_end = 0

period1_start_re_t = 0
period1_end_re_t = 0
period2_start_re_t = 0
period2_end_re_t = 0
period3_start_re_t = 0
period3_end_re_t = 0
period4_start_re_t = 0
period4_end_re_t = 0

D = defaultdict(int)#アクションID付きボール位置データ D[counter]
N = 0

Seq_Team1_of = defaultdict(int)
Seq_Team2_of = defaultdict(int)
N_Team1_of = -1
N_Team2_of = -1

shot_period1_Team1_re_t = []
shot_period2_Team1_re_t = []
shot_period3_Team1_re_t = []
shot_period4_Team1_re_t = []

shot_period1_Team2_re_t = []
shot_period2_Team2_re_t = []
shot_period3_Team2_re_t = []
shot_period4_Team2_re_t = []

shot_success_period1_Team1_re_t = []
shot_success_period2_Team1_re_t = []
shot_success_period3_Team1_re_t = []
shot_success_period4_Team1_re_t = []

shot_success_period1_Team2_re_t = []
shot_success_period2_Team2_re_t = []
shot_success_period3_Team2_re_t = []
shot_success_period4_Team2_re_t = []

#Dx = 9#x方向メッシュ分割数
#Dy = 6#y方向メッシュ分割数
Dx = 6
Dy = 4

K = 5
C = ['blue', 'red', 'green', 'grey', 'gold']
#C = ['#ff7f7f', '#ff7fbf', '#ff7fff', '#bf7fff', '#7f7fff', '#7fbfff', '#7fffff', '#7fffbf', '#7fff7f', '#bfff7f', '#fff7f', '#ffb7f']

kde_width = 0.3
#------------


def input():
#--Input--

    global xmax, xmin, ymax, ymin
    global period1_start, period1_end, period2_start, period2_end, period3_start, period3_end, period4_start, period4_end
    global N

    counter = 0
    t0 = time()
    filename = "processed_metadata.csv"
    fin = open(filename)

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period1_start = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period1_end = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period2_start = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period2_end = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period3_start = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period3_end = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period4_start = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

    temp = fin.readline().rstrip("\r\n").split(",")
    A = temp[0].split(".")
    B = A[0].split(":")
    C = A[1]
    period4_end = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

    counter = 0
    for row in fin:
        temp = row.rstrip("\r\n").split(",")

        A = temp[0].split(".")
        B = A[0].split(":")
        C = A[1]
        t = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

        A = temp[1].split(".")
        B = A[0].split(":")
        C = A[1]
        re_t = int(B[0]) * 3600 + int(B[1]) * 60 + int(B[2])

        team = int(temp[2])
        if team not in team_dic:
            team_dic[team] = len(team_dic)

        player = int(temp[3])
        if team == 8:
            if player not in player1_dic:
                player1_dic[player] = len(player1_dic)
        elif team == 9:
            if player not in player2_dic:
                player2_dic[player] = len(player2_dic)
        else:
            print "err"

        action = int(temp[4])
        if action not in action_dic:
            action_dic[action] = len(action_dic)

        x = float(temp[5])
        if xmin > x:
            xmin = x
        if xmax < x:
            xmax = x

        y = float(temp[6])
        if ymin > y:
            ymin = y
        if ymax < y:
            ymax = y

        s = int(temp[7])

        f = elem(t, re_t, team, player, action, x, y, s)
        D[counter] = f
        counter += 1

    fin.close()
    N = counter - 1
    Reverse_Seq()#後半の攻撃を反転
    Make_re_t()#各ピリオド開始からの相対時間を生成
    print "time:%f" % (time()-t0)


def Make_re_t():
#--各ピリオド開始からの相対時間を生成--
    global period1_start_re_t, period1_end_re_t, period2_start_re_t, period2_end_re_t, period3_start_re_t, period3_end_re_t, period4_start_re_t, period4_end_re_t

    period1_start_re_t = 0
    period1_end_re_t = period1_end
    period2_start_re_t = 0
    period2_end_re_t = period2_end - period2_start + 1
    period3_start_re_t = 0
    period3_end_re_t = period3_end - period3_start + 1
    period4_start_re_t = 0
    period4_end_re_t = period4_end - period4_start + 1

    for n in range(N):
        x = D[n]
        t = x.t
        if period2_start < t and t < period2_end:
            f = elem(t, x.re_t - period2_start, x.team, x.p, x.a, x.x, x.y, x.s)
            D[n] = f
        elif period3_start < t and t < period3_end:
            f = elem(t, x.re_t - period3_start, x.team, x.p, x.a, x.x, x.y, x.s)
            D[n] = f
        elif period4_start < t and t < period4_end:
            f = elem(t, x.re_t - period4_start, x.team, x.p, x.a, x.x, x.y, x.s)
            D[n] = f

        #注意：4ピリオド終了後にアクションがある

def Reverse_Seq():
#--後半反転--
    for n in range(N):
        x = D[n]
        t = x.t
        if t > period3_start:#左方向に攻撃            
            f = elem(t, x.re_t, x.team, x.p, x.a, xmax - x.x, ymax - x.y, x.s)
            D[n] = f


def Seq_Team_of():
#--offense ボール軌跡--
    global N_Team1_of, N_Team2_of
    t0 = time()
    
    n = 0
    pre_team = 0
    flag = 0

    while n < N:
        team = D[n].team
        action = D[n].a

        if action in Stoppage:
            flag = 1
            n += 1

        else:
            if pre_team == team:
                if team == 8:
                    if flag == 1:
                        N_Team1_of += 1
                        flag = 0

                    x = D[n]
                    if np.size(Seq_Team1_of[N_Team1_of]) == 1: 
                        #Seq_Team1_ofにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y, x.s])
                        Seq_Team1_of[N_Team1_of] = f
                    else:
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y, x.s])
                        Seq_Team1_of[N_Team1_of] = np.vstack([Seq_Team1_of[N_Team1_of],f])

                    n += 1
                    pre_team = team

                elif team == 9:
                    if flag == 1:
                        N_Team2_of += 1
                        flag = 0

                    x = D[n]
                    if np.size(Seq_Team2_of[N_Team2_of]) == 1: 
                        #Seq_Team2_ofにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y, x.s])
                        Seq_Team2_of[N_Team2_of] = f
                    else:
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y, x.s])
                        Seq_Team2_of[N_Team2_of] = np.vstack([Seq_Team2_of[N_Team2_of],f])

                    n += 1
                    pre_team = team

            else:
                if team == 8:
                    N_Team1_of += 1
                    x = D[n]
                    if np.size(Seq_Team1_of[N_Team1_of]) == 1: 
                        #Seq_Team1_ofにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y, x.s])
                        Seq_Team1_of[N_Team1_of] = f
                    else:
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y, x.s])
                        Seq_Team1_of[N_Team1_of] = np.vstack([Seq_Team1_of[N_Team1_of],f])

                    n += 1
                    pre_team = team

                elif team == 9:
                    N_Team2_of += 1
                    x = D[n]
                    if np.size(Seq_Team2_of[N_Team2_of]) == 1: 
                        #Seq_Team2_ofにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y, x.s])
                        Seq_Team2_of[N_Team2_of] = f
                    else:
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y, x.s])
                        Seq_Team2_of[N_Team2_of] = np.vstack([Seq_Team2_of[N_Team2_of],f])

                    n += 1
                    pre_team = team

    N_Team1_of = len(Seq_Team1_of)
    N_Team2_of = len(Seq_Team2_of)

    #pdb.set_trace()
    Remove_one()#一回しかアクションがない攻撃は消去
    Quantization()#位置情報をメッシュ状に量子化
    shot_timing()#シュートのタイミングを取得
    shot_success_timing()#成功したシュートのタイミングを取得
    Seq_record()#攻撃機会毎にファイルに掃き出す
    #Visualize_Seq()#オフェンス時のボール軌跡データ可視化
    print 'time:%f' % (time()-t0)

def Seq_record():
    
    for n in range(N_Team1_of):
        np.savetxt('Seq_Team8_Data/Team8_Offence' + str(n) + '.csv', Seq_Team1_of[n], delimiter=',')

    for n in range(N_Team2_of):
        np.savetxt('Seq_Team9_Data/Team9_Offence' + str(n) + '.csv', Seq_Team2_of[n], delimiter=',')


def Remove_one():
#--一回しかアクションがない攻撃は消去--
    global N_Team1_of, N_Team2_of, Seq_Team1_of, Seq_Team2_of

    tmp_Seq_Team1_of = defaultdict(int)
    tmp_Seq_Team2_of = defaultdict(int)    

    counter = 0
    for n in range(N_Team1_of):
        S = Seq_Team1_of[n]
        S_size = np.size(S)/7
        if S_size > 1:
            tmp_Seq_Team1_of[counter] = S
            counter += 1

    counter = 0
    for n in range(N_Team2_of):
        S = Seq_Team2_of[n]
        S_size = np.size(S)/7
        if S_size > 1:
            tmp_Seq_Team2_of[counter] = S
            counter += 1

    Seq_Team1_of = tmp_Seq_Team1_of
    Seq_Team2_of = tmp_Seq_Team2_of
    N_Team1_of = len(tmp_Seq_Team1_of)
    N_Team2_of = len(tmp_Seq_Team2_of)


def shot_timing():
#--シュートのタイミングを取得--

    for n in range(N_Team1_of):
        S = Seq_Team1_of[n]
        action = int(S[len(S)-1][4])
        #攻撃の最後にしたシュートのみになっている

        shot_t = S[len(S)-1][0]
        shot_re_t = S[len(S)-1][1]

        if action == 0 or action == 1:
            if shot_t < period1_end:
                shot_period1_Team1_re_t.append(shot_re_t)
            elif period2_start < shot_t and shot_t < period2_end:
                shot_period2_Team1_re_t.append(shot_re_t)
            elif period3_start < shot_t and shot_t < period3_end:
                shot_period3_Team1_re_t.append(shot_re_t)
            elif period4_start < shot_t and shot_t < period4_end:
                shot_period4_Team1_re_t.append(shot_re_t)

    for n in range(N_Team2_of):
        S = Seq_Team2_of[n]
        action = int(S[len(S)-1][4])
        shot_t = S[len(S)-1][0]
        shot_re_t = S[len(S)-1][1]

        if action == 0 or action == 1:
            if shot_t < period1_end:
                shot_period1_Team2_re_t.append(shot_re_t)
            elif period2_start < shot_t and shot_t < period2_end:
                shot_period2_Team2_re_t.append(shot_re_t)
            elif period3_start < shot_t and shot_t < period3_end:
                shot_period3_Team2_re_t.append(shot_re_t)
            elif period4_start < shot_t and shot_t < period4_end:
                shot_period4_Team2_re_t.append(shot_re_t)


def shot_success_timing():
#--成功したシュートのタイミングを取得--

    for n in range(N_Team1_of):
        S = Seq_Team1_of[n]
        action = int(S[len(S)-1][4])
        #攻撃の最後にしたシュートのみになっている

        success = int(S[len(S)-1][7])

        shot_t = S[len(S)-1][0]
        shot_re_t = S[len(S)-1][1]

        if (action == 0 or action == 1) and success == 1:
            if shot_t < period1_end:
                shot_success_period1_Team1_re_t.append(shot_re_t)
            elif period2_start < shot_t and shot_t < period2_end:
                shot_success_period2_Team1_re_t.append(shot_re_t)
            elif period3_start < shot_t and shot_t < period3_end:
                shot_success_period3_Team1_re_t.append(shot_re_t)
            elif period4_start < shot_t and shot_t < period4_end:
                shot_success_period4_Team1_re_t.append(shot_re_t)

    for n in range(N_Team2_of):
        S = Seq_Team2_of[n]
        action = int(S[len(S)-1][4])

        success = int(S[len(S)-1][7])

        shot_t = S[len(S)-1][0]
        shot_re_t = S[len(S)-1][1]

        if (action == 0 or action == 1) and success == 1:
            if shot_t < period1_end:
                shot_success_period1_Team2_re_t.append(shot_re_t)
            elif period2_start < shot_t and shot_t < period2_end:
                shot_success_period2_Team2_re_t.append(shot_re_t)
            elif period3_start < shot_t and shot_t < period3_end:
                shot_success_period3_Team2_re_t.append(shot_re_t)
            elif period4_start < shot_t and shot_t < period4_end:
                shot_success_period4_Team2_re_t.append(shot_re_t)


def Quantization():
#--位置情報をメッシュ状に量子化--

    tmp_xmax = float(2 * xmax) + 1
    tmp_ymax = float(2 * ymax) + 1

    t0 = time()
    for n in range(N_Team1_of):
        S = Seq_Team1_of[n]
        S_size = np.size(S)/8
        for s in range(S_size):
            line = S[s]
            x = line[5]
            y = line[6]

            tmp_x = float(x) - xmin#原点をコートの左下に
            tmp_y = float(y) - ymin

            Mx_id = int( Dx * tmp_x / tmp_xmax )#メッシュidを計算
            My_id = int( Dy * tmp_y / tmp_ymax )
            My_id = Dy - 1 - int( Dy * tmp_y / tmp_ymax )#左上がメッシュid=0になるように反転

            M_id = My_id * Dx + Mx_id
            if s == 0:
                M_id_set = np.array(M_id)
            else:
                M_id_set = np.vstack([M_id_set, M_id])

        Seq_Team1_of[n] = np.hstack([S, M_id_set])

    for n in range(N_Team2_of):
        S = Seq_Team2_of[n]
        S_size = np.size(S)/8
        for s in range(S_size):
            line = S[s]
            x = line[5]
            y = line[6]

            tmp_x = float(x) - xmin#原点をコートの左下に
            tmp_y = float(y) - ymin

            Mx_id = int( Dx * tmp_x / tmp_xmax )#メッシュidを計算
            My_id = int( Dy * tmp_y / tmp_ymax )
            My_id = Dy - 1 - int( Dy * tmp_y / tmp_ymax )#左上がメッシュid=0になるように反転

            M_id = My_id * Dx + Mx_id
            if s == 0:
                M_id_set = np.array(M_id)
            else:
                M_id_set = np.vstack([M_id_set, M_id])

        Seq_Team2_of[n] = np.hstack([S, M_id_set])

    print 'time:%f' % (time()-t0)


def make_BoF():
#--パス系列と量子化された位置情報を含むBag-of-Feature作成--

    t0 = time()

    flag = 0
    for n in range(N_Team1_of):
        N_player = len(player1_dic)
        M = np.zeros([N_player, N_player])
        L = np.zeros(Dx * Dy)
        S = Seq_Team1_of[n]
        Pass_Series = S[:,3]
        for i in range(len(Pass_Series) - 1):
            now_p_ind = Pass_Series[i]
            next_p_ind = Pass_Series[i+1]

            now_p = player1_dic[now_p_ind]
            next_p = player1_dic[next_p_ind]
            M[now_p, next_p] += 1


        #プレイヤーグラフ対称の場合-----------
        '''
        M2 = np.zeros([N_player, N_player])
        for i in range(N_player):
            for j in range(N_player):
                M2[i,j] = M[i,j] + M[j,i]

        pass_line = []
        for i in range(N_player):
            for j in range(N_player):
                if i < j:
                    pass_line.append(M2[i,j])
        pass_line = np.array(pass_line)
        '''
        #-------------------------------------

        #プレイヤーグラフ非対称の場合---------
        pass_line = []
        for i in range(N_player):
            for j in range(N_player):
                if i != j:
                    pass_line.append(M[i,j])
        pass_line = np.array(pass_line)
        #-------------------------------------

        #正規化----------------------
        S_pass_line = np.sum(pass_line)
        if np.sum(S_pass_line) != 0:
            for i in range(len(pass_line)):
               # if pass_line[i] != 0:
                pass_line[i] = (pass_line[i] - 0.00001) / S_pass_line
        #----------------------------
        
        Loc_Series = S[:,7]
        for i in range(len(Loc_Series)):
            Mesh = Loc_Series[i]
            L[Mesh] += 1

        #正規化----------------------
        S_L = np.sum(L)
        if np.sum(S_L) != 0:
            for i in range(len(L)):
                #if L[i] != 0:
                    L[i] = (L[i] - 0.00001) / S_L
        #---------------------------
        
        line = np.hstack([pass_line, L])
        if flag == 0:
            BoF_Team1 = line
            flag = 1
        else:
            BoF_Team1 = np.vstack([BoF_Team1, line])


    flag = 0
    for n in range(N_Team2_of):
        N_player = len(player2_dic)
        M = np.zeros([N_player, N_player])
        L = np.zeros(Dx * Dy)
        S = Seq_Team2_of[n]
        Pass_Series = S[:,3]
        for i in range(len(Pass_Series) - 1):
            now_p_ind = Pass_Series[i]
            next_p_ind = Pass_Series[i+1]

            now_p = player2_dic[now_p_ind]
            next_p = player2_dic[next_p_ind]
            M[now_p, next_p] += 1

        #プレイヤーグラフ対称の場合-----------
        '''
        M2 = np.zeros([N_player, N_player])
        for i in range(N_player):
            for j in range(N_player):
                M2[i,j] = M[i,j] + M[j,i]

        pass_line = []
        for i in range(N_player):
            for j in range(N_player):
                if i < j:
                    pass_line.append(M2[i,j])
        pass_line = np.array(pass_line)
        '''
        #-------------------------------------

        #プレイヤーグラフ非対称の場合-----------                
        pass_line = []
        for i in range(N_player):
            for j in range(N_player):
                if i != j:
                    pass_line.append(M[i,j])
        pass_line = np.array(pass_line)
        #-------------------------------------

        #正規化----------------------
        S_pass_line = np.sum(pass_line)
        if np.sum(S_pass_line) != 0:
            for i in range(len(pass_line)):
                #if pass_line[i] != 0:
                    pass_line[i] = (pass_line[i] - 0.00001) / S_pass_line
        #----------------------------

        Loc_Series = S[:,7]
        for i in range(len(Loc_Series)):
            Mesh = Loc_Series[i]
            L[Mesh] += 1

        #正規化----------------------
        S_L = np.sum(L)
        if np.sum(L) != 0:
            for i in range(len(L)):
                #if L[i] != 0:
                    L[i] = (L[i] - 0.00001) / S_L
        #---------------------------
        
        line = np.hstack([pass_line, L])

        if flag == 0:
            BoF_Team2 = line
            flag = 1
        else:
            BoF_Team2 = np.vstack([BoF_Team2, line])

    #BoF_Team1, BoF_Team2 = normalization(BoF_Team1, BoF_Team2)#平均0分散1に標準化
    print 'time:%f' % (time()-t0)
    return BoF_Team1, BoF_Team2


def normalization(BoF_Team1, BoF_Team2):
#--平均0分散1に標準化--

    dim = np.shape(BoF_Team1)[1]
    #del_dim_Team1 = []
    #del_dim_Team2 = []
    for i in range(dim):
        X = BoF_Team1[:,i]
        sumX = np.sum(X)
        if sumX != 0:
            average = np.mean(X)
            standard_dev = np.std(X)
            BoF_Team1[:,i] = (BoF_Team1[:,i] - average) / standard_dev
        #else:
        #    del_dim_Team1.append(i)

        X = BoF_Team2[:,i]
        sumX = np.sum(X)
        if sumX != 0:
            average = np.mean(BoF_Team2[:,i])
            standard_dev = np.std(BoF_Team2[:,i])
            BoF_Team2[:,i] = (BoF_Team2[:,i] - average) / standard_dev
        #else:
        #    del_dim_Team2.append(i)

    #BoF_Team1 = np.delete(BoF_Team1, del_dim_Team1, 1)
    #BoF_Team2 = np.delete(BoF_Team2, del_dim_Team2, 1)

    return BoF_Team1, BoF_Team2

def Visualize_Seq():
#--ボール軌跡可視化--
    t0 = time()
    for n in range(N_Team1_of):
        S = Seq_Team1_of[n]
        timing = int(S[0,1])
        team = int(S[0,2])
        action = int(S[np.shape(S)[0] - 1, 4])
        X = S[:,5]
        Y = S[:,6]

        fig = plt.figure()
        if action == 15:
            plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, \
                       scale_units='xy', angles='xy', scale=1, color='red')
        else:
            plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, \
                       scale_units='xy', angles='xy', scale=1, color='darkcyan')
        plt.axis([0, 600, 0, 330])
        plt.savefig('Seq_Team' + str(team) + '/Seq_Team1_of'+'_no'+str(n)+'_t'+str(timing)+'.png')
        plt.close()

    for n in range(N_Team2_of):
        S = Seq_Team2_of[n]
        timing = int(S[0,1])
        team = int(S[0,2])
        action = int(S[np.shape(S)[0] - 1, 4])
        X = S[:,5]
        Y = S[:,6]
        fig = plt.figure()
        if action == 15:
            plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, \
                       scale_units='xy', angles='xy', scale=1, color='red')
        else:
            plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, \
                       scale_units='xy', angles='xy', scale=1, color='darkcyan')
        plt.axis([0, 600, 0, 330])
        plt.savefig('Seq_Team' + str(team) + '/Seq_Team2_of'+'_no'+str(n)+'_t'+str(timing)+'.png')
        plt.close()

    print 'time:%f' % (time()-t0)


def Possession():
#--Possession--

    x_period1 = np.arange(period1_end_re_t + 1)
    y0_period1 = np.zeros(period1_end_re_t + 1)

    x_period2 = np.arange(period2_end_re_t + 1)
    y0_period2 = np.zeros(period2_end_re_t + 1)

    x_period3 = np.arange(period3_end_re_t + 1)
    y0_period3 = np.zeros(period3_end_re_t + 1)

    x_period4 = np.arange(period4_end_re_t + 1)
    y0_period4 = np.zeros(period4_end_re_t + 1)

    Y_period1_Team1_of = np.copy(y0_period1)
    Y_period2_Team1_of = np.copy(y0_period2)
    Y_period3_Team1_of = np.copy(y0_period3)
    Y_period4_Team1_of = np.copy(y0_period4)

    for i in range(N_Team1_of):
        S = Seq_Team1_of[i]
        start_t = S[0,0]
        start_re_t = S[0,1]
        end_re_t = S[len(S) - 1,1]
        period = int(end_re_t - start_re_t)
        if start_t < period1_end:
            for j in range(period):
                Y_period1_Team1_of[start_re_t + j] = 1.0
        elif period2_start < start_t and start_t < period2_end:
            for j in range(period):
                Y_period2_Team1_of[start_re_t + j] = 1.0
        elif period3_start < start_t and start_t < period3_end:
            for j in range(period):
                Y_period3_Team1_of[start_re_t + j] = 1.0
        elif period4_start < start_t and start_t < period4_end:
            for j in range(period):
                Y_period4_Team1_of[start_re_t + j] = 1.0

    fig = plt.figure(figsize=(16,4))
    plt.subplots_adjust(hspace=1.5)

    plt.subplot(4, 1, 1)
    plt.fill_between(x_period1, y0_period1, Y_period1_Team1_of, \
                     edgecolor = 'mediumaquamarine', facecolor = 'mediumaquamarine')
    tempX = np.array(shot_period1_Team1_re_t)
    tempY = np.ones(len(shot_period1_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period1_Team1_re_t)
    tempY = np.ones(len(shot_success_period1_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period1_start_re_t, period1_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_period1_offense')

    plt.subplot(4, 1, 2)
    plt.fill_between(x_period2, y0_period2, Y_period2_Team1_of, edgecolor = 'mediumaquamarine', \
                     facecolor = 'mediumaquamarine')
    tempX = np.array(shot_period2_Team1_re_t)
    tempY = np.ones(len(shot_period2_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period2_Team1_re_t)
    tempY = np.ones(len(shot_success_period2_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period2_start_re_t, period2_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_period2_offense')

    plt.subplot(4, 1, 3)
    plt.fill_between(x_period3, y0_period3, Y_period3_Team1_of, edgecolor = 'mediumaquamarine', \
                     facecolor = 'mediumaquamarine')
    tempX = np.array(shot_period3_Team1_re_t)
    tempY = np.ones(len(shot_period3_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period3_Team1_re_t)
    tempY = np.ones(len(shot_success_period3_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period3_start_re_t, period3_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_period3_offense')

    plt.subplot(4, 1, 4)
    plt.fill_between(x_period4, y0_period4, Y_period4_Team1_of, edgecolor = 'mediumaquamarine', \
                     facecolor = 'mediumaquamarine')
    tempX = np.array(shot_period4_Team1_re_t)
    tempY = np.ones(len(shot_period4_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period4_Team1_re_t)
    tempY = np.ones(len(shot_success_period4_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period4_start_re_t, period4_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_period4_offense')


    plt.savefig('Seq_Team1/Team1_offense.png')
    #plt.show()
    #pdb.set_trace()
    plt.close()



    Y_period1_Team2_of = np.copy(y0_period1)
    Y_period2_Team2_of = np.copy(y0_period2)
    Y_period3_Team2_of = np.copy(y0_period3)
    Y_period4_Team2_of = np.copy(y0_period4)

    for i in range(N_Team2_of):
        S = Seq_Team2_of[i]
        start_t = S[0,0]
        start_re_t = S[0,1]
        end_re_t = S[len(S) - 1,1]
        period = int(end_re_t - start_re_t)
        if start_t < period1_end:
            for j in range(period):
                Y_period1_Team2_of[start_re_t + j] = 1.0
        elif period2_start < start_t and start_t < period2_end:
            for j in range(period):
                Y_period2_Team2_of[start_re_t + j] = 1.0
        elif period3_start < start_t and start_t < period3_end:
            for j in range(period):
                Y_period3_Team2_of[start_re_t + j] = 1.0
        elif period4_start < start_t and start_t < period4_end:
            for j in range(period):
                Y_period4_Team2_of[start_re_t + j] = 1.0

    fig = plt.figure(figsize=(16,4))
    plt.subplots_adjust(hspace=1.5)

    plt.subplot(4, 1, 1)
    plt.fill_between(x_period1, y0_period1, Y_period1_Team2_of, \
                     edgecolor = 'mediumaquamarine', facecolor = 'mediumaquamarine')
    tempX = np.array(shot_period1_Team2_re_t)
    tempY = np.ones(len(shot_period1_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period1_Team2_re_t)
    tempY = np.ones(len(shot_success_period1_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period1_start_re_t, period1_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_period1_offense')

    plt.subplot(4, 1, 2)
    plt.fill_between(x_period2, y0_period2, Y_period2_Team2_of, edgecolor = 'mediumaquamarine', \
                     facecolor = 'mediumaquamarine')
    tempX = np.array(shot_period2_Team2_re_t)
    tempY = np.ones(len(shot_period2_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period2_Team2_re_t)
    tempY = np.ones(len(shot_success_period2_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period2_start_re_t, period2_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_period2_offense')

    plt.subplot(4, 1, 3)
    plt.fill_between(x_period3, y0_period3, Y_period3_Team2_of, edgecolor = 'mediumaquamarine', \
                     facecolor = 'mediumaquamarine')
    tempX = np.array(shot_period3_Team2_re_t)
    tempY = np.ones(len(shot_period3_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period3_Team2_re_t)
    tempY = np.ones(len(shot_success_period3_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period3_start_re_t, period3_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_period3_offense')

    plt.subplot(4, 1, 4)
    plt.fill_between(x_period4, y0_period4, Y_period4_Team2_of, edgecolor = 'mediumaquamarine', \
                     facecolor = 'mediumaquamarine')
    tempX = np.array(shot_period4_Team2_re_t)
    tempY = np.ones(len(shot_period4_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period4_Team2_re_t)
    tempY = np.ones(len(shot_success_period4_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period4_start_re_t, period4_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_period4_offense')

    plt.savefig('Seq_Team2/Team2_offense.png')
    #plt.show()
    #pdb.set_trace()
    plt.close()


def Clustering(BoF_Team1, BoF_Team2):
    t0 = time()

    PCA_threshold = 0.8

    #--Team1--
    dim = np.shape(BoF_Team1)[0]
    threshold_dim = 0
    for i in range(dim):
        pca = PCA(n_components = i)
        pca.fit(BoF_Team1)
        X = pca.transform(BoF_Team1)
        E = pca.explained_variance_ratio_
        if np.sum(E) > PCA_threshold:
            thereshold_dim = len(E)
            print 'Team1 dim:%d' % thereshold_dim
            break

    pca = PCA(n_components = thereshold_dim)
    pca.fit(BoF_Team1)
    X = pca.transform(BoF_Team1)

    min_score = 10000
    for i in range(100):
        model = KMeans(n_clusters=K, init='k-means++', max_iter=300, tol=0.0001).fit(X)
        if min_score > model.score(X):
            min_score = model.score(X)
            labels_Team1 = model.labels_
    print min_score

    pca = PCA(n_components = 2)
    pca.fit(BoF_Team1)
    X = pca.transform(BoF_Team1)
    for k in range(K):
        labels_Team1_ind = np.where(labels_Team1 == k)[0]
        plt.scatter(X[labels_Team1_ind,0], X[labels_Team1_ind,1], color=C[k])
    plt.legend(['C0','C1','C2','C3','C4'], loc=4)

    plt.title('Team1_PCA_kmeans')
    plt.savefig('Seq_Team1/Team1_PCA_kmeans.png')
    plt.show()
    plt.close()
    np.savetxt('Seq_Team1/labels_Team1.csv', labels_Team1, delimiter=',')

    #--Team2--
    dim = np.shape(BoF_Team2)[0]
    threshold_dim = 0
    for i in range(dim):
        pca = PCA(n_components = i)
        pca.fit(BoF_Team2)
        X = pca.transform(BoF_Team2)
        E = pca.explained_variance_ratio_
        if np.sum(E) > PCA_threshold:
            thereshold_dim = len(E)
            print 'Team2 dim:%d' % thereshold_dim
            break

    min_score = 10000
    for i in range(100):
        model = KMeans(n_clusters=K, init='k-means++', max_iter=300, tol=0.0001).fit(X)
        if min_score > model.score(X):
            min_score = model.score(X)
            labels_Team2 = model.labels_
    print min_score

    pca = PCA(n_components = 2)
    pca.fit(BoF_Team2)
    X = pca.transform(BoF_Team2)
    for k in range(K):
        labels_Team2_ind = np.where(labels_Team2 == k)[0]
        plt.scatter(X[labels_Team2_ind,0], X[labels_Team2_ind,1], color=C[k])
    plt.legend(['C0','C1','C2','C3','C4'], loc=4)

    plt.title('Team2_PCA_kmeans')
    plt.savefig('Seq_Team2/Team2_PCA_kmeans.png')
    plt.show()
    plt.close()
    np.savetxt('Seq_Team2/labels_Team2.csv', labels_Team2, delimiter=',')

    print 'time:%f' % (time()-t0)
    return labels_Team1, labels_Team2

def Visualize_tactical_pattern():
#--kmeansで出力されたラベルに基づいて攻撃パターンを色塗り--

    labels_Team1 = np.loadtxt('Seq_Team1/labels_Team1.csv', delimiter=',')
    labels_Team2 = np.loadtxt('Seq_Team2/labels_Team2.csv', delimiter=',')

    x_period1 = np.arange(period1_end_re_t + 1)
    y0_period1 = np.zeros(period1_end_re_t + 1)

    x_period2 = np.arange(period2_end_re_t + 1)
    y0_period2 = np.zeros(period2_end_re_t + 1)

    x_period3 = np.arange(period3_end_re_t + 1)
    y0_period3 = np.zeros(period3_end_re_t + 1)

    x_period4 = np.arange(period4_end_re_t + 1)
    y0_period4 = np.zeros(period4_end_re_t + 1)

    #--Team1--
    fig = plt.figure(figsize=(16,4))
    plt.subplots_adjust(hspace=1.5)


    for k in range(K):
        Y_period1_Team1_of = np.copy(y0_period1)
        Y_period2_Team1_of = np.copy(y0_period2)
        Y_period3_Team1_of = np.copy(y0_period3)
        Y_period4_Team1_of = np.copy(y0_period4)

        for i in range(N_Team1_of):
            if labels_Team1[i] == k:
                S = Seq_Team1_of[i]
                start_t = S[0,0]
                start_re_t = S[0,1]
                end_re_t = S[len(S) - 1,1]
                period = int(end_re_t - start_re_t)

                if start_t < period1_end:
                    for j in range(period):
                        Y_period1_Team1_of[start_re_t + j] = 1.0
                elif period2_start < start_t and start_t < period2_end:
                    for j in range(period):
                        Y_period2_Team1_of[start_re_t + j] = 1.0
                elif period3_start < start_t and start_t < period3_end:
                    for j in range(period):
                        Y_period3_Team1_of[start_re_t + j] = 1.0
                elif period4_start < start_t and start_t < period4_end:
                    for j in range(period):
                        Y_period4_Team1_of[start_re_t + j] = 1.0

        plt.subplot(4, 1, 1)        
        plt.fill_between(x_period1, y0_period1, Y_period1_Team1_of, \
                         edgecolor = C[k], facecolor = C[k])

        plt.subplot(4, 1, 2)
        plt.fill_between(x_period2, y0_period2, Y_period2_Team1_of, \
                         edgecolor = C[k], facecolor = C[k])

        plt.subplot(4, 1, 3)
        plt.fill_between(x_period3, y0_period3, Y_period3_Team1_of, \
                         edgecolor = C[k], facecolor = C[k])

        plt.subplot(4, 1, 4)
        plt.fill_between(x_period4, y0_period4, Y_period4_Team1_of, \
                         edgecolor = C[k], facecolor = C[k])

    plt.subplot(4, 1, 1)        
    tempX = np.array(shot_period1_Team1_re_t)
    tempY = np.ones(len(shot_period1_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period1_Team1_re_t)
    tempY = np.ones(len(shot_success_period1_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period1_start_re_t, period1_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_period1_offense')

    plt.subplot(4, 1, 2)        
    tempX = np.array(shot_period2_Team1_re_t)
    tempY = np.ones(len(shot_period2_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period2_Team1_re_t)
    tempY = np.ones(len(shot_success_period2_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period2_start_re_t, period2_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_period2_offense')

    plt.subplot(4, 1, 3)        
    tempX = np.array(shot_period3_Team1_re_t)
    tempY = np.ones(len(shot_period3_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period3_Team1_re_t)
    tempY = np.ones(len(shot_success_period3_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period2_start_re_t, period2_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_period3_offense')

    plt.subplot(4, 1, 4)        
    tempX = np.array(shot_period4_Team1_re_t)
    tempY = np.ones(len(shot_period4_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period4_Team1_re_t)
    tempY = np.ones(len(shot_success_period4_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period2_start_re_t, period2_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_period4_offense')

    plt.savefig('Seq_Team1/Vis_tactical_pattern_Team1.png')
    plt.show()
    plt.close()

    #--Team2--
    fig = plt.figure(figsize=(16,4))
    plt.subplots_adjust(hspace=1.5)


    for k in range(K):
        Y_period1_Team2_of = np.copy(y0_period1)
        Y_period2_Team2_of = np.copy(y0_period2)
        Y_period3_Team2_of = np.copy(y0_period3)
        Y_period4_Team2_of = np.copy(y0_period4)

        for i in range(N_Team2_of):
            if labels_Team2[i] == k:
                S = Seq_Team2_of[i]
                start_t = S[0,0]
                start_re_t = S[0,1]
                end_re_t = S[len(S) - 1,1]
                period = int(end_re_t - start_re_t)

                if start_t < period1_end:
                    for j in range(period):
                        Y_period1_Team2_of[start_re_t + j] = 1.0
                elif period2_start < start_t and start_t < period2_end:
                    for j in range(period):
                        Y_period2_Team2_of[start_re_t + j] = 1.0
                elif period3_start < start_t and start_t < period3_end:
                    for j in range(period):
                        Y_period3_Team2_of[start_re_t + j] = 1.0
                elif period4_start < start_t and start_t < period4_end:
                    for j in range(period):
                        Y_period4_Team2_of[start_re_t + j] = 1.0

        plt.subplot(4, 1, 1)        
        plt.fill_between(x_period1, y0_period1, Y_period1_Team2_of, \
                         edgecolor = C[k], facecolor = C[k])

        plt.subplot(4, 1, 2)
        plt.fill_between(x_period2, y0_period2, Y_period2_Team2_of, \
                         edgecolor = C[k], facecolor = C[k])

        plt.subplot(4, 1, 3)
        plt.fill_between(x_period3, y0_period3, Y_period3_Team2_of, \
                         edgecolor = C[k], facecolor = C[k])

        plt.subplot(4, 1, 4)
        plt.fill_between(x_period4, y0_period4, Y_period4_Team2_of, \
                         edgecolor = C[k], facecolor = C[k])


    plt.subplot(4, 1, 1)        
    tempX = np.array(shot_period1_Team2_re_t)
    tempY = np.ones(len(shot_period1_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period1_Team2_re_t)
    tempY = np.ones(len(shot_success_period1_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period1_start_re_t, period1_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_period1_offense')

    plt.subplot(4, 1, 2)        
    tempX = np.array(shot_period2_Team2_re_t)
    tempY = np.ones(len(shot_period2_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period2_Team2_re_t)
    tempY = np.ones(len(shot_success_period2_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period2_start_re_t, period2_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_period2_offense')

    plt.subplot(4, 1, 3)        
    tempX = np.array(shot_period3_Team2_re_t)
    tempY = np.ones(len(shot_period3_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period3_Team2_re_t)
    tempY = np.ones(len(shot_success_period3_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period3_start_re_t, period3_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_period3_offense')

    plt.subplot(4, 1, 4)        
    tempX = np.array(shot_period4_Team2_re_t)
    tempY = np.ones(len(shot_period4_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(shot_success_period4_Team2_re_t)
    tempY = np.ones(len(shot_success_period4_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=65, edgecolor = 'black', facecolor = 'none')
    plt.axis([period4_start_re_t, period4_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_period4_offense')

    plt.savefig('Seq_Team2/Vis_tactical_pattern_Team2.png')
    plt.show()
    plt.close()


def Cluster_analysis():
#--kmeansで出力されたクラスタの平均など分析--

    commands.getoutput("rm -r Seq_Team1/Seq_C*")
    commands.getoutput("rm -r Seq_Team2/Seq_C*")

    commands.getoutput("mkdir Seq_Team1/Seq_C0")
    commands.getoutput("mkdir Seq_Team1/Seq_C1")
    commands.getoutput("mkdir Seq_Team1/Seq_C2")
    commands.getoutput("mkdir Seq_Team1/Seq_C3")
    commands.getoutput("mkdir Seq_Team1/Seq_C4")

    commands.getoutput("mkdir Seq_Team2/Seq_C0")
    commands.getoutput("mkdir Seq_Team2/Seq_C1")
    commands.getoutput("mkdir Seq_Team2/Seq_C2")
    commands.getoutput("mkdir Seq_Team2/Seq_C3")
    commands.getoutput("mkdir Seq_Team2/Seq_C4")

    labels_Team1 = np.loadtxt('Seq_Team1/labels_Team1.csv', delimiter=',')
    labels_Team2 = np.loadtxt('Seq_Team2/labels_Team2.csv', delimiter=',')

    f1 = open('Seq_Team1/Out_Team1.csv', 'w')
    csvWriter_Team1 = csv.writer(f1)
    f1_first = open('Seq_Team1/Out_Team1_first_half.csv', 'w')
    csvWriter_Team1_first = csv.writer(f1_first)
    f1_last = open('Seq_Team1/Out_Team1_last_half.csv', 'w')
    csvWriter_Team1_last = csv.writer(f1_last)

    f2 = open('Seq_Team2/Out_Team2.csv', 'w')
    csvWriter_Team2 = csv.writer(f2)
    f2_first = open('Seq_Team2/Out_Team2_first_half.csv', 'w')
    csvWriter_Team2_first = csv.writer(f2_first)
    f2_last = open('Seq_Team2/Out_Team2_last_half.csv', 'w')
    csvWriter_Team2_last = csv.writer(f2_last)

    for k in range(K):
        index = np.where(labels_Team1 == k)[0]

        #--位置情報の可視化--
        for i in range(len(index)):
            n = index[i]
            S = Seq_Team1_of[n]
            x1 = S[:,5]
            x2 = S[:,6]
            d = np.vstack([x1,x2]).T
            if i == 0:
                D = d
            else:
                D = np.vstack([D,d])
                
        X, Y = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        kernel = gaussian_kde(D.T,kde_width)
        Z = np.reshape(kernel(positions).T, X.shape)
        plt.imshow(np.rot90(Z), extent=[xmin, xmax, ymin, ymax])
        plt.scatter(D[:,0],D[:,1], edgecolor='grey',facecolor='grey', s = 10)
        plt.title('Team1_Cluster' + str(k) + '_location')
        plt.savefig('Seq_Team1/Cluster' + str(k) + '_location_Team1.png')
        plt.close()

        #--プレイヤーグラフの可視化--
        N_player = len(player1_dic)
        M = np.zeros([N_player, N_player])
        for i in range(len(index)):
            n = index[i]
            S = Seq_Team1_of[n]

            Pass_Series = S[:,3]
            for i in range(len(Pass_Series) - 1):
                now_p_ind = Pass_Series[i]
                next_p_ind = Pass_Series[i+1]

                now_p = player1_dic[now_p_ind]
                next_p = player1_dic[next_p_ind]
                M[now_p, next_p] += 1


        plt.pcolor(M, cmap=plt.cm.Blues)   
        plt.title('Team1_PlayerGraph_Cluster' + str(k))
        plt.savefig('Seq_Team1/Player_Graph_Cluster' + str(k) + '_Team1.png')
        plt.close()

        #-各クラスタのボール軌跡データ作成
        for i in range(len(index)):
            n = index[i]
            S = Seq_Team1_of[n]
            timing = int(S[0,1])
            team = int(S[0,2])
            action = int(S[np.shape(S)[0] - 1, 4])
            X = S[:,5]
            Y = S[:,6]
            fig = plt.figure()
            if action == 15:
                plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, \
                           scale_units='xy', angles='xy', scale=1, color='red')
            else:
                plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, \
                           scale_units='xy', angles='xy', scale=1, color='darkcyan')
            plt.axis([0, 600, 0, 330])
            plt.savefig('Seq_Team1/Seq_C' + str(k) + '/Seq_Team1_of'+'_no'+str(n)+'_t'+str(timing)+'C' + str(k) + '.png')
            plt.close()
            
        #--可視化ツール用データ出力--
        for i in range(len(index)):
            flag = 0
            n = index[i]
            S = Seq_Team1_of[n]
            #start_time = S[0,0]
            start_time = S[0,0] - 1.0
            if start_time < period3_start:
                flag = 1#前半ならflag=1

            Hour = int(start_time / 3600)
            Minute = int((start_time - Hour * 3600) / 60)
            Second = int((start_time - Hour * 3600 - Minute * 60))
            Epsilon = int((start_time - Hour * 3600 - Minute * 60 - Second) * 10 ** 3)
            start_time = str(Hour) + ':' + str(Minute) + ':' + str(Second) + '.' + str(Epsilon)
            end_time_posi = np.shape(S)[0] - 1
            #end_time = S[end_time_posi,0]
            end_time = S[end_time_posi,0] + 1.0
            Hour = int(end_time / 3600)
            Minute = int((end_time - Hour * 3600) / 60)
            Second = int((end_time - Hour * 3600 - Minute * 60))
            Epsilon = int((end_time - Hour * 3600 - Minute * 60 - Second) * 10 ** 3)
            end_time = str(Hour) + ':' + str(Minute) + ':' + str(Second) + '.' + str(Epsilon)
            line = [str(k+1),str(start_time),str(end_time)]
            csvWriter_Team1.writerow(line)
            if flag == 1:
                csvWriter_Team1_first.writerow(line)
            if flag == 0:
                csvWriter_Team1_last.writerow(line)

        #--位置情報の可視化--
        index = np.where(labels_Team2 == k)[0]
        for i in range(len(index)):
            n = index[i]
            S = Seq_Team2_of[n]
            x1 = S[:,5]
            x2 = S[:,6]
            d = np.vstack([x1,x2]).T
            if i == 0:
                D = d
            else:
                D = np.vstack([D,d])

        X, Y = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        kernel = gaussian_kde(D.T,kde_width)
        Z = np.reshape(kernel(positions).T, X.shape)
        plt.imshow(np.rot90(Z), extent=[xmin, xmax, ymin, ymax])
        plt.scatter(D[:,0],D[:,1], edgecolor='grey',facecolor='grey', s = 10)
        plt.title('Team2_Cluster' + str(k) + '_location')
        plt.savefig('Seq_Team2/Cluster' + str(k) + '_location_Team2.png')
        plt.close()

        #--プレイヤーグラフの可視化--
        N_player = len(player2_dic)
        M = np.zeros([N_player, N_player])
        for i in range(len(index)):
            n = index[i]
            S = Seq_Team2_of[n]

            Pass_Series = S[:,3]
            for i in range(len(Pass_Series) - 1):
                now_p_ind = Pass_Series[i]
                next_p_ind = Pass_Series[i+1]

                now_p = player2_dic[now_p_ind]
                next_p = player2_dic[next_p_ind]
                M[now_p, next_p] += 1


        plt.pcolor(M, cmap=plt.cm.Blues)   
        plt.title('Team2_PlayerGraph_Cluster' + str(k))
        plt.savefig('Seq_Team2/Player_Graph_Cluster' + str(k) + '_Team2.png')
        plt.close()

        #-各クラスタのボール軌跡データ作成
        for i in range(len(index)):
            n = index[i]
            S = Seq_Team1_of[n]
            timing = int(S[0,1])
            team = int(S[0,2])
            action = int(S[np.shape(S)[0] - 1, 4])
            X = S[:,5]
            Y = S[:,6]
            fig = plt.figure()
            if action == 15:
                plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, \
                           scale_units='xy', angles='xy', scale=1, color='red')
            else:
                plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, \
                           scale_units='xy', angles='xy', scale=1, color='darkcyan')
            plt.axis([0, 600, 0, 330])
            plt.savefig('Seq_Team2/Seq_C' + str(k) + '/Seq_Team2_of'+'_no'+str(n)+'_t'+str(timing)+'C' + str(k) + '.png')
            plt.close()

        #--可視化ツール用データ出力--
        for i in range(len(index)):
            flag = 0
            n = index[i]
            S = Seq_Team2_of[n]
            #start_time = S[0,0]
            start_time = S[0,0] - 1.0
            if start_time < period3_start:
                flag = 1#前半ならflag=1

            Hour = int(start_time / 3600)
            Minute = int((start_time - Hour * 3600) / 60)
            Second = int((start_time - Hour * 3600 - Minute * 60))
            Epsilon = int((start_time - Hour * 3600 - Minute * 60 - Second) * 10 ** 3)
            start_time = str(Hour) + ':' + str(Minute) + ':' + str(Second) + '.' + str(Epsilon)
            end_time_posi = np.shape(S)[0] - 1
            #end_time = S[end_time_posi,0]
            end_time = S[end_time_posi,0] + 1.0
            Hour = int(end_time / 3600)
            Minute = int((end_time - Hour * 3600) / 60)
            Second = int((end_time - Hour * 3600 - Minute * 60))
            Epsilon = int((end_time - Hour * 3600 - Minute * 60 - Second) * 10 ** 3)
            end_time = str(Hour) + ':' + str(Minute) + ':' + str(Second) + '.' + str(Epsilon)
            line = [str(k+1),str(start_time),str(end_time)]
            csvWriter_Team2.writerow(line)
            if flag == 1:
                csvWriter_Team2_first.writerow(line)
            if flag == 0:
                csvWriter_Team2_last.writerow(line)

    f1.close()
    f1_first.close()
    f1_last.close()

    f2.close()
    f2_first.close()
    f2_last.close()    

#--main--
input()
#データ読み込み

Seq_Team_of()
#オフェンス時のボール軌跡データ作成

BoF_Team1, BoF_Team2 = make_BoF()
#パス系列と量子化された位置情報を含むBag-of-Feature作成
#pdb.set_trace()

Possession()
#各チームのボール保持時間とシュートのタイミングを描画
#pdb.set_trace()

labels_Team1, labels_Team2 = Clustering(BoF_Team1, BoF_Team2)
#BoFを入力にして攻撃パターンをクラスタリング
#pdb.set_trace()

Visualize_tactical_pattern()
#kmeansで出力されたラベルに基づいて攻撃パターンを色塗り

Cluster_analysis()
#kmeansで出力されたクラスタの平均など分析

pdb.set_trace()
