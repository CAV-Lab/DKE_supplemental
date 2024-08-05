import os
import pandas as pd
import csv
import numpy as np
import datetime
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import copy
from collections import OrderedDict
from IPython import embed


def calculateMovecount():

    data = pd.read_csv(filepath, encoding='utf-8')
    results = data['UserID'].value_counts()
    df_data_counts = pd.DataFrame(results)

    id_list = df_data_counts.index.values.tolist()
    id_list_str = [str(x) for x in id_list]
    counts_list = df_data_counts['UserID'].tolist()
    counts_list_str = [str(x) for x in counts_list]
    results_dic = dict(zip(id_list_str,counts_list_str))
    return results_dic

def interpretTrait(sum_dic, o_mean,o_std,j,o_high_IDs,o_low_IDs,o_average_IDs):

    for uid in sum_dic:
        o_s = sum_dic[uid][j]
        if o_s < o_mean- 0.5 * o_std:
            o_low_IDs.append(uid)
        if o_s > o_mean + 0.5 * o_std:
            o_high_IDs.append(uid)
        if o_mean - 0.5 * o_std <= o_s <= o_mean + 0.5 * o_std:
            o_average_IDs.append(uid)

def countPerTrait(o_high_IDs,o_low_IDs, trait):
    high_count=[]
    low_count=[]
    for uid in results_dic:
        if uid in o_high_IDs:
            high_count.append(int(results_dic[uid]))
        if uid in o_low_IDs:
            low_count.append(int(results_dic[uid]))

    high_mean=sum(high_count)/len(high_count)
    high_SD=np.std(high_count)

    print(trait + " high mean: " + str(high_mean))
    print(trait + " high sd: " + str(high_SD))

    low_mean = sum(low_count) / len(low_count)
    low_SD = np.std(low_count)
    print(trait + " low mean: " + str(low_mean))
    print(trait + " low sd: " + str(low_SD))

def calculateTrait():
    que = [
    'Q1',
    'Q3_1', 'Q3_2', 'Q3_3', 'Q3_4', 'Q3_5',
    'Q4_1', 'Q4_2', 'Q4_3', 'Q4_4', 'Q4_5',
    'Q5',
    'Q6_1', 'Q6_2', 'Q6_3', 'Q6_4', 'Q6_5',
    'Q7_1', 'Q7_2', 'Q7_3', 'Q7_4', 'Q7_5',
    'Q8',
    'Q9_1', 'Q9_2', 'Q9_3', 'Q9_4', 'Q9_5']

    plus = ['Q3_1', 'Q3_2', 'Q3_3', 'Q3_4', 'Q3_5', 'Q6_1', 'Q6_2', 'Q6_3', 'Q6_4']
    minus = ['Q4_1', 'Q4_2', 'Q4_3', 'Q4_4', 'Q4_5',  'Q6_5', 'Q7_1', 'Q7_2', 'Q7_3', 'Q7_4', 'Q7_5']
    choice = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']
    plus_score = [1, 2, 3, 4, 5]
    minus_score = [5, 4, 3, 2, 1]
    Openness = ['Q3_5', 'Q4_5', 'Q6_5', 'Q7_5']
    Conscientiousness = ['Q3_3', 'Q4_3', 'Q6_3', 'Q7_3']
    Extraversion = ['Q3_1', 'Q4_1', 'Q6_1', 'Q7_1']
    Agreeableness = ['Q3_2', 'Q4_2', 'Q6_2', 'Q7_2']
    Neuroticism = ['Q3_4', 'Q4_4', 'Q6_4', 'Q7_4']


    est_path = './data/Personality_39 ID.csv'
    with open(est_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        traits_rows = []
        for row in reader:
            select_row = {}
            for key in row.keys():
                if key in que:
                    select_row[key] = row[key]
            traits_rows.append(select_row)

    traits_rows = traits_rows[2:]

    score = {}  # score = {id: [1,2,3,4],[],[],[],[]}
    for row in traits_rows:
        if row['Q1'] in completeUids:  #finished the puzzle and passed the attention check
            per_score = [[], [], [], [], []]  # [O,C,E,A,N]
            for key in row.keys():
                if key in plus:
                    if key in Openness:
                        per_score[0].append(plus_score[choice.index(row[key])])
                    if key in Conscientiousness:
                        per_score[1].append(plus_score[choice.index(row[key])])
                    if key in Extraversion:
                        per_score[2].append(plus_score[choice.index(row[key])])
                    if key in Agreeableness:
                        per_score[3].append(plus_score[choice.index(row[key])])
                    if key in Neuroticism:
                        per_score[4].append(plus_score[choice.index(row[key])])

                if key in minus:
                    if key in Openness:
                        per_score[0].append(minus_score[choice.index(row[key])])
                    if key in Conscientiousness:
                        per_score[1].append(minus_score[choice.index(row[key])])
                    if key in Extraversion:
                        per_score[2].append(minus_score[choice.index(row[key])])
                    if key in Agreeableness:
                        per_score[3].append(minus_score[choice.index(row[key])])
                    if key in Neuroticism:
                        per_score[4].append(minus_score[choice.index(row[key])])

            score[row['Q1']] = per_score

    # -----sum trait score------:
    sum_dic = {}  # sum_dic = {id: [1,2,3,4,5]}
    for item in score.items():
        sum_score = []
        sum_score.append(sum(item[1][0]))
        sum_score.append(sum(item[1][1]))
        sum_score.append(sum(item[1][2]))
        sum_score.append(sum(item[1][3]))
        sum_score.append(sum(item[1][4]))

        sum_dic[item[0]] = sum_score

    # ---- interpret score----
    o = []
    c = []
    e = []
    a = []
    n = []
    for item in sum_dic.items():
        o.append(item[1][0])
        c.append(item[1][1])
        e.append(item[1][2])
        a.append(item[1][3])
        n.append(item[1][4])

    o_mean = np.mean(o)
    c_mean = np.mean(c)
    e_mean = np.mean(e)
    a_mean = np.mean(a)
    n_mean = np.mean(n)

    o_std = np.std(o)
    c_std = np.std(c)
    e_std = np.std(e)
    a_std = np.std(a)
    n_std = np.std(n)

    o_high_IDs = []
    o_low_IDs = []
    o_average_IDs = []
    interpretTrait(sum_dic, o_mean, o_std, 0, o_high_IDs, o_low_IDs, o_average_IDs)

    c_high_IDs = []
    c_low_IDs = []
    c_average_IDs = []
    interpretTrait(sum_dic, c_mean, c_std, 1, c_high_IDs, c_low_IDs, c_average_IDs)

    e_high_IDs = []
    e_low_IDs = []
    e_average_IDs = []
    interpretTrait(sum_dic, e_mean, e_std, 2, e_high_IDs, e_low_IDs, e_average_IDs)

    a_high_IDs = []
    a_low_IDs = []
    a_average_IDs = []
    interpretTrait(sum_dic, a_mean, a_std, 3, a_high_IDs, a_low_IDs, a_average_IDs)

    n_high_IDs = []
    n_low_IDs = []
    n_average_IDs = []
    interpretTrait(sum_dic, n_mean, n_std, 4, n_high_IDs, n_low_IDs, n_average_IDs)

    #move counts per trait:
    countPerTrait(o_high_IDs, o_low_IDs, 'o')
    countPerTrait(c_high_IDs, c_low_IDs, 'c')
    countPerTrait(e_high_IDs, e_low_IDs, 'e')
    countPerTrait(a_high_IDs, a_low_IDs, 'a')
    countPerTrait(n_high_IDs, n_low_IDs, 'n')

    return o_high_IDs, o_low_IDs, c_high_IDs, c_low_IDs, e_high_IDs, e_low_IDs, a_high_IDs, a_low_IDs, n_high_IDs, n_low_IDs, sum_dic

def sort_to_dict(lst):

    dict_ = {}
    for it in lst:
        dict_[it[0]]=it[1]

    return dict_

def plot_DK(j, title, conutsPercentile, actual_timeSpent, movesPercentile,timePercentile, abilityPercentile):

    font_size = 20


    users_id = ['Bottom Quartile', '2nd Quartile', '3rd Quartile', 'Top Quartile']
    x = len(users_id)
    plt.figure(figsize=(9,9))
    plt.ylabel("Percentile",  fontsize=font_size, weight='bold')
    plt.ylim(0,100)
    plt.xticks(fontsize=16, weight='bold')
    plt.yticks(fontsize=font_size)

    if j == 0:
        plt.plot(users_id, conutsPercentile, 'o', ls='-', linewidth=4, label="Actual Move Count")
    if j == 1:
        plt.plot(users_id, actual_timeSpent, 'o', ls='-', linewidth=4, label="Actual Time Spent")
    plt.plot(users_id, movesPercentile, 's', ls=':',linewidth=4,label="Perceived Move Count")
    plt.plot(users_id, timePercentile, '^', ls=':',linewidth=4,label="Perceived Time Spent")
    plt.plot(users_id, abilityPercentile, 's', ls=':', linewidth=4,label="Perceived Reasoning Ability")

    plt.legend(fontsize=font_size, loc=4)
    plt.grid(True)
    root = 'dkResult'
    plt.tight_layout()
    plt.savefig(os.path.join(root,title + " .jpg"), dpi=300)
    plt.show()

def dkResult():
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = [row for row in reader]

    time_dic = {}
    for i, row in enumerate(rows):
        if i == 0:
            continue
        userId = row[0]

        if userId not in time_dic:
            time_dic[userId] = []
            time_dic[userId].append(row[-1])
        else:
            time_dic[userId].append(row[-1])

    timeSpent_dic = {}
    for uid in time_dic:
        timeList = time_dic[uid]
        start = timeList[0]
        end = timeList[-1]
        startTime = datetime.datetime.strptime(start, "%H:%M:%S")
        endTime = datetime.datetime.strptime(end, "%H:%M:%S")
        timeSpent = (endTime - startTime).seconds / 60  # min
        timeSpent_dic[uid] = timeSpent


    df = pd.read_csv("data/Puzzle game_39 ID.csv")
    data = df[['Q1', 'Q10_1', 'Q11_1', 'Q12_1']]

    data = data.iloc[2:]
    dic_estimation_survey = data.set_index('Q1').T.to_dict('list')


    dic = {}  # dic={uid:[perceived moves percentile, perceived time spent precentile, perceived reasoning ability, actual counts, actual timeSpent(min)]}
    for uid in results_dic:
        for key in dic_estimation_survey:
            if uid == key:
                dic[uid] = []
                dic[uid].append(int(dic_estimation_survey[key][0]))
                dic[uid].append(int(dic_estimation_survey[key][1]))
                dic[uid].append(int(dic_estimation_survey[key][2]))
                dic[uid].append(int(results_dic[uid]))

        for id in timeSpent_dic:
            if uid == id and id in dic_estimation_survey:
                dic[uid].append(timeSpent_dic[id])

    # -----actual counts percentile---------#
    # sorted based on actual counts
    dic_sort_counts = sorted(dic.items(), key=lambda item: item[1][3])
    dic_sort_counts = sort_to_dict(dic_sort_counts)

    top_group_IDs = [str(key) for key in dic_sort_counts.keys()][: int((len(dic_sort_counts) / 4))]
    third_group_IDs = [str(key) for key in dic_sort_counts.keys()][
                      int((len(dic_sort_counts) / 4)):int(2 * len(dic_sort_counts) / 4)]
    second_group_IDs = [str(key) for key in dic_sort_counts.keys()][
                       int(2 * len(dic_sort_counts) / 4):-int((len(dic_sort_counts)) / 4)]
    bottom_group_IDs = [str(key) for key in dic_sort_counts.keys()][-int(
        (len(dic_sort_counts)) / 4):]

    # actual percentile of each user
    leng = len(dic.keys())
    S = range(leng)
    real_per = []
    for i, s in enumerate(S):
        real_per.append(stats.percentileofscore(S,s))

    # actual percentile of each quartile
    conutsPercentile = np.array([0, 0, 0, 0])
    num = np.array([0, 0, 0, 0])

    for i, s in enumerate(S):

        if i < leng / 4:
            conutsPercentile[0] += stats.percentileofscore(S, s)
            num[0] += 1

        elif leng / 4 <= i < 2 * leng / 4:
            conutsPercentile[1] += stats.percentileofscore(S, s)
            num[1] += 1

        elif 2 * leng / 4 <= i < 3 * leng / 4:
            conutsPercentile[2] += stats.percentileofscore(S, s)
            num[2] += 1

        elif 3 * leng / 4 <= i:
            conutsPercentile[3] += stats.percentileofscore(S, s)
            num[3] += 1

    conutsPercentile = conutsPercentile / num

    #compute actual time spent percentile:
    id_timeSpent = {}
    timeSpent_value = []
    timeSpent_top = []
    timeSpent_bot = []
    for key in dic_sort_counts:
        id_timeSpent[key] = dic_sort_counts[key][4]
        timeSpent_value.append(dic_sort_counts[key][4])
        if key in top_group_IDs:
            timeSpent_top.append(dic_sort_counts[key][4])
        if key in bottom_group_IDs:
            timeSpent_bot.append(dic_sort_counts[key][4])

    dic_sort_countNum = copy.deepcopy(dic_sort_counts)

    # time spent across bot and top:
    mean_timeSpent_top = sum(timeSpent_top) / len(timeSpent_top)
    sd_timeSpent_top = np.std(timeSpent_top)

    mean_timeSpent_bot = sum(timeSpent_bot) / len(timeSpent_bot)
    sd_timeSpent_bot = np.std(timeSpent_bot)

    timeSpent_percentile = [stats.percentileofscore(timeSpent_value, a) for a in timeSpent_value]

    for j, k in enumerate(dic_sort_counts):
        dic_sort_counts[k][4] = timeSpent_percentile[j]
        dic_sort_counts[k][3] = real_per[j]

    # define two extreme group by their counts:
    # --------Percevied Moves percentile------#
    moves_est = []
    for user in dic_sort_counts:
        moves_est.append(dic_sort_counts[user][0])
    movesPercentile, moves_error = est_percentile(num, leng, dic_sort_counts, moves_est)
    # [top, 3rd, 2nd, bottom] --> [bottom, 2nd, 3rd, top]
    movesPercentile = movesPercentile[::-1]
    moves_error = moves_error[::-1]

    # --------Perceviced Time percentile------#
    time_est = []
    for user in dic_sort_counts:
        time_est.append(dic_sort_counts[user][1])
    timePercentile, time_error = est_percentile(num, leng, dic_sort_counts, time_est)
    timePercentile = timePercentile[::-1]
    time_error = time_error[::-1]

    # --------Percevied Ability percentile------#
    ability_est = []
    for user in dic_sort_counts:
        ability_est.append(dic_sort_counts[user][2])
    abilityPercentile, ability_error = est_percentile(num, leng, dic_sort_counts, ability_est)
    # [top, 3rd, 2nd, bottom] --> [bottom, 2nd, 3rd, top]
    abilityPercentile = abilityPercentile[::-1]
    ability_error = ability_error[::-1]

    # to define two extreme group by their time spent
    # --------Actual Time Spent percentile------#
    dic_sort_timeSpent = sorted(dic.items(), key=lambda item: item[1][4])
    dic_sort_timeSpent = sort_to_dict(dic_sort_timeSpent)

    # to get the list of IDs in top and bottom quartile based on actual time spent:
    top_group_IDs_timeSpent = [str(key) for key in dic_sort_timeSpent.keys()][: int((len(dic_sort_timeSpent) / 4))]
    bottom_group_IDs_timeSpent = [str(key) for key in dic_sort_timeSpent.keys()][-int((len(dic_sort_timeSpent)) / 4):]


    diff_top = []
    diff_bot = []
    for key, value in dic_sort_timeSpent.items():
        if key in top_group_IDs_timeSpent:
            diff_top.append(value[1] - value[-1])
        elif key in bottom_group_IDs_timeSpent:
            diff_bot.append(value[1] - value[-1])

    stat, p = stats.ttest_ind(diff_top, diff_bot)
    print('time spent ttest: statistics=%.3f, p=%.3f' % (stat, p))


    actual_time_Spent = []
    for user in dic_sort_timeSpent:
        actual_time_Spent.append(dic_sort_timeSpent[user][-1])

    actual_timeSpent_2, timeSpent_error_2 = est_percentile(num, leng, dic_sort_counts, actual_time_Spent)

    ## --------Percevied Count percentile------#
    moves_est = []
    for user in dic_sort_timeSpent:
        moves_est.append(dic_sort_timeSpent[user][0])
    movesPercentile_2, moves_error_2 = est_percentile(num, leng, dic_sort_counts,moves_est)
    movesPercentile_2 = movesPercentile_2[::-1]
    moves_error_2 = moves_error_2[::-1]

    # --------Percevied Time Spent percentile------#
    time_est = []
    for user in dic_sort_timeSpent:
        time_est.append(dic_sort_timeSpent[user][1])
    timePercentile_2, time_error_2 = est_percentile(num, leng, dic_sort_counts,time_est)
    timePercentile_2 = timePercentile_2[::-1]
    time_error_2 = time_error_2[::-1]

    # --------Percevied Ability percentile------#
    ability_est = []
    for user in dic_sort_timeSpent:
        ability_est.append(dic_sort_timeSpent[user][2])
    abilityPercentile_2, ability_error_2 = est_percentile(num, leng, dic_sort_counts,ability_est)
    abilityPercentile_2 = abilityPercentile_2[::-1]
    ability_error_2 = ability_error_2[::-1]

    title_1 = 'DK by Move Count 39 Ids'
    plot_DK(0, title_1, conutsPercentile, actual_timeSpent_2, movesPercentile,timePercentile, abilityPercentile)
    title_2 = 'DK by Time Spent 39 Ids'
    plot_DK(1, title_2, conutsPercentile, actual_timeSpent_2, movesPercentile_2, timePercentile_2, abilityPercentile_2)
    return dic_sort_counts, top_group_IDs,third_group_IDs,second_group_IDs, bottom_group_IDs, dic_sort_countNum

def est_percentile(num, leng, dic_sort_counts, est):
    est_percentile = np.array([0, 0, 0, 0])
    est_error = [[], [], [], []]

    for i, user in enumerate(dic_sort_counts):
        if i < leng / 4:
            est_percentile[0] += est[i]
            est_error[0].append(est[i])
        if leng / 4 <= i < 2 * leng / 4:
            est_percentile[1] += est[i]
            est_error[1].append(est[i])
        if 2 * leng / 4 <= i < 3 * leng / 4:
            est_percentile[2] += est[i]
            est_error[2].append(est[i])
        if 3 * leng / 4 <= i:
            est_percentile[3] += est[i]
            est_error[3].append(est[i])

    est_percentile = est_percentile / num

    return est_percentile, est_error

def calculateWidth(pointA, pointB, x, y, j):
        count = 1
        for k, p in enumerate(pointA):
            if (p == x and pointB[k] == y) or (p == y and pointB[k] == x):
                count += 1
        width = count
        return width

def get_coordinates(high_IDs, low_IDs, high, low):
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = [row for row in reader]
        for i, row in enumerate(rows):
            if i == 0:
                continue
            userId = row[0]
            if userId in high_IDs:
                if userId not in high:

                    high[userId] = [[], []]
                    high[userId][0].append([row[2], row[3]])
                    high[userId][1].append([row[4], row[5]])
                else:
                    high[userId][0].append([row[2], row[3]])
                    high[userId][1].append([row[4], row[5]])

            if userId in low_IDs:
                if userId not in low:
                    low[userId] = [[], []]
                    low[userId][0].append([row[2], row[3]])
                    low[userId][1].append([row[4], row[5]])
                else:
                    low[userId][0].append([row[2], row[3]])
                    low[userId][1].append([row[4], row[5]])
        return high, low

def plot_Path(dic, group, title):
    plt.figure()
    for uid in group:
        counts = dic_sort_countNum[uid][-2]

        pointA = []
        pointB = []
        d_from = dic[uid][0]
        d_to = dic[uid][1]

        for j, i_from in enumerate(d_from):
            i_to = d_to[j]
            x1 = i_from[0]
            y1 = i_from[1]
            x2 = i_to[0]
            y2 = i_to[1]
            y = [int(x1), int(x2)]
            x = [int(y1), int(y2)]

            width = calculateWidth(pointA, pointB, x, y, j)
            plt.plot(x, y, color='#bbb1b1', linewidth=width/counts*100)
            pointA.append(x)
            pointB.append(y)

    plt.xticks([])
    plt.yticks([])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    root = 'strategies'
    plt.savefig(os.path.join(root,title + " .jpg"), dpi=300)

    plt.show()

def MovementPathVis():
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = [row for row in reader]
    # ---- get all ID's from-to coordinates#
    dic = {}
    for i, row in enumerate(rows):
        if i == 0:
            continue
        userId = row[0]
        if userId not in dic:

            dic[userId] = [[], []]
            dic[userId][0].append([row[2], row[3]])
            dic[userId][1].append([row[4], row[5]])
        else:
            dic[userId][0].append([row[2], row[3]])
            dic[userId][1].append([row[4], row[5]])

    dic_40 = {'1657211007756',
              '1657210794863',
              '1657210780716',
              '1657210099426',
              '1657211552048',
              }

    dic_13_40 = {
        '1657210421904',
        '1657211180966',
        '1657214448186',
        '1657211073669',
        '1657211010904',
    }

    dic_12 = {'1657211427113',
              '1657211281011',
              '1657211301879',
              '1657212053902',
              '1657208318540',
              '1657211003033'}

    # ---- get from-to coordinates for different counts groups#
    dic_top = {}
    dic_third = {}
    dic_second = {}
    dic_bottom = {}

    dic_cor_40 = {}
    dic_cor_13_40 = {}
    dic_cor_12 = {}
    for i, row in enumerate(rows):
        if i == 0:
            continue
        userId = row[0]
        if userId in top_group_IDs:
            if userId not in dic_top:

                dic_top[userId] = [[], []]
                dic_top[userId][0].append([row[2], row[3]])
                dic_top[userId][1].append([row[4], row[5]])
            else:
                dic_top[userId][0].append([row[2], row[3]])
                dic_top[userId][1].append([row[4], row[5]])

        if userId in bottom_group_IDs:
            if userId not in dic_bottom:
                dic_bottom[userId] = [[], []]
                dic_bottom[userId][0].append([row[2], row[3]])
                dic_bottom[userId][1].append([row[4], row[5]])
            else:
                dic_bottom[userId][0].append([row[2], row[3]])
                dic_bottom[userId][1].append([row[4], row[5]])

        if userId in second_group_IDs:
            if userId not in dic_second:

                dic_second[userId] = [[], []]
                dic_second[userId][0].append([row[2], row[3]])
                dic_second[userId][1].append([row[4], row[5]])
            else:
                dic_second[userId][0].append([row[2], row[3]])
                dic_second[userId][1].append([row[4], row[5]])

        if userId in third_group_IDs:
            if userId not in dic_third:

                dic_third[userId] = [[], []]
                dic_third[userId][0].append([row[2], row[3]])
                dic_third[userId][1].append([row[4], row[5]])
            else:
                dic_third[userId][0].append([row[2], row[3]])
                dic_third[userId][1].append([row[4], row[5]])

        if userId in dic_40:
            if userId not in dic_cor_40:

                dic_cor_40[userId] = [[], []]
                dic_cor_40[userId][0].append([row[2], row[3]])
                dic_cor_40[userId][1].append([row[4], row[5]])
            else:
                dic_cor_40[userId][0].append([row[2], row[3]])
                dic_cor_40[userId][1].append([row[4], row[5]])

        if userId in dic_13_40:
            if userId not in dic_cor_13_40:

                dic_cor_13_40[userId] = [[], []]
                dic_cor_13_40[userId][0].append([row[2], row[3]])
                dic_cor_13_40[userId][1].append([row[4], row[5]])
            else:
                dic_cor_13_40[userId][0].append([row[2], row[3]])
                dic_cor_13_40[userId][1].append([row[4], row[5]])

        if userId in dic_12:
            if userId not in dic_cor_12:

                dic_cor_12[userId] = [[], []]
                dic_cor_12[userId][0].append([row[2], row[3]])
                dic_cor_12[userId][1].append([row[4], row[5]])
            else:
                dic_cor_12[userId][0].append([row[2], row[3]])
                dic_cor_12[userId][1].append([row[4], row[5]])

    plot_Path(dic, dic_top, 'Norm Top Group sequence 39 IDs')
    plot_Path(dic, dic_third, 'Norm third Group sequence 39 IDs')
    plot_Path(dic, dic_second, 'Norm second Group sequence 39 IDs')
    plot_Path(dic, dic_bottom, 'Norm bottom Group sequence 39 IDs')


    plot_Path(dic, dic_cor_40, 'Norm move 40+')
    plot_Path(dic, dic_cor_13_40, 'Norm move 13-40')
    plot_Path(dic, dic_cor_12, 'Norm move 12')

    # ---- get from-to coordinates for high/low score trait
    o_high = {}
    o_low = {}
    o_high, o_low = get_coordinates(o_high_IDs, o_low_IDs, o_high, o_low)

    c_high = {}
    c_low = {}
    c_high, c_low  = get_coordinates(c_high_IDs, c_low_IDs, c_high, c_low)

    e_high = {}
    e_low = {}
    e_high, e_low = get_coordinates(e_high_IDs, e_low_IDs, e_high, e_low)

    a_high = {}
    a_low = {}
    a_high, a_low = get_coordinates(a_high_IDs, a_low_IDs, a_high, a_low)

    n_high = {}
    n_low = {}
    n_high, n_low = get_coordinates(n_high_IDs, n_low_IDs, n_high, n_low)

    plot_Path(dic, o_high_IDs,'High Openness')
    plot_Path(dic, o_low_IDs,'Low Openness')
    plot_Path(dic, c_high_IDs,'High Conscientiousness')
    plot_Path(dic, c_low_IDs,'Low Conscientiousness')
    plot_Path(dic, e_high_IDs,'High Extraversion')
    plot_Path(dic, e_low_IDs,'Low Extraversion')
    plot_Path(dic, a_high_IDs,'High Agreeableness')
    plot_Path(dic, a_low_IDs,'Low Agreeableness')
    plot_Path(dic, n_high_IDs,'High Neuroticism')
    plot_Path(dic, n_low_IDs,'Low Neuroticism')

def traitPerformanceScatter(sum_dic, varb, title):
    traitsList = ['O', 'C', 'E', 'A', 'N']
    personalityName= ['Openness','Conscientiousness','Extraversion','Agreeableness','Neuroticism']

    if varb == 'diff':
        diffMoves = {}
        for key, value in dic_sort_counts.items():
            diffMoves[key] = value[0] - value[3]

        y_values = [diffMoves[key] for key in sum_dic.keys()]
    if varb == 'perceivedMove':
        y_values = [dic_sort_counts[key][0] for key in sum_dic.keys()]

    fig, axs = plt.subplots(1, 5, figsize=(25, 5))


    for i in range(len(next(iter(sum_dic.values())))):

        traitsScore_values = [value[i] for value in sum_dic.values()]

        #checkout the distribution of two variables:
        stat1, p1 = stats.shapiro(traitsScore_values)
        stat2, p2 = stats.shapiro(y_values)

        print('# ' + traitsList[i] + ': Check out if normal distribution:')
        print(traitsList[i] + ': Statistics=%.3f, p=%.3f' % (stat1, p1))
        print('DiffScore: Statistics=%.3f, p=%.3f' % (stat2, p2))

        correlation, p_value = pearsonr(traitsScore_values, y_values)

        print(f'Correlation between values in traits {traitsList[i]} and DiffScore: {correlation:.3f}, p-value: {p_value:.3f}')

        sns.regplot(x=traitsScore_values, y=y_values, ax=axs[i])
        axs[i].set_xlabel(personalityName[i] + 'Trait Score', fontsize=16)
        if varb == 'diff':
            axs[i].set_ylabel('Δ = EstimatedPercentile - ActualPercentile',
                          fontsize=13,)
        if varb == 'perceivedMove':
            axs[i].set_ylabel('Perceived Move Counts',
                          fontsize=13,)

        axs[i].set_title(f"r({len(traitsScore_values) - 2}) = {correlation:.3f}, p = {p_value:.3f}", fontsize=15)

    plt.tight_layout()
    root = 'personality'
    plt.savefig(os.path.join(root,title + " .jpg"), dpi=300)
    plt.show()

filepath = 'data/movement_39IDs.csv'
completeUids = ['1657208318540',
 '1657208506776',
 '1657209804853',
 '1657209818165',
 '1657209980224',
 '1657209998278',
 '1657209996421',
 '1657210099426',
 '1657210359013',
 '1657210461354',
 '1657210421904',
 '1657210556144',
 '1657210794863',
 '1657210780716',
 '1657210761245',
 '1657211073669',
 '1657210790770',
 '1657211207862',
 '1657211007756',
 '1657211180966',
 '1657211301879',
 '1657211281011',
 '1657210928959',
 '1657211228952',
 '1657211262958',
 '1657211050500',
 '1657211232699',
 '1657211427113',
 '1657211302464',
 '1657211010904',
 '1657211003033',
 '1657211552048',
 '1657212053902',
 '1657212459585',
 '1657213206472',
 '1657214138357',
 '1657214448186',
 '1657214996530',
 '1657557945041']

results_dic = calculateMovecount()
o_high_IDs, o_low_IDs, c_high_IDs, c_low_IDs, e_high_IDs, e_low_IDs, a_high_IDs, a_low_IDs, n_high_IDs, n_low_IDs, sum_dic = calculateTrait()
dic_sort_counts, top_group_IDs,third_group_IDs,second_group_IDs, bottom_group_IDs, dic_sort_countNum = dkResult()
MovementPathVis()
traitPerformanceScatter(sum_dic, 'diff', 'dif_traits_scatterplot')
traitPerformanceScatter(sum_dic, 'perceivedMove', 'perceivedMove_traits_scatterplot')



