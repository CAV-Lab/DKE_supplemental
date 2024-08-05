import json
import os
from datetime import datetime
from IPython import embed
import pandas as pd
from pandas import DataFrame
import csv
import copy
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn as sns
import cv2
from tqdm import tqdm
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
import statsmodels.stats.multitest as mt
from collections import Counter
import statsmodels.api as sm
import statsmodels.formula.api as smf



class ParsingLogs:

    def __init__(self, logfilepath, car_groundtruth_file, credit_groundtruth_file, sum_=30):
        self.logfilepath = logfilepath
        self.car_groundtruth_file = car_groundtruth_file
        self.credit_groundtruth_file = credit_groundtruth_file
        self.sum = sum_

        self.car_truth = self.get_groundtruth('car', self.car_groundtruth_file)
        self.credit_truth = self.get_groundtruth('credit', self.credit_groundtruth_file)

        with open(self.logfilepath, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

        self.Uids = ['6800865064849',
                     '2727686234398',
                     '7387239454636',
                     '8328444589173',
                     '4160540621495',
                     '9630166401988',
                     '6626244448331',
                     '9087541321377',
                     '2150593473485',
                     '5126037822081',
                     '3151707465034',
                     '7617571087558',
                     '6697638286263',
                     '3738511881824',
                     '5670927203125',
                     '0633911296732',
                     '9840433293820',
                     '7102599943132',
                     '8744028530856',
                     '3620255537370',
                     '5400157466432',
                     '6166422512463',
                     '4068349319560',
                     '7919508456562',
                     '5254278982816',
                     '5108681096432',
                     '6013092148633',
                     '5281930515725',
                     '2243602718222',
                     '8871662858566',
                     '8423098739747',
                     '3591396471401',
                     '8170508262885',
                     '3876484308826',
                     '4261067049419',
                     '3762146379222',
                     '8285630915797',
                     '0949235200363',
                     '5085296258955',
                     '5542722596220',
                     '9702923591985',
                     '8412785301271',
                     '5647917213340',
                     '3963422565415',
                     '6983780868617',
                     '9343927106523']
        items = self.dateTime(self.data)

        userlabel_car, acc_car, acc_per_dic_car, lenLabel_car = self.getuserlabels_acc('car', self.car_truth)
        userlabel_credit, acc_credit, acc_per_dic_credit, lenLabel_credit = self.getuserlabels_acc('credit', self.credit_truth)

        # Exclude the outliers whose performance falls outside of 3 SD from the mean
        userlabel_car, acc_car, acc_per_dic_car = self.exOutliers(userlabel_car, acc_car, acc_per_dic_car)
        userlabel_credit, acc_credit, acc_per_dic_credit = self.exOutliers(userlabel_credit, acc_credit, acc_per_dic_credit)


        # car_interactions = self.interaction_csv('car')
        # credit_interactions = self.interaction_csv('credit')

        # Compute elapsedTime:
        filePath_car = 'interaction_Car.csv'
        filePath_credit = 'interaction_Credit.csv'
        inter_car_df = pd.read_csv(filePath_car)
        inter_credit_df = pd.read_csv(filePath_credit)

        self.inter_car_df = self.CombineXY(inter_car_df)
        self.inter_credit_df = self.CombineXY(inter_credit_df)

        # self.calElapsedTime(self.inter_car_df, 'interaction_elapsed_Car')
        # self.calElapsedTime(self.inter_credit_df, 'interaction_elapsed_Credit')


        self.est_rows = self.postsruveyRows(postSurvey)
        # actual_estimation_car = {uid:[actual acc, actual per, estimated acc, estimated ranking, estimated reasoningAbility]}
        self.actual_estimation_car, self.differenceScore_car, self.familiarity_car, self.estAcc_car = self.get_acc_est(acc_per_dic_car, 'Q12_2','Q13_1', 'Q20_1', 'Q11')
        self.actual_estimation_credit, self.differenceScore_credit, self.familiarity_credit, self.estAcc_credit = self.get_acc_est(acc_per_dic_credit, 'Q17_1', 'Q18_1', 'Q20_1', 'Q16')

        self.sort_actual_estimation_car = self.sort_to_dict(sorted(self.actual_estimation_car.items(), key=lambda item: item[1][0]))
        self.sort_actual_estimation_credit = self.sort_to_dict(sorted(self.actual_estimation_credit.items(), key=lambda item: item[1][0]))

        self.bot_userID_car, self.top_userID_car = self.getUid_BotTop(self.sort_actual_estimation_car)
        self.bot_userID_credit, self.top_userID_credit = self.getUid_BotTop(self.sort_actual_estimation_credit )

        #Aggregated:
        self.actual_estimation_agg, newkeyList = self.aggDic(self.sort_actual_estimation_car, self.sort_actual_estimation_credit)
        self.sort_actual_estimation_agg = self.sort_to_dict(sorted(self.actual_estimation_agg.items(), key=lambda item: item[1][0]))
        self.aggBotUid, self.aggTopUid = self.getUid_BotTop(self.actual_estimation_agg)
        self.diffScore_agg, newkeyList_diffScore = self.aggDic(self.differenceScore_car, self.differenceScore_credit)
        self.domainFami_agg, newkeyList_fami = self.aggDic(self.familiarity_car, self.familiarity_credit)

    #-------------------check accuracy and parse logs into csv file----------------------#
    def getfilepath(self, logfilepath):
        filePaths = []
        Uids = []
        for root, dirs, files in os.walk(logfilepath):
            for name in files:
                userID = name.split('-')[0]
                Uids.append(userID)
                filePaths.append(os.path.join(root, name))
        return filePaths, Uids

    def get_groundtruth(self, task, groundtruth_file):
        df = pd.read_csv(groundtruth_file)
        truth = {}
        if task == 'credit':
            credit_level = ["Poor", "Standard", "Good"]  # {"Poor":0, "Standard":1,"Good":2}
            for i, name in enumerate(list(df['name'])):
                numID = name.split('_')[1]
                numericValue = list(df['Credit Score'])[i]
                stringLabel = credit_level[numericValue]
                truth[numID] = stringLabel
        if task == 'car':
            car_type = ["Sedan", "SUV", "Minivan"]  # {"Sedan":0, "SUV":1,"Minivan":2}
            for i, name in enumerate(list(df['name'])):
                numID = name.split('_')[1]
                numericValue = list(df['CarType'])[i]
                stringLabel = car_type[numericValue]
                truth[numID] = stringLabel
        return truth

    def sort_to_dict(self, lst):

        dict_ = {}
        for it in lst:
            dict_[it[0]]=it[1]

        return dict_

    def exOutliers(self, userLabel_dics, acc_dic, acc_per_dic):
        mean = np.mean(list(acc_dic.values()))
        std_car = np.std(list(acc_dic.values()))

        q1 = np.percentile(list(acc_dic.values()), 25)
        q3 = np.percentile(list(acc_dic.values()), 75)
        iqr = q3 - q1
        threshold = 1.5
        outliers_ids = {key: value for key, value in acc_dic.items() if value < (q1 - threshold * iqr) or value > (q3 + threshold * iqr)}

        userLabel_dics_filtered = {key: value for key, value in userLabel_dics.items() if key not in outliers_ids}
        acc_dic_filtered = {key: value for key, value in acc_dic.items() if key not in outliers_ids}
        acc_per_dic_filtered = {key: value for key, value in acc_per_dic.items() if key not in outliers_ids}

        return userLabel_dics_filtered, acc_dic_filtered, acc_per_dic_filtered

    def get_unique_uids(self, data):
        return list(data.get('data', {}).get('__collections__', {}).keys())

    def dateTime(self, data):
        collections = data.get('data', {}).get('__collections__', {})
        for uid in self.Uids:
            indivCollection = collections[uid]
            items = list(indivCollection.values())
            for item in items:
                timestamp_str = item['timestamp']['__time__']
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                item['timestamp'] = timestamp

        return items

    def aggDic(self, carDic, creidtDic):
        combined_dict = {}
        newkeyList = []
        for key, value in carDic.items():
            new_key = key
            while new_key in combined_dict:
                new_key = str(int(key) + 1)
                newkeyList.append(new_key)
            combined_dict[new_key] = value

        for key, value in creidtDic.items():
            new_key = key
            while new_key in combined_dict:
                new_key = str(int(key) + 1)
                newkeyList.append(new_key)
            combined_dict[new_key] = value

        return combined_dict, newkeyList


    def getuserlabels_acc(self, task, truth_dic):

        userLabel_dics = {}  #{uid1:{}, uid2:{},...}
        acc_dic = {}
        lenLabel = {}
        collections = self.data.get('data', {}).get('__collections__', {})
        for uid in self.Uids:
            T_num = 0
            userLabel_dic = {}
            indivCollection = collections[uid]
            items = list(indivCollection.values())
            items.sort(key=lambda x: x['timestamp'])

            # Get users' labels
            for item in items:
                if item.get('task') == task and item.get('type') == 'click':
                    point = item.get('point', {})
                    category = item.get('category')
                    pointID = point.split('_')[1]
                    userLabel_dic[pointID] = category


            for pointID in userLabel_dic:
                if userLabel_dic[pointID] == truth_dic[pointID]:
                    T_num += 1
            accuray = T_num / len(userLabel_dic)
            acc_dic[uid] = accuray
            userLabel_dics[uid] = userLabel_dic

            #check out the num of labeled points:
            lenLabel[uid] = len(userLabel_dic)

        # Compute individual percentile ranking based on the accuracy:
        acc_per_dic = {}
        for item in acc_dic:
            acc_per_dic[item] = []
            indiv_per = stats.percentileofscore(list(acc_dic.values()), acc_dic[item], kind='weak')
            acc_per_dic[item].append(acc_dic[item])
            acc_per_dic[item].append(indiv_per)
        return userLabel_dics, acc_dic, acc_per_dic, lenLabel

    def interaction_csv(self, task):
        interactions = []

        collections = self.data.get('data', {}).get('__collections__', {})
        for uid in self.Uids:
            T_num = 0
            userLabel_dic = {}
            indivCollection = collections[uid]
            items = list(indivCollection.values())
            items.sort(key=lambda x: x['timestamp'])

            # Write each item as a row:
            for item in items:
                row = []
                row.append(uid)

                if item.get('task') == task and item.get('event') == 'interaction':
                    #get point ID:
                    pointID = None
                    if item.get('type') == 'click' or item.get('type') == 'hover':
                        point = item.get('point')
                        if point:
                            pointID = point.split('_')[1]
                    row.append(pointID)

                    #get event and time stamp:
                    event_type = item.get('type')
                    eventTimeStamp = item.get('timestamp')
                    row.append(event_type)
                    row.append(eventTimeStamp)

                    #get x- and y- attributes:
                    org_axis = new_axis = None
                    if item.get('type') == 'axis_x' or item.get('type') == 'axis_y':
                        org_axis = item.get('org_axis')
                        new_axis = item.get('new_axis')
                    row.extend([org_axis, new_axis])

                    interactions.append(row)


        with open('interaction_' + task + '.csv', 'a', newline='') as f:
            write = csv.writer(f)
            for i, row in enumerate(interactions):
                if i == 0:
                    write.writerow(
                        ['user_id', 'PointID', 'event_type', 'eventTimeStamp', 'org_axis', 'new_axis'])
                write.writerow(row)
        return interactions

    #-------------------------DK results--------------------------#
    def postsruveyRows(self, postSurvey):

        postSurvey_paths = []
        for root, dirs, files in os.walk(postSurvey):
            for name in files:
                if 'categorization' in name:
                    postSurvey_paths.append(os.path.join(root, name))

        est_rows = []
        for p in postSurvey_paths:
            rows = []
            with open(p, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                # column = [row for row in reader]
                for row in reader:
                    select_row = {}
                    for key in row.keys():
                        if key in colIndex:
                            # row[key] = row[key].replace(' ', '')
                            select_row[key] = row[key]
                    rows.append(select_row)
                est_rows += rows[2:]
        est_rows = [d for d in est_rows if d['Q1'] in self.Uids]

        return est_rows

    def get_acc_est(self, acc_per_data, pointQue, rankingQue, abilityQue, domainQue):
        actual_estimation = {}  #{uid:[actual acc, actual per, percentile of estimated acc, estimated ranking]}
        estAcc = []
        differenceScore = {}
        familiarity = {}
        for uid in acc_per_data:
            for row in self.est_rows:
                if uid == row['Q1']:
                    actual_estimation[uid] = []
                    actual_estimation[uid].append(acc_per_data[uid][0])   #actual acc
                    actual_estimation[uid].append(acc_per_data[uid][1])   #actual percentile
                    actual_estimation[uid].append(float(row[pointQue])/self.sum)   #estimated acc
                    estAcc.append(float(row[pointQue])/self.sum)
                    actual_estimation[uid].append(float(row[rankingQue]))    #estimated ranking
                    actual_estimation[uid].append(float(row[abilityQue]))

                    differenceScore[uid]= float(row[pointQue])/self.sum - acc_per_data[uid][0]  #est acc - act acc

                    familiarity[uid] = []
                    familiarity[uid].append(float(row[pointQue])/self.sum - acc_per_data[uid][0])
                    familiarity[uid].append(row[domainQue])
                    familiarity[uid].append(domainFaNum[domainFaText.index(row[domainQue])])


        #compute percentile of estimated acc
        for item in actual_estimation:
            estACC_per = stats.percentileofscore(estAcc, actual_estimation[item][2], kind='weak')
            actual_estimation[item][2] = estACC_per

        return actual_estimation, differenceScore, familiarity, estAcc

    def get_quartile_per(self, sort_actual_estimation_data, col, ):
        quartile_per=np.array([0, 0, 0, 0])
        num = np.array([0,0,0,0])
        leng = len(sort_actual_estimation_data.keys())

        for i, item in enumerate(sort_actual_estimation_data):

            if i<leng/4:
                quartile_per[0] += sort_actual_estimation_data[item][col] #actual percentile
                num[0]+=1

            elif leng/4<=i<2*leng/4:
                quartile_per[1] += sort_actual_estimation_data[item][col]
                num[1] += 1

            elif 2*leng/4<=i<3*leng/4:
                quartile_per[2] += sort_actual_estimation_data[item][col]
                num[2] += 1

            elif 3*leng/4<=i:
                quartile_per[3] += sort_actual_estimation_data[item][col]
                num[3] += 1
        quartile_per = quartile_per / num
        return quartile_per

    def plot_DK(self, act_quartile_per, estAcc_quartile_per, est_quartile_per, abilityPercentile, title):
        font_size = 20

        users_id = ['Bottom Quartile', '2nd Quartile', '3rd Quartile', 'Top Quartile']
        x = len(users_id)

        plt.figure(figsize=(9,9))
        plt.ylabel("Percentile",  fontsize=font_size, weight='bold')
        plt.ylim(0,101)
        plt.xticks(fontsize=16, weight='bold')
        plt.yticks(fontsize=font_size)

        plt.plot(users_id, act_quartile_per, 'o', ls='-', linewidth=4, label="Actual Accuracy")
        plt.plot(users_id, estAcc_quartile_per, 's', ls=':', linewidth=4, label="Perceived Accuracy")
        plt.plot(users_id, est_quartile_per, '^', ls=':',linewidth=4, label="Perceived Ranking Percentile")
        plt.plot(users_id, abilityPercentile, 'o', ls=':', linewidth=4, label="Perceived Reasoning Ability")

        plt.legend(fontsize=font_size, loc=4)
        plt.grid(True)
        root = 'result/DK curves'
        plt.tight_layout()
        plt.savefig(os.path.join(root, '48' + title + " .jpg"), dpi=200)
        plt.show()

    def CombineXY(self, df):
        df = df[~df['event_type'].isin(['click_axis_x', 'click_axis_y', 'help'])]
        # Replace 'axis_x' and 'axis_y' with 'change_axis'
        df['event_type'] = df['event_type'].replace(['axis_x', 'axis_y'], 'change_axis')
        return df

    def calElapsedTime(self, df, task):

        df['eventTimeStamp'] = pd.to_datetime(df['eventTimeStamp'], format='%Y-%m-%d %H:%M:%S.%f').dt.time
        df['eventTimeStamp'] = pd.to_datetime(df['eventTimeStamp'].astype(str))
        df['elapsedTime'] = df.groupby('user_id')['eventTimeStamp'].diff().dt.total_seconds()

        df.to_csv(task + '.csv', index=False)


    def getUid_BotTop(self, sort_actual_estimation):
        bot_userID = list(sort_actual_estimation.keys())[0:round(len(sort_actual_estimation) / 4)]
        top_userID = list(sort_actual_estimation.keys())[round(3 * len(sort_actual_estimation) / 4):]
        return bot_userID, top_userID


    def dkLine(self, sort_actual_estimation_car, sort_actual_estimation_credit):

        # actual percentile of each quartile
        act_quartile_per_car = self.get_quartile_per(sort_actual_estimation_car, 1)
        act_quartile_per_credit = self.get_quartile_per(sort_actual_estimation_credit, 1)

        # estimated acc percentile of each quartile
        estAcc_quartile_per_car = self.get_quartile_per(sort_actual_estimation_car, 2)
        estAcc_quartile_per_credit = self.get_quartile_per(sort_actual_estimation_credit, 2)

        # estimated ranking percentile of each quartile
        est_quartile_per_car = self.get_quartile_per(sort_actual_estimation_car, 3)
        est_quartile_per_credit = self.get_quartile_per(sort_actual_estimation_credit, 3)

        # estimated ability percentile of each quartile
        est_abilityPercentile_per_car = self.get_quartile_per(sort_actual_estimation_car, 4)
        est_abilityPercentile_per_credit = self.get_quartile_per(sort_actual_estimation_credit, 4)

        # plot DK curves:
        self.plot_DK(act_quartile_per_car, estAcc_quartile_per_car, est_quartile_per_car, est_abilityPercentile_per_car,'Car Task')
        self.plot_DK(act_quartile_per_credit, estAcc_quartile_per_credit, est_quartile_per_credit, est_abilityPercentile_per_credit, 'Credit Task')

class PreSurvey:
    def __init__(self, que, attention_check_que, visLiteracy_que, vis_answer, opt_5, opt_3, est_path, sort_actual_estimation_car,
                 plus, minus, Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism, plus_score, minus_score, choice):
        traits_rows, visLiteracy_rows = self.TraitsLiteracy(est_path, sort_actual_estimation_car, que, attention_check_que, visLiteracy_que)
        self.visScore = self.LiteracyScore(visLiteracy_rows, visLiteracy_que, vis_answer, opt_5, opt_3)

        count = sum(1 for value in self.visScore.values() if value < 3)
        self.sum_dic = self.TraitsScore(traits_rows, plus, minus, Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism, plus_score, minus_score, choice)

        self.o_high_IDs, self.o_low_IDs, self.o_average_IDs = self.interpret(0)
        self.c_high_IDs, self.c_low_IDs, self.c_average_IDs = self.interpret(1)
        self.e_high_IDs, self.e_low_IDs, self.e_average_IDs = self.interpret(2)
        self.a_high_IDs, self.a_low_IDs, self.a_average_IDs = self.interpret(3)
        self.n_high_IDs, self.n_low_IDs, self.n_average_IDs = self.interpret(4)

    def TraitsLiteracy(self, est_path, sort_actual_estimation_car, que, attention_check_que, visLiteracy_que):
        with open(est_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            traits_rows = []
            attention_check_rows = []
            visLiteracy_rows = []
            for row in reader:
                if row['Q1'] in sort_actual_estimation_car.keys():  # car task user IDs
                    select_row = {}
                    attention_check_row = {}
                    visLiteracy_row = {}
                    for key in row.keys():
                        if key in que:
                            select_row[key] = row[key]
                        if key in attention_check_que:
                            attention_check_row[key] = row[key]
                        if key in visLiteracy_que:
                            visLiteracy_row[key] = row[key]
                    traits_rows.append(select_row)
                    attention_check_rows.append(attention_check_row)
                    visLiteracy_rows.append(visLiteracy_row)
        return traits_rows, visLiteracy_rows

    def LiteracyScore(self,visLiteracy_rows, visLiteracy_que, vis_answer, opt_5, opt_3):
        visScore = {}
        for i, user in enumerate(visLiteracy_rows):
            correct = 0
            incorrect = 0
            noanswer = 0

            correct_5 = 0
            incorrect_5 = 0
            correct_3 = 0
            incorrect_3 = 0

            for j, q in enumerate(visLiteracy_que[1:]):
                if user.get(q, "") == vis_answer[j]:
                    correct += 1
                    if q in opt_5:
                        correct_5 += 1
                    if q in opt_3:
                        correct_3 += 1
                elif user.get(q, "") != "":  # only increment incorrect if the user provided an answer
                    incorrect += 1
                    if q in opt_5:
                        incorrect_5 += 1
                    if q in opt_3:
                        incorrect_3 += 1
                else:
                    noanswer += 1
            print(f"User {user['Q1']}- correct: {correct},  incorrect: {incorrect}, no answered: {noanswer}")
            visLiteracy_score_5 = correct_5 - incorrect_5 / (5 - 1)
            visLiteracy_score_3 = correct_3 - incorrect_3 / (3 - 1)
            visLiteracy_score_sum = visLiteracy_score_5 + visLiteracy_score_3
            visScore[user['Q1']] = visLiteracy_score_sum
        return  visScore

    def TraitsScore(self,traits_rows, plus, minus, Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism, plus_score, minus_score, choice):
        traitsScore = {}  # score = {id: [1,2,3,4],[],[],[],[]}  #[O,C,E,A,N]
        for row in traits_rows:
            per_score = [[], [], [], [], []]  # [O,C,E,A,N]
            for key in row.keys():
                if key in plus :
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

            # score[row['Q1']] = sum(per_score)
            traitsScore[row['Q1']] = per_score

        sum_dic = {}  # sum_dic = {id: [1,2,3,4,5]}
        for item in traitsScore.items():
            sum_score = []
            sum_score.append(sum(item[1][0]))
            sum_score.append(sum(item[1][1]))
            sum_score.append(sum(item[1][2]))
            sum_score.append(sum(item[1][3]))
            sum_score.append(sum(item[1][4]))

            sum_dic[item[0]] = sum_score
        return sum_dic

    def interpret(self, index_personality):
        score_perTrait = []
        for item in self.sum_dic.items():
            score_perTrait.append(item[1][index_personality])
        mean = np.mean(score_perTrait)
        std = np.std(score_perTrait)

        high_IDs = []
        low_IDs = []
        average_IDs = []

        for uid in self.sum_dic:
            indiviScore = self.sum_dic[uid][index_personality]
            if indiviScore < mean - 0.5 * std:
                low_IDs.append(uid)
            if indiviScore > mean + 0.5 * std:
                high_IDs.append(uid)
            if mean - 0.5 * std <= indiviScore <= mean + 0.5 * std:
                average_IDs.append(uid)
        return high_IDs, low_IDs, average_IDs

class InterSequence:
    def __init__(self, filePath_car, filePath_credit, sort_actual_estimation_car, sort_actual_estimation_credit, o_high_IDs, o_low_IDs, c_high_IDs, c_low_IDs, e_high_IDs, e_low_IDs,
                 a_high_IDs, a_low_IDs, n_high_IDs, n_low_IDs, bot_userID_car, top_userID_car, bot_userID_credit, top_userID_credit):
        # filePath_car = 'interaction_elapsed_Car.csv'
        # filePath_credit = 'interaction_elapsed_Credit.csv'

        self.interaction_dict_car = self.read_as_dict(filePath_car)
        self.interaction_dict_credit = self.read_as_dict(filePath_credit)
        actions = [
            'drag',
            'hover',
            'click',
            'zoom',
            'change_axis',
        ]
        actions_col = [
            'drag',
            'hover',
            'click',
            'zoom',
            'change_axis',
        ]
        self.botMatrix_car, self.topMatrix_car = self.get_tran_matr(self.interaction_dict_car, sort_actual_estimation_car, actions, actions_col)
        self.botMatrix_credit, self.topMatrix_credit = self.get_tran_matr(self.interaction_dict_credit, sort_actual_estimation_credit, actions, actions_col)

        self.plot_transition_matrix(self.botMatrix_car,'car','48_bottom')
        self.plot_transition_matrix(self.topMatrix_car,'car','48_top')

        self.plot_transition_matrix(self.botMatrix_credit,'credit','48_bottom')
        self.plot_transition_matrix(self.topMatrix_credit,'credit','48_top')

        self.botMatrix_o_car, self.topMatrix_o_car = self.getTranMatr_personality(self.interaction_dict_car, o_high_IDs, o_low_IDs, actions, actions_col)
        self.botMatrix_c_car, self.topMatrix_c_car = self.getTranMatr_personality(self.interaction_dict_car, c_high_IDs, c_low_IDs, actions, actions_col)
        self.botMatrix_e_car, self.topMatrix_e_car = self.getTranMatr_personality(self.interaction_dict_car, e_high_IDs, e_low_IDs, actions, actions_col)
        self.botMatrix_a_car, self.topMatrix_a_car = self.getTranMatr_personality(self.interaction_dict_car, a_high_IDs, a_low_IDs, actions, actions_col)
        self.botMatrix_n_car, self.topMatrix_n_car = self.getTranMatr_personality(self.interaction_dict_car, n_high_IDs, n_low_IDs, actions, actions_col)

        self.botMatrix_o_credit, self.topMatrix_o_credit = self.getTranMatr_personality(self.interaction_dict_credit, o_high_IDs, o_low_IDs, actions, actions_col)
        self.botMatrix_c_credit, self.topMatrix_c_credit = self.getTranMatr_personality(self.interaction_dict_credit, c_high_IDs, c_low_IDs, actions, actions_col)
        self.botMatrix_e_credit, self.topMatrix_e_credit = self.getTranMatr_personality(self.interaction_dict_credit, e_high_IDs, e_low_IDs, actions, actions_col)
        self.botMatrix_a_credit, self.topMatrix_a_credit = self.getTranMatr_personality(self.interaction_dict_credit, a_high_IDs, a_low_IDs, actions, actions_col)
        self.botMatrix_n_credit, self.topMatrix_n_credit = self.getTranMatr_personality(self.interaction_dict_credit, n_high_IDs, n_low_IDs, actions, actions_col)

        # the count of change_axis:
        dfCar = pd.read_csv(filePath_car)
        botCar = dfCar[(dfCar['user_id'].isin([int(i) for i in bot_userID_car])) & (dfCar['event_type'] == 'change_axis')]
        topCar = dfCar[(dfCar['user_id'].isin([int(i) for i in top_userID_car])) & (dfCar['event_type'] == 'change_axis')]

        # Count the number of 'change_axis' events for each user_id
        count_botCar = botCar.groupby('user_id').size()
        count_topCar = topCar.groupby('user_id').size()

        mean_botCar = count_botCar.mean()
        print(f"The bot Car mean count is {mean_botCar}")
        mean_topCar = count_topCar.mean()
        print(f"The top Car mean count is {mean_topCar}")

        dfCredit = pd.read_csv(filePath_credit)
        botCredit = dfCredit[(dfCredit['user_id'].isin([int(i) for i in bot_userID_credit])) & (dfCredit['event_type'] == 'change_axis')]
        topCredit = dfCredit[(dfCredit['user_id'].isin([int(i) for i in top_userID_credit])) & (dfCredit['event_type'] == 'change_axis')]

        # Count the number of 'change_axis' events for each user_id
        count_botCredit = botCredit.groupby('user_id').size()
        count_topCredit = topCredit.groupby('user_id').size()
        mean_botCredit = count_botCredit.mean()
        print(f"The bot Credit mean count is {mean_botCredit}")
        mean_topCredit = count_topCredit.mean()
        print(f"The top Credit mean count is {mean_topCredit}")

    def read_as_dict(self, file_name):

        csvFile_all = open(file_name, 'r')
        track_csv = csv.DictReader(csvFile_all)

        interaction_dict = OrderedDict()

        for i, row in enumerate(track_csv):

            if row['elapsedTime'] != '' and float(
                    row['elapsedTime']) < 0.1:
                continue
            user_id = row['user_id']

            if user_id not in interaction_dict:
                interaction_dict[user_id] = []
            interaction_dict[user_id].append(row)
        csvFile_all.close()

        return interaction_dict

    def get_tran_matr(self, interaction_dict, sort_actual_estimation, actions, actions_col):

        bot_userID = list(sort_actual_estimation.keys())[0:round(len(sort_actual_estimation) / 4)]
        top_userID = list(sort_actual_estimation.keys())[round(3 * len(sort_actual_estimation) / 4):]

        bot_matrix = np.zeros((len(actions), len(actions)))
        top_matrix = np.zeros((len(actions), len(actions)))

        for user_id in interaction_dict:
            if user_id in bot_userID:
                all_act = interaction_dict[user_id]
                for j, act in enumerate(all_act):
                    if j + 1 < len(all_act):

                        act_type = act['event_type']
                        next_act = all_act[j + 1]['event_type']
                        if (act_type not in actions) or (next_act not in actions):
                            continue
                        idx = actions.index(act_type)
                        idy = actions_col.index(next_act)
                        bot_matrix[idx][idy] += 1
                        bot_matrix = bot_matrix.astype(int)

            if user_id in top_userID:
                all_act = interaction_dict[user_id]
                for j, act in enumerate(all_act):
                    if j + 1 < len(all_act):

                        act_type = act['event_type']
                        next_act = all_act[j + 1]['event_type']
                        if (act_type not in actions) or (next_act not in actions):
                            continue
                        idx = actions.index(act_type)
                        idy = actions_col.index(next_act)
                        top_matrix[idx][idy] += 1
                        top_matrix = top_matrix.astype(int)

        return bot_matrix, top_matrix

    def plot_transition_matrix(self, Matrix, task, group):
        data_frame = DataFrame(Matrix, columns=['drag', 'hover', 'click', 'zoom', 'change_axis'],
                               index=['drag', 'hover', 'click', 'zoom', 'change_axis'])
        sns.set(font_scale=1)
        plt.subplots(figsize=(8, 8))
        plt.clf()
        plt.figure(dpi=200)
        plt.title('Transition matrix for ' + group + ' quartile in ' + task, y=-0.1)
        ax = sns.heatmap(data=data_frame,
                         square=True,
                         vmin=Matrix.min(),
                         vmax=Matrix.max(),
                         annot=True,
                         fmt="d",
                         cmap=plt.get_cmap('Greys'),
                         linewidths=1.5,
                         linecolor="white", )

        ax.set_yticklabels(ax.get_yticklabels(), rotation=360)
        ax.xaxis.tick_top()
        root = 'result/interaction analysis'
        plt.savefig(os.path.join(root, group + ' ' + task + ' ' + 'TransitionMatrix.jpg'), bbox_inches='tight')
        plt.show()

    def getTranMatr_personality(self,interaction_dict, high_IDs, low_IDs, actions, actions_col):

        bot_matrix = np.zeros((len(actions), len(actions)))
        top_matrix = np.zeros((len(actions), len(actions)))

        for user_id in interaction_dict:
            if user_id in high_IDs:
                all_act = interaction_dict[user_id]
                for j, act in enumerate(all_act):
                    if j + 1 < len(all_act):

                        act_type = act['event_type']
                        next_act = all_act[j + 1]['event_type']
                        if (act_type not in actions) or (next_act not in actions):
                            continue
                        idx = actions.index(act_type)
                        idy = actions_col.index(next_act)
                        top_matrix[idx][idy] += 1
                        top_matrix = top_matrix.astype(int)

            if user_id in low_IDs:
                all_act = interaction_dict[user_id]
                for j, act in enumerate(all_act):
                    if j + 1 < len(all_act):

                        act_type = act['event_type']
                        next_act = all_act[j + 1]['event_type']
                        if (act_type not in actions) or (next_act not in actions):
                            continue
                        idx = actions.index(act_type)
                        idy = actions_col.index(next_act)
                        bot_matrix[idx][idy] += 1
                        bot_matrix = bot_matrix.astype(int)

        return bot_matrix, top_matrix

class InterPace:
    def __init__(self, filePath_car, filePath_credit, bot_userID_car, top_userID_car, bot_userID_credit, top_userID_credit,
                 o_high_IDs, o_low_IDs, c_high_IDs, c_low_IDs, e_high_IDs, e_low_IDs, a_high_IDs, a_low_IDs, n_high_IDs, n_low_IDs):
        interaction_car = pd.read_csv(filePath_car)
        interaction_credit = pd.read_csv(filePath_credit)

        self.df_car, self.bot_thinking_car, self.top_thinking_car, self.bot_carData, self.top_carData = self.cleaning(interaction_car, bot_userID_car, top_userID_car)
        self.df_credit, self.bot_thinking_credit, self.top_thinking_credit, self.bot_creditData, self.top_creditData = self.cleaning(interaction_credit, bot_userID_credit, top_userID_credit)

        self.h2TimeCountPlot(self.bot_carData, self.top_carData, 'CAR')
        self.h2TimeCountPlot(self.bot_creditData, self.top_creditData,  'CREDIT')

        self.highThink_carO, self.lowThink_carO = self.getThinkTime_personality(self.df_car, o_high_IDs, o_low_IDs)
        self.highThink_carC, self.lowThink_carC = self.getThinkTime_personality(self.df_car, c_high_IDs, c_low_IDs)
        self.highThink_carE, self.lowThink_carE = self.getThinkTime_personality(self.df_car, e_high_IDs, e_low_IDs)
        self.highThink_carA, self.lowThink_carA = self.getThinkTime_personality(self.df_car, a_high_IDs, a_low_IDs)
        self.highThink_carN, self.lowThink_carN = self.getThinkTime_personality(self.df_car, n_high_IDs, n_low_IDs)

        self.highThink_creditO, self.lowThink_creditO = self.getThinkTime_personality(self.df_credit, o_high_IDs, o_low_IDs)
        self.highThink_creditC, self.lowThink_creditC = self.getThinkTime_personality(self.df_credit, c_high_IDs, c_low_IDs)
        self.highThink_creditE, self.lowThink_creditE = self.getThinkTime_personality(self.df_credit, e_high_IDs, e_low_IDs)
        self.highThink_creditA, self.lowThink_creditA = self.getThinkTime_personality(self.df_credit, a_high_IDs, a_low_IDs)
        self.highThink_creditN, self.lowThink_creditN = self.getThinkTime_personality(self.df_credit, n_high_IDs, n_low_IDs)

    def h2TimeCountPlot(self, bot_carData, top_carData, task):
            top_carData['group'] = 'Top'
            bot_carData['group'] = 'Bottom'
            data_all = pd.concat([top_carData, bot_carData], ignore_index=True)

            #stats test:
            from scipy.stats import shapiro
            from scipy.stats import levene
            from scipy.stats import ttest_ind
            event_types = data_all['event_type'].unique()
            for event in event_types:
                data_group_top = data_all[(data_all['event_type'] == event) & (data_all['group'] == 'Top')]['thinking']
                _, p1 = shapiro(data_group_top)

                data_group_bottom = data_all[(data_all['event_type'] == event) & (data_all['group'] == 'Bottom')][
                    'thinking']
                _, p2 = shapiro(data_group_bottom)

                # equal variances?
                top_group = data_all[(data_all['event_type'] == event) & (data_all['group'] == 'Top')]['thinking']
                bottom_group = data_all[(data_all['event_type'] == event) & (data_all['group'] == 'Bottom')]['thinking']
                _, p = levene(top_group, bottom_group)


                print("###Conclusion###")
                if p1 > 0.05 and p2 > 0.05 and p > 0.05:
                    print(task + " - normally distributed and equal variances ->  independent samples t-test")
                elif p1 > 0.05 and p2 > 0.05 and p < 0.05:
                    print(task + " - normally distributed and unequal variances ->  Welch's t-test")
                elif p1 < 0.05 and p2 < 0.05 and p > 0.05:
                    print(task + " - NOT normally distributed but in a similar shape ->  Mann-Whitney U test")
                else:
                    print("Need to choose other tests.")

                t_stat, p_value = ttest_ind(top_group, bottom_group, equal_var=False)
                print(task + ' ' + event + ': Statistics=%.3f, p=%.3f' % (t_stat, p_value))

            sns.set(style="whitegrid")
            custom_palette = {"Top": "#87719f", "Bottom": "#cdc7e2"}
            event_order = ["change_axis", "zoom", "hover", "click", "drag"]
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6), sharey=True)

            sns.pointplot(x='thinking', y='event_type', hue='group', data=data_all, ci=95, capsize=0, orient='h',
                          ax=ax1, dodge=True, linestyles="", palette=custom_palette, order=event_order)

            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, ['Top Quartiles', 'Bottom Quartiles'], fontsize=14,
                       loc='lower right')

            ax1.set_title('Mean and 95% Confidence Interval of Thinking Time', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Mean think times preceding each interaction type (secs)', fontsize=14,fontweight='bold')

            ax1.tick_params(axis='y', labelsize=15)
            ax1.set_ylabel('Interaction Type', fontsize=15,fontweight='bold')

            sns.countplot(y='event_type', hue='group', data=data_all, orient='h', ax=ax2, palette=custom_palette, order=event_order)
            for p in ax2.patches:
                width = p.get_width()
                ax2.text(width + 1,
                         p.get_y() + p.get_height() / 2,
                         '{:1.0f}'.format(width),
                         ha='left',
                         va='center')

            ax2.set_title('Number of Occurrences of Each Interaction Type', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Count', fontsize=14,fontweight='bold')
            ax2.set_ylabel('')
            ax2.legend(handles, ['Top Quartiles', 'Bottom Quartiles'], fontsize=14,
                       loc='lower right')

            plt.subplots_adjust(wspace=0.06)
            plt.tight_layout()
            root = 'result/interaction analysis'
            plt.savefig(os.path.join(root, task + 'BarInterThink.jpg'), bbox_inches='tight')
            plt.show()

    def timedelta_to_seconds(self, td):
        if pd.isnull(td):
            return np.nan
        else:
            return int(td.total_seconds() * 1000)

    def cleaning(self, df, bot_userID, top_userID):
        # pd.set_option('display.max_rows', None) #show all rows
        df['eventTimeStamp'] = pd.to_datetime(df['eventTimeStamp'])

        df['elapsedTime'] = df['elapsedTime'].fillna(0)  # 'click' fill 0
        df = df[(df['elapsedTime'] >= 0.1) | (df['elapsedTime'] == 0)]
        # df['thinking'] = np.nan
        df['thinking'] = df['elapsedTime']
        filtered_df = df

        print(filtered_df['user_id'].dtypes)
        bot_userID = [int(id) for id in bot_userID]
        top_userID = [int(id) for id in top_userID]

        bot_thinking = filtered_df[filtered_df['user_id'].isin(bot_userID)]['thinking']
        top_thinking = filtered_df[filtered_df['user_id'].isin(top_userID)]['thinking']

        bot_data = filtered_df[filtered_df['user_id'].isin(bot_userID)]
        top_data = filtered_df[filtered_df['user_id'].isin(top_userID)]

        event_types = filtered_df['event_type'].unique()
        event_id_mapping = {event_type: i for i, event_type in
                            enumerate(event_types)}  # {'change_axis': 0, 'hover': 1, 'zoom': 2, 'click': 3, 'drag': 4}
        filtered_df['event_id'] = filtered_df['event_type'].map(event_id_mapping)
        filtered_df['interationSeq'] = filtered_df.groupby('user_id').cumcount() + 1   #filtered_df['elapsedTime']


        return filtered_df, bot_thinking, top_thinking, bot_data, top_data

    def getThinkTime_personality(self, df, high_IDs, low_IDs):
        low_IDs = [int(id) for id in low_IDs]
        high_IDs = [int(id) for id in high_IDs]
        low_thinking = df[df['user_id'].isin(low_IDs)]['thinking']
        high_thinking = df[df['user_id'].isin(high_IDs)]['thinking']
        return high_thinking, low_thinking

class AttentionMap:
    def __init__(self, Uids, data, bot_userID_car, top_userID_car, bot_userID_credit, top_userID_credit, inter_car_df, inter_credit_df,
                 o_high_IDs, o_low_IDs, c_high_IDs, c_low_IDs, e_high_IDs, e_low_IDs, a_high_IDs, a_low_IDs, n_high_IDs, n_low_IDs):
        self.root = 'result/eyeTracking'
        file_name_car = 'result/eyeTracking/Gaze_car.csv'
        file_name_credit = 'result/eyeTracking/Gaze_credit.csv'
        v_threshold_car = 35
        v_threshold_credit = 35
        Gaze_car, self.timeSpent_car = self.get_rawGaze('car', Uids, data)
        Gaze_credit, self.timeSpent_credit = self.get_rawGaze('credit', Uids, data)
        rawGaze_dict_car = self.read_as_dict(file_name_car)
        rawGaze_dict_credit = self.read_as_dict(file_name_credit)

        per_fixTime_botCar, per_fixCount_botCar, timespent_dic_botCar, count_list_botCar, duration_list_botCar = self.ivt(rawGaze_dict_car, bot_userID_car, v_threshold_car, self.timeSpent_car)
        per_fixTime_topCar, per_fixCount_topCar, timespent_dic_topCar, count_list_topCar, duration_list_topCar = self.ivt(rawGaze_dict_car, top_userID_car, v_threshold_car, self.timeSpent_car)
        per_fixTime_botCredit, per_fixCount_botCredit, timespent_dic_botCredit, count_list_botCredit, duration_list_botCredit = self.ivt(rawGaze_dict_credit, bot_userID_credit, v_threshold_credit, self.timeSpent_credit)
        per_fixTime_topCredit, per_fixCount_topCredit, timespent_dic_topCredit, count_list_topCredit, duration_list_topCredit  = self.ivt(rawGaze_dict_credit, top_userID_credit, v_threshold_credit, self.timeSpent_credit)

        self.fixationTest(count_list_botCar, count_list_topCar, 'count', 'car')
        self.fixationTest(duration_list_botCar, duration_list_topCar, 'duration', 'car')
        self.fixationTest(count_list_botCredit, count_list_topCredit, 'count', 'credit')
        self.fixationTest(duration_list_botCredit, duration_list_topCredit, 'duration', 'credit')



        print('-------Combined Car mixed-------')
        # mixed-effects model
        merged_timeSpent_car = {**timespent_dic_botCar, **timespent_dic_topCar}
        df1 = pd.DataFrame({
            'User_ID': list(merged_timeSpent_car.keys()),
            'Fixation_Duration': per_fixTime_botCar + per_fixTime_topCar,
            'Fixation_Count': per_fixCount_botCar + per_fixCount_topCar,
            'Task_Completion_Time': list(merged_timeSpent_car.values())
        })
        model = smf.mixedlm("Task_Completion_Time ~ Fixation_Duration + Fixation_Count", df1, groups=df1["User_ID"])
        result = model.fit()
        print(result.summary())

        print('-------Combined Credit mixed-------')
        merged_timeSpent_credit = {**timespent_dic_botCredit, **timespent_dic_topCredit}
        df2 = pd.DataFrame({
            'User_ID': list(merged_timeSpent_credit.keys()),
            'Fixation_Duration': per_fixTime_botCredit + per_fixTime_topCredit,
            'Fixation_Count': per_fixCount_botCredit + per_fixCount_topCredit,
            'Task_Completion_Time': list(merged_timeSpent_credit.values())
        })
        model = smf.mixedlm("Task_Completion_Time ~ Fixation_Duration + Fixation_Count", df2, groups=df2["User_ID"])
        result = model.fit()
        print(result.summary())

        #interacction pace across bot and top in both task:
        self.InteractionPace(inter_car_df, bot_userID_car, top_userID_car, 'car')
        self.InteractionPace(inter_credit_df, bot_userID_car, top_userID_car, 'credit')

        #interacction pace across high and low scores on each trait:
        self.InteractionPace(inter_car_df, o_low_IDs, o_high_IDs, 'carO')
        self.InteractionPace(inter_car_df, c_low_IDs, c_high_IDs, 'carC')
        self.InteractionPace(inter_car_df, e_low_IDs, e_high_IDs, 'carE')
        self.InteractionPace(inter_car_df, a_low_IDs, a_high_IDs, 'carA')
        self.InteractionPace(inter_car_df, n_low_IDs, n_high_IDs, 'carN')

        self.InteractionPace(inter_credit_df, o_low_IDs, o_high_IDs, 'creditO')
        self.InteractionPace(inter_credit_df, c_low_IDs, c_high_IDs, 'creditC')
        self.InteractionPace(inter_credit_df, e_low_IDs, e_high_IDs, 'creditE')
        self.InteractionPace(inter_credit_df, a_low_IDs, a_high_IDs, 'creditA')
        self.InteractionPace(inter_credit_df, n_low_IDs, n_high_IDs, 'creditN')

    def assumpCheck(self, bot_data, top_data, assumpNum, task):

        print('-------------' + assumpNum + ' ' + task + '---------------')
        stat1, p1 = stats.shapiro(bot_data)
        stat2, p2 = stats.shapiro(top_data)

        # checkout the homogeneity of variances: p-value > .05 --> equal variances
        stat, p = stats.levene(bot_data, top_data)

        # check out the distributional similarity: p-value > .05 --> same
        D_stat, p_value = stats.ks_2samp(bot_data, top_data)

    def ttest(self, value1, value2, task):
        stat, p = stats.ttest_ind(value1, value2)
        print('-------------' + task + '---------------')
        print('Ttest: Statistics=%.3f, p=%.3f' % (stat, p))
        return p

    def fixationTest(self, count_list_botCar, count_list_topCar, var, task):

        self.assumpCheck(count_list_botCar, count_list_topCar, var, task)
        self.ttest(count_list_botCar, count_list_topCar, task)


    def InteractionPace(self, inter_car_df, bot_userID_car, top_userID_car, task):


        total_counts = inter_car_df.groupby('user_id').sum().to_dict()['PointID']
        bot_pace = []
        top_pace = []

        for user_id, time_spent in self.timeSpent_car.items():
            if user_id in bot_userID_car:
                ratio = total_counts[int(user_id)] / time_spent
                bot_pace.append(ratio)
            elif user_id in top_userID_car:
                ratio = total_counts[int(user_id)] / time_spent
                top_pace.append(ratio)
        alpha = 0.05
        corrected_alpha = alpha / 5
        self.assumpCheck(bot_pace, top_pace, 'interaction pace', task)
        self.ttest(bot_pace, top_pace, task)

        # Calculate the mean
        bot_mean_pace = sum(bot_pace) / len(bot_pace)
        top_mean_pace = sum(top_pace) / len(top_pace)
        print(task + " bot interaction Pace: {:.3f}".format(bot_mean_pace))
        print(task + " top interaction Pace: {:.3f}".format(top_mean_pace))

    def get_rawGaze(self, task, Uids, data):

        Gaze = []
        click_test = []
        collections = data.get('data', {}).get('__collections__', {})
        screenSize_dic = {}  # {uid:[screenWidth, screenHeight]}
        timeSpent = {}

        for uid in Uids:
            screenSize_list = []
            indivCollection = collections[uid]
            items = list(indivCollection.values())
            items.sort(key=lambda x: x['timestamp'])

            # Get gaze coordinates:
            for i, item in enumerate(items):
                if i == 0:
                    startTime = item['timestamp']
                if item.get('task') == task and item.get('event') == 'complete logging':
                    endTime = item['timestamp']
                    timeDiff = endTime - startTime
                    timeSpent[uid] = timeDiff.total_seconds()
                if item.get('task') == task and item.get('event') == 'eyetracking':
                    gaze_row = []
                    userID = uid
                    x_coordinate = item.get('x')
                    y_coordinate = item.get('y')
                    viewHeight = item.get('viewHeight')
                    viewWidth = item.get('viewWidth')
                    normalizedX = item.get('normalizedX')
                    normalizedy = item.get('normalizedy')
                    eventTimeStamp = item.get('timestamp')
                    gaze_row.extend([userID,
                                     x_coordinate, y_coordinate,
                                     viewHeight, viewWidth,
                                     normalizedX, normalizedy,
                                     eventTimeStamp])
                    Gaze.append(gaze_row)

                if item.get('task') == task and item.get('event') == 'click':
                    click_row = []
                    userID = uid
                    x_coordinate = item.get('x')
                    y_coordinate = item.get('y')
                    viewHeight = item.get('viewHeight')
                    viewWidth = item.get('viewWidth')
                    normalizedX = item.get('normalizedX')
                    normalizedy = item.get('normalizedy')
                    eventTimeStamp = item.get('timestamp')
                    click_row.extend([userID,
                                      x_coordinate, y_coordinate,
                                      viewHeight, viewWidth,
                                      normalizedX, normalizedy,
                                      eventTimeStamp])
                    click_test.append(click_row)

                if item.get('task') == task and item.get('type') == 'click_axis_x':
                    click_row = []
                    userID = uid
                    x_coordinate = item.get('x')
                    y_coordinate = item.get('y')

                    viewHeight = item.get('viewHeight')
                    viewWidth = item.get('viewWidth')
                    normalizedX = item.get('normalizedX')
                    normalizedy = item.get('normalizedy')
                    eventTimeStamp = item.get('timestamp')
                    click_row.extend([userID,
                                      x_coordinate, y_coordinate,
                                      viewHeight, viewWidth,
                                      normalizedX, normalizedy,
                                      eventTimeStamp])
                    click_test.append(click_row)

                if item.get('task') == task and item.get('type') == 'click_axis_y':
                    click_row = []
                    userID = uid
                    x_coordinate = item.get('x')
                    y_coordinate = item.get('y')
                    viewHeight = item.get('viewHeight')
                    viewWidth = item.get('viewWidth')
                    normalizedX = item.get('normalizedX')
                    normalizedy = item.get('normalizedy')
                    eventTimeStamp = item.get('timestamp')
                    click_row.extend([userID,
                                      x_coordinate, y_coordinate,
                                      viewHeight, viewWidth,
                                      normalizedX, normalizedy,
                                      eventTimeStamp])
                    click_test.append(click_row)

                if item.get('task') == task and item.get('type') == 'help':
                    click_row = []
                    userID = uid
                    x_coordinate = item.get('x')
                    y_coordinate = item.get('y')
                    viewHeight = item.get('viewHeight')
                    viewWidth = item.get('viewWidth')
                    normalizedX = item.get('normalizedX')
                    normalizedy = item.get('normalizedy')
                    eventTimeStamp = item.get('timestamp')
                    click_row.extend([userID,
                                      x_coordinate, y_coordinate,
                                      viewHeight, viewWidth,
                                      normalizedX, normalizedy,
                                      eventTimeStamp])
                    click_test.append(click_row)

                if item.get('event') == 'complete logging':
                    screenWidth = item.get('screenWidth')
                    screenHeight = item.get('screenHeight')
                    screenSize_list.extend([screenWidth, screenHeight])
                    screenSize_dic[uid] = screenSize_list

        # with open(os.path.join(self.root, 'Gaze_' + task + '.csv'), 'a', newline='') as f:
        #     write = csv.writer(f)
        #     for i, row in enumerate(Gaze):
        #         if i == 0:
        #             write.writerow(['user_id', 'X', 'Y', 'viewHeight', 'viewWidth', 'normalizedX', 'normalizedy','TimeStamp'])
        #             # write.writerow(['X', 'Y'])   #Used for density heatmap
        #         write.writerow(row)
        #
        # with open(os.path.join(self.root, 'Click_' + task + '.csv'), 'a', newline='') as f:
        #     write = csv.writer(f)
        #     for i, row in enumerate(click_test):
        #         if i == 0:
        #             write.writerow(['user_id', 'X', 'Y', 'viewHeight', 'viewWidth', 'normalizedX', 'normalizedy','TimeStamp'])
        #             # write.writerow(['X', 'Y'])   #Used for density heatmap
        #         write.writerow(row)
        return Gaze, timeSpent

    def read_as_dict(self, file_name):

        csvFile_all = open(file_name, 'r')
        track_csv = csv.DictReader(csvFile_all)

        rawGaze_dict = OrderedDict()

        for i, row in enumerate(track_csv):

            user_id = row['user_id']

            if user_id not in rawGaze_dict:
                rawGaze_dict[user_id] = []
            rawGaze_dict[user_id].append(row)
        csvFile_all.close()

        return rawGaze_dict

    def parse_date(self, date_str):
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

        return date_obj

    def ivt(self, rawGaze_dict, bot_userID_car, v_threshold, timeSpent_car):
        fixation_groups = {}  # {uid: []}
        sum_fixCount = 0
        sum_fixTime = 0
        sum_user = 0
        timespent_dic = {}
        per_fixCount = []
        per_fixTime = []

        count_list = []
        duration_list = []
        for user_id in rawGaze_dict:
            if user_id in bot_userID_car:
                timespent_dic[user_id] = timeSpent_car[user_id]

                sum_user += 1
                fixation_groups[user_id] = []
                Xs = []
                Ys = []
                timeStamps = []
                attr = rawGaze_dict[user_id]
                for a in attr:
                    Xs.append(a['X'])
                    Ys.append(a['Y'])
                    timeStamps.append(a['TimeStamp'])

                ts = []  # (ts[1] - ts[0])
                epoch_start = datetime(1970, 1, 1)

                for t in timeStamps:
                    date_obj = self.parse_date(t)
                    seconds_since_epoch = (date_obj - epoch_start).total_seconds()
                    ts.append(seconds_since_epoch)

                times = ts  # TOD0: CHECK if times in sec

                difX = []
                difY = []
                tdif = []

                for i in range(len(attr) - 1):
                    difX.append(float(Xs[i + 1]) - float(Xs[i]))
                    difY.append(float(Ys[i + 1]) - float(Ys[i]))
                    tdif.append((times[i + 1] - times[i]))

                difDistance = np.sqrt(np.power(difX, 2) + np.power(difY, 2))  # in pix
                velocity = difDistance / tdif  # velocity in pix/sec

                fixation_indices = np.where(velocity < v_threshold)[0]

                fixations = []
                current_fixation = []

                for idx in fixation_indices:
                    if len(current_fixation) == 0 or idx == current_fixation[-1] + 1:
                        current_fixation.append(idx)
                    else:
                        fixations.append(current_fixation)
                        current_fixation = [idx]


                if len(current_fixation) > 0:
                    fixations.append(current_fixation)


                fixation_durations = [(times[fix[-1] + 1] - times[fix[0]]) for fix in fixations]
                sum_fixCount += len(fixations)
                sum_fixTime += sum(fixation_durations)

                if len(fixations) > 0:
                    per_fixTime.append(sum(fixation_durations) / len(fixations))
                else:
                    per_fixTime.append(0)

                per_fixCount.append(len(fixations))

                if len(fixations) > 0:
                    count_list.append(len(fixations))
                    duration_list.append(sum(fixation_durations)/len(fixations))


        avgFixCount = sum_fixCount/sum_user
        avgFixTime = sum_fixTime/sum_fixCount
        print('-------------')
        print('num of user:', sum_user)
        print('avgFixCount:', avgFixCount)
        print('avgFixTime:', avgFixTime)

        return per_fixTime, per_fixCount, timespent_dic, count_list, duration_list

    def GaussianMask(self, sizex, sizey, sigma, center=None, fix=1):
        """
        sizex  : mask width
        sizey  : mask height
        sigma  : gaussian Sd
        center : gaussian mean
        fix    : gaussian max
        return gaussian mask
        """
        x = np.arange(0, sizex, 1, float)
        y = np.arange(0, sizey, 1, float)
        x, y = np.meshgrid(x, y)

        if center is None:
            x0 = sizex // 2
            y0 = sizey // 2
        else:
            if np.isnan(center[0]) == False and np.isnan(center[1]) == False:
                x0 = center[0]
                y0 = center[1]
            else:
                return np.zeros((sizey, sizex))

        return fix * np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    def Fixpos2Densemap(self, fix_arr, imgfile, H, W, alpha, threshold):
        """
        fix_arr   : fixation array number of subjects x 3(x,y,fixation)
        width     : output image width
        height    : output image height
        imgfile   : image file (optional)
        alpha     : marge rate imgfile and heatmap (optional)
        threshold : heatmap threshold(0~255)
        return heatmap
        """

        heatmap = np.zeros((H, W), np.float32)
        for n_subject in tqdm(range(fix_arr.shape[0])):
            heatmap += self.GaussianMask(W, H, 33, (fix_arr[n_subject, 0], fix_arr[n_subject, 1]),
                                    fix_arr[n_subject, 2])

        # Normalization
        heatmap = heatmap / np.amax(heatmap)
        heatmap = heatmap * 255
        heatmap = heatmap.astype("uint8")

        if imgfile.any():
            # Resize heatmap to imgfile shape
            h, w, _ = imgfile.shape
            heatmap = cv2.resize(heatmap, (w, h))
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Create mask
            mask = np.where(heatmap <= threshold, 1, 0)
            mask = np.reshape(mask, (h, w, 1))
            mask = np.repeat(mask, 3, axis=2)

            # Marge images
            marge = imgfile * mask + heatmap_color * (1 - mask)
            marge = marge.astype("uint8")
            marge = cv2.addWeighted(imgfile, 1 - alpha, marge, alpha, 0)
            return marge

        else:
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            return heatmap

    def fixationArray(self, filePath_car, selecUids, convert, W, H):
        fix_arr_car = pd.read_csv(filePath_car)

        # fix_toy = np.random.randint(0,2, size = len(fix_arr_car))
        fix_toy = np.ones(len(fix_arr_car), dtype=int)
        fix_arr_car['fixation'] = fix_toy

        indi_fix_list = []

        for uid in selecUids:
            print(fix_arr_car['user_id'].dtypes)
            uid = int(uid)
            
            viewport_width = fix_arr_car[fix_arr_car['user_id'] == uid]['viewWidth'].unique()[0]
            viewport_height = fix_arr_car[fix_arr_car['user_id'] == uid]['viewHeight'].unique()[0]
            indi_fix = fix_arr_car[['X', 'Y', 'fixation']][fix_arr_car['user_id'] == uid]

            indi_fix = np.array(indi_fix, dtype=float)

            if convert == True:
                fix_arr_car[:, 0] += viewport_width / 2
                fix_arr_car[:, 1] = viewport_height / 2 - fix_arr_car[:, 1]

            # rescale gaze coordinates (to image coordinate system)
            ratio_x = W / int(viewport_width)
            ratio_y = H / int(viewport_height)

            indi_fix[:, 0] *= ratio_x
            indi_fix[:, 1] *= ratio_y
            indi_fix_list.append(indi_fix)

        all_indi_fix = np.concatenate(indi_fix_list, axis=0)

        return all_indi_fix

    def heatmapPlot(self, backgroundPath_car, output, bot_userID_car, top_userID_car, groupB, groupT, task):
        img_car = cv2.imread(backgroundPath_car)
        H_car, W_car, _ = img_car.shape  # 1795*2880
        all_indi_fix_bot = self.fixationArray(filePath_car, bot_userID_car, False, W_car, H_car)
        heatmap = self.Fixpos2Densemap(all_indi_fix_bot, img_car, H_car, W_car, 0.6, 0)
        cv2.imwrite(os.path.join(output + '48_' + groupB + 'click_mac_' + task + '.png'), heatmap)
        all_indi_fix_top = self.fixationArray(filePath_car, top_userID_car, False, W_car, H_car)
        heatmap = self.Fixpos2Densemap(all_indi_fix_top, img_car, H_car, W_car, 0.6, 0)
        cv2.imwrite(os.path.join(output + '48_' + groupT + 'click_mac_' + task + '.png'), heatmap)


class StaTests:

    def __init__(self, alpha, num_test, sort_actual_estimation_car, sort_actual_estimation_credit, differenceScore_car, differenceScore_credit,
                 bot_userID_car, top_userID_car, bot_userID_credit, top_userID_credit, botMatrix_car, topMatrix_car, botMatrix_credit, topMatrix_credit,
                 bot_thinking_car, top_thinking_car, bot_thinking_credit, top_thinking_credit,
                 bot_carData, top_carData, bot_creditData, top_creditData,
                 timeSpent_car, timeSpent_credit,
                 botMatrix_o_car, topMatrix_o_car, botMatrix_c_car, topMatrix_c_car, botMatrix_e_car, topMatrix_e_car, botMatrix_a_car, topMatrix_a_car, botMatrix_n_car, topMatrix_n_car,
                 botMatrix_o_credit, topMatrix_o_credit, botMatrix_c_credit, topMatrix_c_credit, botMatrix_e_credit, topMatrix_e_credit, botMatrix_a_credit, topMatrix_a_credit,
                 botMatrix_n_credit, topMatrix_n_credit,
                 highThink_carO, lowThink_carO, highThink_carC, lowThink_carC, highThink_carE, lowThink_carE, highThink_carA, lowThink_carA, highThink_carN, lowThink_carN,
                 highThink_creditO, lowThink_creditO, highThink_creditC, lowThink_creditC, highThink_creditE, lowThink_creditE, highThink_creditA, lowThink_creditA, highThink_creditN, lowThink_creditN,
                 df_car, df_credit,
                 o_high_IDs, o_low_IDs,c_high_IDs, c_low_IDs,e_high_IDs, e_low_IDs,a_high_IDs, a_low_IDs,n_high_IDs, n_low_IDs,
                 visScore,
                 sum_dic,
                 familiarity_car, familiarity_credit, domainFami_agg,
                 diffScore_agg):

        self.root = 'result/stats/'

        # Task difficulty - task accuracy across the two tasks
        acc_car = [value[0] for value in sort_actual_estimation_car.values()]
        acc_credit = [value[0] for value in sort_actual_estimation_credit.values()]
        print("Car mean actual acc:", sum(acc_car)/len(acc_car))
        print("Credit mean actual acc:", sum(acc_credit) / len(acc_credit))
        self.assumpCheck(acc_car, acc_credit, 'H0', 'Task Difficulty')
        p = self.ttest(acc_car, acc_credit, 'Task Difficulty')

        #h1: difference score across top and bottom
        H1_bot_car = list({k: differenceScore_car[k] for k in bot_userID_car if k in differenceScore_car}.values())
        H1_top_car = list({k: differenceScore_car[k] for k in top_userID_car if k in differenceScore_car}.values())
        H1_bot_credit = list({k: differenceScore_credit[k] for k in bot_userID_credit if k in differenceScore_credit}.values())
        H1_top_credit = list({k: differenceScore_credit[k] for k in top_userID_credit if k in differenceScore_credit}.values())
        self.assumpCheck(H1_bot_car, H1_top_car, 'H1', 'Car Difference Score')
        self.assumpCheck(H1_bot_credit, H1_top_credit, 'H1', ' Credit Difference Score')
        p = self.ttest(H1_bot_car, H1_top_car, 'Car Difference Score')
        p = self.ttest(H1_bot_credit, H1_top_credit, 'Credit Difference Score')

        #h2.1 Transition matrices across top and bottom
        #normalized:
        normalized_btMatrix_car = self.normalize_transition_matrix(botMatrix_car)
        normalized_topMatrix_car = self.normalize_transition_matrix(topMatrix_car)
        normalized_botMatrix_credit = self.normalize_transition_matrix(botMatrix_credit)
        normalized_topMatrix_credit = self.normalize_transition_matrix(topMatrix_credit)

        #correlation coeff:
        v1_car = normalized_btMatrix_car.ravel()
        v2_car = normalized_topMatrix_car.ravel()
        r_car, p_value_car = pearsonr(v1_car, v2_car)
        print("Car Correlation Coefficient:", r_car)
        print("Car P-value:", p_value_car)

        v1_credit = normalized_botMatrix_credit.ravel()
        v2_credit = normalized_topMatrix_credit.ravel()
        r_credit, p_value_credit = pearsonr(v1_credit, v2_credit)
        print("Credit Correlation Coefficient:", r_credit)
        print("Credit P-value:", p_value_credit)

        self.h2_1Plot(normalized_btMatrix_car, botMatrix_car, normalized_topMatrix_car,
                 topMatrix_car, 'car')

        self.h2_1Plot(normalized_botMatrix_credit, botMatrix_credit, normalized_topMatrix_credit,
                     topMatrix_credit, 'credit')


        # h2.2 think time between two consecutive interactions across top and bottom
        botTTime_car = list(bot_thinking_car.dropna())
        topTTime_car = list(top_thinking_car.dropna())
        botTTime_credit = list(bot_thinking_credit.dropna())
        topTTime_credit = list(top_thinking_credit.dropna())

        self.assumpCheck(botTTime_car, topTTime_car, 'H2.2', 'Car Consecutive TTime')
        self.assumpCheck(botTTime_credit, topTTime_credit, 'H2.2', ' Credit Consecutive TTime')

        p = self.mwTest(botTTime_car, topTTime_car, 'Car Consecutive TTime')
        p = self.mwTest(botTTime_credit, topTTime_credit, 'Credit Consecutive TTime')

        self.h2_2Plot(botTTime_car, topTTime_car, botTTime_credit, topTTime_credit)

        #h2.3 Think time preceding each Interaction type across top and bottom
        # ['change_axis', 'hover', 'click', 'zoom', 'drag']
        self.h2_3thinkTimeType(bot_carData, top_carData, 'H2.3', 'Car TTime Type')
        self.h2_3thinkTimeType(bot_creditData, top_creditData, 'H2.3', 'Credit TTime Type')

        self.h2_3Plot(bot_carData, top_carData, 'car')
        self.h2_3Plot(bot_creditData, top_creditData, 'credit')

        #h2.4 counts per interaction type across top and bottom
        bot_counts_car, top_counts_car = self.h2_4InterCount(bot_carData, top_carData, 'H2.4: Car Inter Counts')
        bot_counts_credit, top_counts_credit = self.h2_4InterCount(bot_creditData, top_creditData, 'H2.4: Credit Inter Counts')

        self.h2_4Plot(bot_counts_car, top_counts_car, 'car')
        self.h2_4Plot(bot_counts_credit, top_counts_credit, 'credit')

        #h2.5 time spent across top and bottom
        self.bot_timeSpent_car, self.top_timeSpent_car = self.h2_5timeSpent(timeSpent_car, bot_userID_car, top_userID_car, 'h2.5', 'car time spent')
        self.bot_timeSpent_credit, self.top_timeSpent_credit = self.h2_5timeSpent(timeSpent_credit, bot_userID_credit, top_userID_credit, 'h2.5', 'credit time spent')

        p = self.ttest(self.bot_timeSpent_car, self.top_timeSpent_car, 'car time spent')
        p = self.ttest(self.bot_timeSpent_credit, self.top_timeSpent_credit, 'credit time spent')

        #h3.1 tratis & transition Matrx
        self.h3_1traitsTransmatrx(botMatrix_o_car, topMatrix_o_car, 'car: h3.1 O personality & sequence')
        self.h3_1traitsTransmatrx(botMatrix_c_car, topMatrix_c_car, 'car: h3.1 C personality & sequence')
        self.h3_1traitsTransmatrx(botMatrix_e_car, topMatrix_e_car, 'car: h3.1 E personality & sequence')
        self.h3_1traitsTransmatrx(botMatrix_a_car, topMatrix_a_car, 'car: h3.1 A personality & sequence')
        self.h3_1traitsTransmatrx(botMatrix_n_car, topMatrix_n_car, 'car: h3.1 N personality & sequence')

        self.h3_1traitsTransmatrx(botMatrix_o_credit, topMatrix_o_credit, 'credit: h3.1 O personality & sequence')
        self.h3_1traitsTransmatrx(botMatrix_c_credit, topMatrix_c_credit, 'credit: h3.1 C personality & sequence')
        self.h3_1traitsTransmatrx(botMatrix_e_credit, topMatrix_e_credit, 'credit: h3.1 E personality & sequence')
        self.h3_1traitsTransmatrx(botMatrix_a_credit, topMatrix_a_credit, 'credit: h3.1 A personality & sequence')
        self.h3_1traitsTransmatrx(botMatrix_n_credit, topMatrix_n_credit, 'credit: h3.1 N personality & sequence')

        # h3.2 tratis & consecutive TTime
        self.assumpCheck(highThink_carO, lowThink_carO, 'H3.2 - O', 'Car consecutive TTime')
        self.assumpCheck(highThink_carC, lowThink_carC, 'H3.2 - C', 'Car consecutive TTime')
        self.assumpCheck(highThink_carE, lowThink_carE, 'H3.2 - E', 'Car consecutive TTime')
        self.assumpCheck(highThink_carA, lowThink_carA, 'H3.2 - A', 'Car consecutive TTime')
        self.assumpCheck(highThink_carN, lowThink_carN, 'H3.2 - N', 'Car consecutive TTime')

        self.assumpCheck(highThink_creditO, lowThink_creditO, 'H3.2 - O', 'Credit consecutive TTime')
        self.assumpCheck(highThink_creditC, lowThink_creditC, 'H3.2 - C', 'Credit consecutive TTime')
        self.assumpCheck(highThink_creditE, lowThink_creditE, 'H3.2 - E', 'Credit consecutive TTime')
        self.assumpCheck(highThink_creditA, lowThink_creditA, 'H3.2 - A', 'Credit consecutive TTime')
        self.assumpCheck(highThink_creditN, lowThink_creditN, 'H3.2 - N', 'Credit consecutive TTime')

        p = self.mwTest(highThink_carO, lowThink_carO, 'H3.2 - O Car consecutive TTime')
        p = self.mwTest(highThink_carC, lowThink_carC, 'H3.2 - C Car consecutive TTime')
        p = self.mwTest(highThink_carE, lowThink_carE, 'H3.2 - E Car consecutive TTime')
        p = self.mwTest(highThink_carA, lowThink_carA, 'H3.2 - A Car consecutive TTime')
        p = self.mwTest(highThink_carN, lowThink_carN, 'H3.2 - N Car consecutive TTime')


        p = self.mwTest(highThink_creditO, lowThink_creditO, 'H3.2 - O Credit consecutive TTime')
        p = self.mwTest(highThink_creditC, lowThink_creditC, 'H3.2 - C Credit consecutive TTime')
        p = self.mwTest(highThink_creditE, lowThink_creditE, 'H3.2 - E Credit consecutive TTime')
        p = self.mwTest(highThink_creditA, lowThink_creditA, 'H3.2 - A Credit consecutive TTime')
        p = self.mwTest(highThink_creditN, lowThink_creditN, 'H3.2 - N Credit consecutive TTime')


        #h3.2 mixed effect model:
        self.h3_2MixedEM(df_car, bot_userID_car, top_userID_car, 'car', o_high_IDs, o_low_IDs,c_high_IDs, c_low_IDs,e_high_IDs,
                         e_low_IDs,a_high_IDs, a_low_IDs,n_high_IDs, n_low_IDs,)
        self.h3_2MixedEM(df_credit, bot_userID_credit, top_userID_credit, 'credit', o_high_IDs, o_low_IDs, c_high_IDs, c_low_IDs,
                         e_high_IDs, e_low_IDs, a_high_IDs, a_low_IDs, n_high_IDs, n_low_IDs, )

        # h3.2 regression analysis: not cutting out the average
        self.h3_2MixedEM_noCutting(df_car, sum_dic, 'car')
        self.h3_2MixedEM_noCutting(df_credit, sum_dic, 'credit')

        # h3.3 regression analysis: not cutting out the average
        self.h3_3MixedEM_noCutting(df_car, sum_dic, 'car')
        self.h3_3MixedEM_noCutting(df_credit, sum_dic, 'credit')

        # h3.3 tratis & TTime per type
        # ['change_axis', 'zoom', 'hover', 'drag', 'click']
        #mixed effect model
        self.h3_3MixedEM(df_car, bot_userID_car, top_userID_car, 'car', o_high_IDs, o_low_IDs,c_high_IDs, c_low_IDs,e_high_IDs,
                         e_low_IDs,a_high_IDs, a_low_IDs,n_high_IDs, n_low_IDs,)
        self.h3_3MixedEM(df_credit, bot_userID_credit, top_userID_credit, 'credit', o_high_IDs, o_low_IDs, c_high_IDs, c_low_IDs,
                         e_high_IDs, e_low_IDs, a_high_IDs, a_low_IDs, n_high_IDs, n_low_IDs, )

        self.h3_3traitsTTtimeType(df_car, o_high_IDs, o_low_IDs, 'H3.3 - O', 'car')
        self.h3_3traitsTTtimeType(df_car, c_high_IDs, c_low_IDs, 'H3.3 - C', 'car')
        self.h3_3traitsTTtimeType(df_car, e_high_IDs, e_low_IDs, 'H3.3 - E', 'car')
        self.h3_3traitsTTtimeType(df_car, a_high_IDs, a_low_IDs, 'H3.3 - A', 'car')
        self.h3_3traitsTTtimeType(df_car, n_high_IDs, n_low_IDs, 'H3.3 - N', 'car')

        self.h3_3traitsTTtimeType(df_credit, o_high_IDs, o_low_IDs, 'H3.3 - O', 'credit')
        self.h3_3traitsTTtimeType(df_credit, c_high_IDs, c_low_IDs, 'H3.3 - C', 'credit')
        self.h3_3traitsTTtimeType(df_credit, e_high_IDs, e_low_IDs, 'H3.3 - E', 'credit')
        self.h3_3traitsTTtimeType(df_credit, a_high_IDs, a_low_IDs, 'H3.3 - A', 'credit')
        self.h3_3traitsTTtimeType(df_credit, n_high_IDs, n_low_IDs, 'H3.3 - N', 'credit')


        # h3.4 tratis & inter counts
        #mixed effect model:
        self.h3_4MixedEM(df_car, bot_userID_car, top_userID_car, 'car', o_high_IDs, o_low_IDs, c_high_IDs, c_low_IDs,
                         e_high_IDs,
                         e_low_IDs, a_high_IDs, a_low_IDs, n_high_IDs, n_low_IDs, )
        self.h3_4MixedEM(df_credit, bot_userID_credit, top_userID_credit, 'credit', o_high_IDs, o_low_IDs, c_high_IDs, c_low_IDs,
                         e_high_IDs, e_low_IDs, a_high_IDs, a_low_IDs, n_high_IDs, n_low_IDs, )

        self.h3_4traitsInterCount(df_car, o_high_IDs, o_low_IDs, 'H3.4 - O', 'car')
        self.h3_4traitsInterCount(df_car, c_high_IDs, c_low_IDs, 'H3.4 - C', 'car')
        self.h3_4traitsInterCount(df_car, e_high_IDs, e_low_IDs, 'H3.4 - E', 'car')
        self.h3_4traitsInterCount(df_car, a_high_IDs, a_low_IDs, 'H3.4 - A', 'car')
        self.h3_4traitsInterCount(df_car, n_high_IDs, n_low_IDs, 'H3.4 - N', 'car')

        self.h3_4traitsInterCount(df_credit, o_high_IDs, o_low_IDs, 'H3.4 - O', 'credit')
        self.h3_4traitsInterCount(df_credit, c_high_IDs, c_low_IDs, 'H3.4 - C', 'credit')
        self.h3_4traitsInterCount(df_credit, e_high_IDs, e_low_IDs, 'H3.4 - E', 'credit')
        self.h3_4traitsInterCount(df_credit, a_high_IDs, a_low_IDs, 'H3.4 - A', 'credit')
        self.h3_4traitsInterCount(df_credit, n_high_IDs, n_low_IDs, 'H3.4 - N', 'credit')


        # h4 traits & difference score
        traitsList = ['O', 'C', 'E', 'A', 'N']
        self.h4_traitsDiffScore(differenceScore_car, sum_dic, traitsList, 'H4', 'car')
        self.h4_traitsDiffScore(differenceScore_credit, sum_dic, traitsList, 'H4', 'credit')


        # h5 domain & difference score - Test 57
        self.h5_diffScoreDomain(familiarity_car, familiarity_credit, domainFami_agg)
        self.h5Plot(familiarity_car, familiarity_credit, domainFami_agg)


        # h6 vis literacy & difference score
        self.h6_diffScoreLiteracy(differenceScore_car, visScore, 'car')
        self.h6_diffScoreLiteracy(differenceScore_credit, visScore, 'credit')

        # h7 vis literacy & domain familiarity
        self.domainLiteracy(familiarity_car, visScore, 'H7 literacy & domain', 'car')
        self.domainLiteracy(familiarity_credit, visScore, 'H7 literacy & domain', 'credit')

        # h8 DK susceptibility
        mae_car, mae_credit = self.calculate_mae(differenceScore_car, differenceScore_credit)

    def assumpCheck(self, bot_data, top_data, assumpNum, task):

        print('-------------' + assumpNum + ' ' + task + '---------------')
        stat1, p1 = stats.shapiro(bot_data)
        stat2, p2 = stats.shapiro(top_data)
        stat, p = stats.levene(bot_data, top_data)

        # check out the distributional similarity: p-value > .05 --> same
        D_stat, p_value = stats.ks_2samp(bot_data, top_data)

        print("###Conclusion###")
        if p1 > 0.05 and p2 > 0.05 and p > 0.05:
            print(task + " - normally distributed and equal variances ->  independent samples t-test")
        elif p1 > 0.05 and p2 > 0.05 and p < 0.05:
            print(task + " - normally distributed and unequal variances ->  Welch's t-test")
        elif p1 < 0.05 and p2 < 0.05 and p_value > 0.05:
            print(task + " - NOT normally distributed but in a similar shape ->  Mann-Whitney U test")
        else:
            print("Need to choose other tests.")

    def ttest(self, value1, value2, task):
        stat, p = stats.ttest_ind(value1, value2)
        print('-------------' + task + '---------------')
        print('Ttest: Statistics=%.3f, p=%.3f' % (stat, p))
        corrected_alpha = alpha / num_test
        print('Corrected alpha: corrected_alpha=%.3f' % corrected_alpha)
        return p


    def mwTest(self, value1, value2, task):
        stat, p = mannwhitneyu(value1, value2)
        print('-------------' + task + '---------------')
        print('MW: Statistics=%.3f, p=%.3f' % (stat, p))
        corrected_alpha = alpha / num_test
        print('Corrected alpha: corrected_alpha=%.3f' % corrected_alpha)
        return p

    def chiTest(self, combined_matrix, task):
        chi2, p, dof, ex = chi2_contingency(combined_matrix)
        print('-------------' + task + '---------------')
        print('chi: chi2=%.3f, p=%.3f' % (chi2, p))
        corrected_alpha = alpha / num_test
        print('Corrected alpha: corrected_alpha=%.3f' % corrected_alpha)
        return p

    def calculate_mae(self, differenceScore_car, differenceScore_credit):
        differences_car = [abs(i) for i in list(differenceScore_car.values())]
        mae_car = sum(differences_car) / len(differences_car)

        differences_credit = [abs(i) for i in list(differenceScore_credit.values())]
        mae_credit = sum(differences_credit) / len(differences_credit)
        print(f'mae_car: {mae_car}')
        print(f'mae_credit: {mae_credit}')

        if mae_car > mae_credit:
            print("DKE susceptibility is higher for car.")
        elif mae_credit > mae_car:
            print("DKE susceptibility is higher for credit.")
        else:
            print("DKE susceptibility is equal for both tasks.")
        return mae_car, mae_credit

    def h2_3thinkTimeType(self, bot_data, top_data, assumpNum, task):
        interaction_types = bot_data['event_type'].unique()

        for type in interaction_types:  #['change_axis', 'hover', 'click', 'zoom', 'drag']
            bot_think_time = bot_data[bot_data['event_type'] == type]['thinking'].replace(0, np.nan).dropna()
            top_think_time = top_data[top_data['event_type'] == type]['thinking'].replace(0, np.nan).dropna()

            #checkout the assumption:
            if len(bot_think_time) ==0 or len(top_think_time) == 0:
                print(f"Type {type}: length is 0")
                continue
            self.assumpCheck(bot_think_time, top_think_time, assumpNum, task)
            stat, p_value= stats.mannwhitneyu(bot_think_time, top_think_time)

            print(f"For interaction type {type}, statistic={stat:.3f}, p-value={p_value:.3f}")
            corrected_alpha = alpha / num_test
            print('Corrected alpha: corrected_alpha=%.3f' % corrected_alpha)

            if p_value < 0.05:
                print('Significant different')
            else:
                print('Non-significant')

    # Interaction Pace --> counts per interaction type:
    def h2_4InterCount(self, bot_data, top_data, task):
        bot_counts = {}
        for i, event_type in enumerate(bot_data['event_type'].unique()):
            bot_counts[event_type] = (bot_data['event_type'] == event_type).sum()

        top_counts = {}
        for i, event_type in enumerate(bot_data['event_type'].unique()):
            top_counts[event_type] = (top_data['event_type'] == event_type).sum()

        s1 = pd.Series(bot_counts, name='Bottom')
        s2 = pd.Series(top_counts, name='Top')
        contingency_table = pd.concat([s1, s2], axis=1)

        p = self.chiTest(contingency_table, task)
        return bot_counts, top_counts

    def h2_5timeSpent(self, timeSpent_car, bot_userID_car, top_userID_car, assumpNum, task):
        bot_timeSpent_car = [timeSpent_car[key] for key in bot_userID_car]
        top_timeSpent_car = [timeSpent_car[key] for key in top_userID_car]

        self.assumpCheck(bot_timeSpent_car, top_timeSpent_car, assumpNum, task)
        return bot_timeSpent_car, top_timeSpent_car

    def h3_1traitsTransmatrx(self, botMatrix, topMatrix, task):
        normalize_matrix_bot = self.normalize_transition_matrix(botMatrix)
        normalize_matrix_top = self.normalize_transition_matrix(topMatrix)
        combined_matrix = np.vstack((normalize_matrix_bot.sum(axis=0), normalize_matrix_top.sum(axis=0)))
        p = self.chiTest(combined_matrix, task)

    def h3_2MixedEM(self, df_car, bot_userID_car, top_userID_car, task, o_high_IDs, o_low_IDs,c_high_IDs, c_low_IDs,e_high_IDs, e_low_IDs,a_high_IDs, a_low_IDs,n_high_IDs, n_low_IDs):

        mixed_dfCar = df_car[df_car['user_id'].isin([int(i) for i in bot_userID_car + top_userID_car])]

        mixed_dfCar['quartile'] = np.where(mixed_dfCar['user_id'].isin([int(i) for i in bot_userID_car]), 'bottom',
                                           np.where(mixed_dfCar['user_id'].isin([int(i) for i in top_userID_car]),
                                                    'top', 'average'))
        mixed_dfCar['Trait_o'] = np.where(mixed_dfCar['user_id'].isin([int(i) for i in o_low_IDs]), 'low',
                                          np.where(mixed_dfCar['user_id'].isin([int(i) for i in o_high_IDs]), 'high',
                                                   'average'))
        mixed_dfCar['Trait_c'] = np.where(mixed_dfCar['user_id'].isin([int(i) for i in c_low_IDs]), 'low',
                                          np.where(mixed_dfCar['user_id'].isin([int(i) for i in c_high_IDs]), 'high',
                                                   'average'))
        mixed_dfCar['Trait_e'] = np.where(mixed_dfCar['user_id'].isin([int(i) for i in e_low_IDs]), 'low',
                                          np.where(mixed_dfCar['user_id'].isin([int(i) for i in e_high_IDs]), 'high',
                                                   'average'))
        mixed_dfCar['Trait_a'] = np.where(mixed_dfCar['user_id'].isin([int(i) for i in a_low_IDs]), 'low',
                                          np.where(mixed_dfCar['user_id'].isin([int(i) for i in a_high_IDs]), 'high',
                                                   'average'))
        mixed_dfCar['Trait_n'] = np.where(mixed_dfCar['user_id'].isin([int(i) for i in n_low_IDs]), 'low',
                                          np.where(mixed_dfCar['user_id'].isin([int(i) for i in n_high_IDs]), 'high',
                                                   'average'))
        model = smf.mixedlm(
            "thinking ~ C(Trait_o) + C(Trait_c) + C(Trait_e) + C(Trait_a) + C(Trait_n) + C(quartile)", mixed_dfCar,
            groups=mixed_dfCar["user_id"])
        result = model.fit()

        print('-----' + task + ': h3.2 mixed effect model-----')
        print(result.summary())

    #not cutting out average:
    def h3_2MixedEM_noCutting(self, df_car, dic_traitsScore, task):

        mixed_dfCar = df_car

        # [O,C,E,A,N]
        dic_traitsScore = {int(k): v for k, v in dic_traitsScore.items()}
        mixed_dfCar['Trait_o'] = mixed_dfCar['user_id'].map(lambda x: dic_traitsScore.get(x, [None])[0])
        mixed_dfCar['Trait_c'] = mixed_dfCar['user_id'].map(lambda x: dic_traitsScore.get(x, [None])[1])
        mixed_dfCar['Trait_e'] = mixed_dfCar['user_id'].map(lambda x: dic_traitsScore.get(x, [None])[2])
        mixed_dfCar['Trait_a'] = mixed_dfCar['user_id'].map(lambda x: dic_traitsScore.get(x, [None])[3])
        mixed_dfCar['Trait_n'] = mixed_dfCar['user_id'].map(lambda x: dic_traitsScore.get(x, [None])[4])
        mixed_dfCar.dropna(subset=['thinking', 'Trait_o', 'Trait_c', 'Trait_e', 'Trait_a', 'Trait_n'], inplace=True)
        formula = "thinking ~ Trait_o + Trait_c + Trait_e + Trait_a + Trait_n + (1|user_id)"
        model = smf.mixedlm(formula, mixed_dfCar,
            groups=mixed_dfCar["user_id"])
        result = model.fit()

        print('-----' + task + ': h3.2 mixed effect model not cutting out average-----')
        print(result.summary())


    def h3_3MixedEM_noCutting(self, df_car, dic_traitsScore, task):

        mixed_dfCar = df_car

        # [O,C,E,A,N]
        dic_traitsScore = {int(k): v for k, v in dic_traitsScore.items()}
        mixed_dfCar['Trait_o'] = mixed_dfCar['user_id'].map(lambda x: dic_traitsScore.get(x, [None])[0])
        mixed_dfCar['Trait_c'] = mixed_dfCar['user_id'].map(lambda x: dic_traitsScore.get(x, [None])[1])
        mixed_dfCar['Trait_e'] = mixed_dfCar['user_id'].map(lambda x: dic_traitsScore.get(x, [None])[2])
        mixed_dfCar['Trait_a'] = mixed_dfCar['user_id'].map(lambda x: dic_traitsScore.get(x, [None])[3])
        mixed_dfCar['Trait_n'] = mixed_dfCar['user_id'].map(lambda x: dic_traitsScore.get(x, [None])[4])
        mixed_dfCar.dropna(subset=['thinking', 'Trait_o', 'Trait_c', 'Trait_e', 'Trait_a', 'Trait_n'], inplace=True)
        formula = "thinking ~ Trait_o * C(event_type) + Trait_c * C(event_type) + Trait_e * C(event_type) + Trait_a * C(event_type) + Trait_n * C(event_type) + (1 | user_id)"

        model = smf.mixedlm(formula, mixed_dfCar,
            groups=mixed_dfCar["user_id"])
        result = model.fit()

        print('-----' + task + ': h3.3 mixed effect model not cutting out average-----')
        print(result.summary())


    # TTime per interaction type:
    def h3_3MixedEM(self, df_car, bot_userID_car, top_userID_car, task, o_high_IDs, o_low_IDs,c_high_IDs, c_low_IDs,e_high_IDs, e_low_IDs,a_high_IDs, a_low_IDs,n_high_IDs, n_low_IDs):

        mixed_dfCar = df_car[df_car['user_id'].isin([int(i) for i in bot_userID_car + top_userID_car])]

        mixed_dfCar['quartile'] = np.where(mixed_dfCar['user_id'].isin([int(i) for i in bot_userID_car]), 'bottom',
                                           np.where(mixed_dfCar['user_id'].isin([int(i) for i in top_userID_car]),
                                                    'top', 'average'))
        mixed_dfCar['Trait_o'] = np.where(mixed_dfCar['user_id'].isin([int(i) for i in o_low_IDs]), 'low',
                                          np.where(mixed_dfCar['user_id'].isin([int(i) for i in o_high_IDs]), 'high',
                                                   'average'))
        mixed_dfCar['Trait_c'] = np.where(mixed_dfCar['user_id'].isin([int(i) for i in c_low_IDs]), 'low',
                                          np.where(mixed_dfCar['user_id'].isin([int(i) for i in c_high_IDs]), 'high',
                                                   'average'))
        mixed_dfCar['Trait_e'] = np.where(mixed_dfCar['user_id'].isin([int(i) for i in e_low_IDs]), 'low',
                                          np.where(mixed_dfCar['user_id'].isin([int(i) for i in e_high_IDs]), 'high',
                                                   'average'))
        mixed_dfCar['Trait_a'] = np.where(mixed_dfCar['user_id'].isin([int(i) for i in a_low_IDs]), 'low',
                                          np.where(mixed_dfCar['user_id'].isin([int(i) for i in a_high_IDs]), 'high',
                                                   'average'))
        mixed_dfCar['Trait_n'] = np.where(mixed_dfCar['user_id'].isin([int(i) for i in n_low_IDs]), 'low',
                                          np.where(mixed_dfCar['user_id'].isin([int(i) for i in n_high_IDs]), 'high',
                                                   'average'))

        formula = """thinking ~ C(Trait_o) * C(event_type) + C(Trait_c) * C(event_type) +
                                C(Trait_e) * C(event_type) + C(Trait_a) * C(event_type) + 
                                C(Trait_n) * C(event_type) + C(quartile)"""

        # Mixed effects model fit
        mixed = smf.mixedlm(formula, mixed_dfCar, groups=mixed_dfCar['user_id'])
        mixed_fit = mixed.fit()
        print('-----' + task + ': h3.3 mixed effect model -----')
        print(mixed_fit.summary())

    # Inter counts per type:
    def h3_4MixedEM(self, df_car, bot_userID_car, top_userID_car, task, o_high_IDs, o_low_IDs,c_high_IDs, c_low_IDs,e_high_IDs, e_low_IDs,a_high_IDs, a_low_IDs,n_high_IDs, n_low_IDs):

        mixed_dfCar = df_car[df_car['user_id'].isin([int(i) for i in bot_userID_car + top_userID_car])]

        df_counts = mixed_dfCar.groupby(['user_id', 'event_type']).size().reset_index(name='counts')

        df_counts['quartile'] = np.where(df_counts['user_id'].isin([int(i) for i in bot_userID_car]), 'bottom',
                                           np.where(df_counts['user_id'].isin([int(i) for i in top_userID_car]),
                                                    'top', 'average'))
        df_counts['Trait_o'] = np.where(df_counts['user_id'].isin([int(i) for i in o_low_IDs]), 'low',
                                          np.where(df_counts['user_id'].isin([int(i) for i in o_high_IDs]), 'high',
                                                   'average'))
        df_counts['Trait_c'] = np.where(df_counts['user_id'].isin([int(i) for i in c_low_IDs]), 'low',
                                          np.where(df_counts['user_id'].isin([int(i) for i in c_high_IDs]), 'high',
                                                   'average'))
        df_counts['Trait_e'] = np.where(df_counts['user_id'].isin([int(i) for i in e_low_IDs]), 'low',
                                          np.where(df_counts['user_id'].isin([int(i) for i in e_high_IDs]), 'high',
                                                   'average'))
        df_counts['Trait_a'] = np.where(df_counts['user_id'].isin([int(i) for i in a_low_IDs]), 'low',
                                          np.where(df_counts['user_id'].isin([int(i) for i in a_high_IDs]), 'high',
                                                   'average'))
        df_counts['Trait_n'] = np.where(df_counts['user_id'].isin([int(i) for i in n_low_IDs]), 'low',
                                          np.where(df_counts['user_id'].isin([int(i) for i in n_high_IDs]), 'high',
                                                   'average'))
        model = smf.mixedlm(
            "counts ~ C(Trait_o) * C(event_type) + C(Trait_c) * C(event_type) + C(Trait_e) * C(event_type) + C(Trait_a) * C(event_type) + C(Trait_n)  * C(event_type)", df_counts,
            groups=df_counts["user_id"])
        result = model.fit()

        print('-----' + task + ': h3.4 mixed effect model-----')
        print(result.summary())


    def h3_3traitsTTtimeType(self, df, lowScore_ID, highScore_ID, assumpNum, task):
        lowScore_ID = [int(id) for id in lowScore_ID]
        highScore_ID = [int(id) for id in highScore_ID]
        low_data = df[df['user_id'].isin(lowScore_ID)]
        high_data = df[df['user_id'].isin(highScore_ID)]

        interaction_types = low_data['event_type'].unique()

        for type in interaction_types:  #['change_axis', 'zoom', 'hover', 'drag', 'click']
            bot_think_time = low_data[low_data['event_type'] == type]['thinking'].dropna()
            top_think_time = high_data[high_data['event_type'] == type]['thinking'].dropna()

            if len(bot_think_time) < 3 or len(top_think_time) < 3:
                print(f"Type {type} lenght is < 3")
                continue

            #checkout the assumption:
            self.assumpCheck(bot_think_time, top_think_time, assumpNum, task)
            print('-' + type + '-')
            p = self.mwTest(bot_think_time, top_think_time, task)
            print("Bot think time mean: {:.3f}".format(bot_think_time.mean()))
            print("Top think time mean: {:.3f}".format(top_think_time.mean()))

    def h3_4traitsInterCount(self, df, lowScore_ID, highScore_ID, assumpNum, task):
        lowScore_ID = [int(id) for id in lowScore_ID]
        highScore_ID = [int(id) for id in highScore_ID]
        low_data = df[df['user_id'].isin(lowScore_ID)]
        high_data = df[df['user_id'].isin(highScore_ID)]

        bot_counts = {}
        for i, event_type in enumerate(low_data['event_type'].unique()):
            bot_counts[event_type] = (low_data['event_type'] == event_type).sum()

        top_counts = {}
        for i, event_type in enumerate(high_data['event_type'].unique()):
            top_counts[event_type] = (high_data['event_type'] == event_type).sum()

        all_keys = set(bot_counts.keys()).union(set(top_counts.keys()))
        for key in all_keys:
            bot_counts.setdefault(key, 0)
            top_counts.setdefault(key, 0)

        self.assumpCheck(list(bot_counts.values()), list(top_counts.values()), assumpNum, task)
        s1 = pd.Series(bot_counts, name='Bottom')
        s2 = pd.Series(top_counts, name='Top')
        contingency_table = pd.concat([s1, s2], axis=1)
        p = self.chiTest(contingency_table, task)
        print("Bot interaction counts per type:", bot_counts)
        print("Top interaction counts per type:", top_counts)

    def h4_traitsDiffScore(self, differenceScore, sum_dic, traitsList, assumpNum, task):
        DiffScore_values = [differenceScore[key] for key in sum_dic.keys()]

        fig, axs = plt.subplots(1, 5, figsize=(25, 5))

        for i in range(len(next(iter(sum_dic.values())))):
            traitsScore_values = [value[i] for value in sum_dic.values()]

            #checkout the distribution of two variables:
            stat1, p1 = stats.shapiro(traitsScore_values)
            stat2, p2 = stats.shapiro(DiffScore_values)

            print('# ' + traitsList[i] + ': Check out if normal distribution:')
            print(traitsList[i] + ': Statistics=%.3f, p=%.3f' % (stat1, p1))
            print('DiffScore: Statistics=%.3f, p=%.3f' % (stat2, p2))
            if p1 > 0.05 and p2 >0.05:
                print('Use Pearson correlation')
            else:
                print('Spearman\'s Correlation')

            correlation, p_value = pearsonr(traitsScore_values, DiffScore_values)
            print('--------' + assumpNum + ' ' + task + '---------')
            print(f'Correlation between values in traits {traitsList[i]} and DiffScore: {correlation:.3f}, p-value: {p_value:.3f}')
            corrected_alpha = alpha / num_test
            print('Corrected alpha: corrected_alpha=%.3f' % corrected_alpha)

            sns.regplot(x=traitsScore_values, y=DiffScore_values, ax=axs[i])
            axs[i].set_xlabel(personalityName[i] + 'Trait Score', fontsize=16)
            axs[i].set_ylabel('Δ = EstimatedPercentile - ActualPercentile',
                              fontsize=13,)
            axs[i].set_title(f"r({len(traitsScore_values) - 2}) = {correlation:.3f}, p = {p_value:.3f}", fontsize=15)

        plt.tight_layout()
        title = 'h4diffScorePerson_' + task
        plt.savefig(os.path.join(self.root, title + " .jpg"), bbox_inches='tight', dpi=300)
        plt.show()


    def h5_diffScoreDomain(self, familiarity_car, familiarity_credit, domainFami_agg):

        differenceScore_car = [row[0] for row in list(familiarity_car.values())]
        famiScore_car = [row[2] for row in list(familiarity_car.values())]

        correlation_car, p_value_domian_car = stats.pearsonr(differenceScore_car, famiScore_car)
        # p_values.append(p_value_domian_car)
        print('-------------H5 domain & difference score: Car---------------')
        print(f'Correlation: {correlation_car:.3f}')
        print(f'p-value: {p_value_domian_car:.3f}')
        corrected_alpha = alpha / num_test
        print('Corrected alpha: corrected_alpha=%.3f' % corrected_alpha)

        differenceScore_credit = [row[0] for row in list(familiarity_credit.values())]
        famiScore_credit = [row[2] for row in list(familiarity_credit.values())]
        correlation_credit, p_value_domian_credit = stats.pearsonr(differenceScore_credit, famiScore_credit)
        # p_values_credit.append(p_value_domian_credit)
        print('-------------H5 domain & difference score: Credit---------------')
        print(f'Correlation: {correlation_credit:.3f}')
        print(f'p-value: {p_value_domian_credit:.3f}')
        corrected_alpha = alpha / num_test
        print('Corrected alpha: corrected_alpha=%.3f' % corrected_alpha)

        diffScore_agg= [row[0] for row in list(domainFami_agg.values())]
        famiScore_agg = [row[2] for row in list(domainFami_agg.values())]
        correlation_agg, p_value_domian_agg = stats.pearsonr(diffScore_agg, famiScore_agg)
        print('-------------H5 domain & difference score: agg---------------')
        print(f'Correlation: {correlation_agg:.3f}')
        print(f'p-value: {p_value_domian_agg:.3f}')
        corrected_alpha = alpha / num_test
        print('Corrected alpha: corrected_alpha=%.3f' % corrected_alpha)


    def h6_diffScoreLiteracy(self, differenceScore_car, visScore, task):
        keys = differenceScore_car.keys()
        difference_score_values = [differenceScore_car[key] for key in keys]
        vis_score_values = [visScore[key] for key in keys]

        # Calculate the Pearson correlation coefficient
        correlation, pvalue = pearsonr(difference_score_values, vis_score_values)

        print("------------H6: " + task + "--------------")
        print("Correlation coefficient: ", correlation)
        print("P-value: ", pvalue)
        corrected_alpha = alpha / num_test
        print('Corrected alpha: corrected_alpha=%.3f' % corrected_alpha)

        self.h6Plot(vis_score_values, difference_score_values, task)

    def domainLiteracy(self, familiarity_car, visScore, assumpNum, task):

        famiScore_car = [familiarity_car[key][2] for key in list(visScore.keys())]
        stat1, p1 = stats.shapiro(list(visScore.values()))
        stat2, p2 = stats.shapiro(famiScore_car)
        self.assumpCheck(list(visScore.values()), famiScore_car, assumpNum, task)
        correlation_car, p_value_domian_car = stats.pearsonr(list(visScore.values()), famiScore_car)
        print(f'Correlation: {correlation_car:.3f}')
        print(f'p-value: {p_value_domian_car:.3f}')
        corrected_alpha = alpha / num_test
        print('Corrected alpha: corrected_alpha=%.3f' % corrected_alpha)


    def top_botBoxPlot(self, bot_car, top_car, bot_credit, top_credit, ylabel, title):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        car_data = [bot_car, top_car]
        credit_data = [bot_credit, top_credit]
        ax1.boxplot(car_data)
        ax2.boxplot(credit_data)
        ax1.set_xticklabels(['Bottom Quartile', 'Top Quartile'])
        ax2.set_xticklabels(['Bottom Quartile', 'Top Quartile'])
        ax1.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)
        ax1.set_title(ylabel + ' Across Bot and Top Quartiles in Car')
        ax2.set_title(ylabel + ' Across Bot and Top Quartiles in Credit')
        ax1.grid(True, color='gray', linestyle='--', linewidth=0.3)
        ax2.grid(True, color='gray', linestyle='--', linewidth=0.3)
        plt.subplots_adjust(wspace=0.4)
        title = title
        plt.savefig(os.path.join(self.root, title + " .jpg"), bbox_inches='tight', dpi=300)
        plt.show()

    def top_botViolinPlot(self, bot_car, top_car, bot_credit, top_credit, ylabel, title):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        car_data = [bot_car, top_car]
        credit_data = [bot_credit, top_credit]

        axs[0].set_title('Car: Think Time - Top vs Bottom Quartile')
        sns.violinplot(data=car_data, palette="Set3", ax=axs[0])
        # axs[0].set_xlabel('Quartile')
        axs[0].set_xticklabels(['Bottom Quartile', 'Top Quartile'])
        axs[0].set_ylabel(ylabel)

        axs[1].set_title('Credit: Think Time - Top vs Bottom Quartile')
        sns.violinplot(data=credit_data, palette="Set3", ax=axs[1])
        # axs[1].set_xlabel('Quartile')
        axs[1].set_xticklabels(['Bottom Quartile', 'Top Quartile'])
        axs[1].set_ylabel(ylabel)

        plt.subplots_adjust(wspace=0.4)
        title = title
        plt.savefig(os.path.join(self.root, title + " .jpg"), bbox_inches='tight', dpi=300)
        plt.show()

    def taskDifficultyPlot(self, acc_car, acc_credit):

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        acc_data = [acc_car, acc_credit]
        ax.boxplot(acc_data)
        ax.set_xticklabels(['AccCAr', 'AccCredit'])
        ax.set_ylabel('Task Accuracy')
        ax.set_title('Task Accuracy Across Two Tasks')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.3)
        title = 'taskDifficulty'
        plt.savefig(os.path.join(self. root, title + ".jpg"), bbox_inches='tight', dpi=300)
        plt.show()

    # normalized within each matrix:
    def normalize_transition_matrix(self, matrix):
        row_sums = matrix.sum(axis=1)
        normalized_matrix = matrix / row_sums[:, np.newaxis]
        return normalized_matrix

    def h1Plot(self, H1_bot_car, H1_top_car, task):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        acc_data = [H1_bot_car, H1_top_car]
        ax.boxplot(acc_data)
        ax.set_xticklabels(['Bottom', 'Top'])
        ax.set_ylabel('DiffScore')
        ax.set_title(task + ': DiffScore Across Top & Bottom Quartile')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.3)
        title = 'h1DiffScore'
        plt.savefig(os.path.join(self.root, title + '_' + task + ".jpg"), bbox_inches='tight', dpi=300)
        plt.show()

    def h2_1Plot(self, normalized_btMatrix_car, botMatrix_car, normalized_topMatrix_car, topMatrix_car, task):


        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        sns.heatmap(normalized_btMatrix_car, annot=botMatrix_car, cmap="Blues", cbar=False, fmt="d", ax=axs[0])
        axs[0].set_xticklabels(['drag', 'hover', 'click', 'zoom', 'axis change'], fontsize=13)
        axs[0].set_yticklabels(['drag', 'hover', 'click', 'zoom', 'axis change'], fontsize=13)
        axs[0].set_xlabel('To', fontsize=13, fontweight='bold')
        axs[0].set_ylabel('From', fontsize=13, fontweight='bold')

        from matplotlib.colors import ListedColormap
        sns.heatmap(normalized_topMatrix_car, annot=topMatrix_car, cmap="Blues", cbar=False, fmt="d", ax=axs[1])


        axs[1].set_xticklabels(['drag', 'hover', 'click', 'zoom', 'axis change'], fontsize=13)
        axs[1].set_yticklabels(['drag', 'hover', 'click', 'zoom', 'axis change'], fontsize=13)
        axs[1].set_xlabel('To', fontsize=13, fontweight='bold')
        axs[1].set_ylabel('From', fontsize=13, fontweight='bold')


        plt.subplots_adjust(hspace=0.1)
        plt.tight_layout()
        title = 'h2Matrix_'
        if task == 'car':
            cbar_ticks = [0, 0.2, 0.4, 0.6, 0.8]
            cbar_tick_labels = ["0", "70", "140", "210", "280"]
        elif task == 'credit':

            cbar_ticks = [0, 0.175, 0.35, 0.525, 0.7]
            cbar_tick_labels = ["0", "70", "140", "210", "280"]
        cbar = fig.colorbar(axs[0].collections[0], ax=axs, location='right', pad=0.02)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_tick_labels)
        cbar.set_label('Value')

        # plt.tight_layout()
        plt.savefig(os.path.join(self.root, title + task + ".jpg"), bbox_inches='tight', dpi=300)
        plt.show()


    def h2_2Plot(self, botTTime_car, topTTime_car, botTTime_credit, topTTime_credit):

        self.top_botViolinPlot(botTTime_car, topTTime_car, botTTime_credit, topTTime_credit, 'Think Time', 'h2ThinkTimeConsecutive_violin')
        self.top_botBoxPlot(botTTime_car, topTTime_car, botTTime_credit, topTTime_credit, 'Think Time', 'h2ThinkTimeConsecutive_box')

    def h2_3Plot(self, bot_carData, top_carData, task):

        data_bot_drag = bot_carData[bot_carData['event_type'] == 'drag'][['thinking']].assign(event_type='Drag', Quartile='Bottom Quartile')
        data_bot_hover = bot_carData[bot_carData['event_type'] == 'hover'][['thinking']].assign(event_type='Hover', Quartile='Bottom Quartile')
        data_bot_click = bot_carData[bot_carData['event_type'] == 'click'][['thinking']].assign(event_type='Click', Quartile='Bottom Quartile')
        data_bot_changAxis = bot_carData[bot_carData['event_type'] == 'change_axis'][['thinking']].assign(event_type='change_axis', Quartile='Bottom Quartile')
        data_bot_zoom = bot_carData[bot_carData['event_type'] == 'zoom'][['thinking']].assign(event_type='zoom', Quartile='Bottom Quartile')

        data_top_drag = top_carData[top_carData['event_type'] == 'drag'][['thinking']].assign(event_type='Drag', Quartile='Top Quartile')
        data_top_hover = top_carData[top_carData['event_type'] == 'hover'][['thinking']].assign(event_type='Hover', Quartile='Top Quartile')
        data_top_click = top_carData[top_carData['event_type'] == 'click'][['thinking']].assign(event_type='Click', Quartile='Top Quartile')
        data_top_changAxis = top_carData[top_carData['event_type'] == 'change_axis'][['thinking']].assign(event_type='change_axis', Quartile='Top Quartile')
        data_top_zoom = top_carData[top_carData['event_type'] == 'zoom'][['thinking']].assign(event_type='zoom', Quartile='Top Quartile')

        # Concatenate all dataframes
        combined_data = pd.concat([data_bot_drag, data_bot_hover, data_bot_click, data_bot_changAxis, data_bot_zoom,
                                   data_top_drag, data_top_hover, data_top_click, data_top_changAxis, data_top_zoom])

        # Create the violin plot
        plt.figure(figsize=(8, 6))
        sns.violinplot(x="event_type", y="thinking", hue="Quartile", data=combined_data, split=True, inner='box', palette=sns.color_palette("Set3", desat=0.7))

        plt.title(task + ': Think Time Preceding Interaction Types')
        plt.ylabel('Think Time (Secs)')
        plt.legend(title='Quartile')
        title = 'h2ThinkTimePerType'+ '_' + task
        plt.savefig(os.path.join(self.root, title +  ".jpg"), bbox_inches='tight', dpi=300)
        plt.show()

    def h2_4Plot(self, bot_counts_car, top_counts_car, task):

        labels = list(bot_counts_car.keys())
        x = range(len(labels))

        # Plotting the first task's quartile data
        plt.bar(x, list(bot_counts_car.values()), width=0.4, label='Bottom Quartile', align='center', alpha=0.7)
        # axs[0].bar([val + bar_width for val in x], list(top_counts_car.values()), width=bar_width, label='Top Quartile', align='center', alpha=0.7)
        plt.bar(x, list(top_counts_car.values()), width=0.4, label='Top Quartile', align='edge', alpha=0.7)

        for i, value in enumerate(bot_counts_car.values()):
            plt.text(i, value, str(value), ha='center', va='bottom')

        for i, value in enumerate(top_counts_car.values()):
            plt.text(i+0.4, value, str(value), ha='center', va='bottom')

        title = 'h2.4CountPerType_Bar'+task
        plt.xlabel('Actions')
        plt.ylabel('Frequency')
        plt.title(task + ': Top and Bottom Quartile Comparison')
        plt.xticks(x, labels)
        plt.legend()
        plt.savefig(os.path.join(self.root, title + ".jpg"), bbox_inches='tight', dpi=300)
        plt.show()

    def h5Plot(self, familiarity_car, familiarity_credit, domainFami_agg):
        differenceScore_car = [row[0] for row in list(familiarity_car.values())]
        famiScore_car = [row[2] for row in list(familiarity_car.values())]

        differenceScore_credit = [row[0] for row in list(familiarity_credit.values())]
        famiScore_credit = [row[2] for row in list(familiarity_credit.values())]

        diffScore_agg= [row[0] for row in list(domainFami_agg.values())]
        famiScore_agg = [row[2] for row in list(domainFami_agg.values())]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Create regplots
        sns.regplot(x=famiScore_car, y=differenceScore_car, ax=axs[0])
        axs[0].set_title('Car')
        axs[0].set_xlabel('Domain Familiarity Score')
        axs[0].set_ylabel('Δ = EstimatedPercentile - ActualPercentile')
        axs[0].set_xticks([0, 1, 2, 3, 4, 5])

        sns.regplot(x=famiScore_credit, y=differenceScore_credit, ax=axs[1])
        axs[1].set_title('Credit')
        axs[1].set_xlabel('Domain Familiarity Score')
        axs[1].set_ylabel('Δ = EstimatedPercentile - ActualPercentile')
        axs[1].set_xticks([0, 1, 2, 3, 4, 5])

        # Show the plots
        plt.tight_layout()
        title = 'h5famiDiffScore'
        plt.savefig(os.path.join(self.root, title + " .jpg"), bbox_inches='tight', dpi=300)
        plt.show()

        #agg plot:
        x = range(0,5)
        sns.regplot(x=famiScore_agg, y=diffScore_agg)
        # plt.title('Agg - famiDiffScore')
        plt.xticks([0, 1, 2, 3, 4, 5])
        plt.xlabel('Domain Familiarity Score')
        plt.ylabel('Δ = EstimatedPercentile - ActualPercentile')
        title = 'AggH5famiDiffScore'
        plt.tight_layout()
        plt.savefig(os.path.join(self.root, title + " .jpg"), bbox_inches='tight', dpi=300)
        plt.show()


    def h6Plot(self, vis_score_values, difference_score_values, task):
        sns.regplot(x=vis_score_values, y=difference_score_values)
        # Add a horizontal dashed reference line at y=0
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(task + ': VisDiffScore')
        # plt.xticks([0, 1, 2, 3, 4, 5])
        plt.xlabel('Vis Score')
        plt.ylabel('Difference Score')
        title =  'h6VisDiffScore'
        plt.tight_layout()
        plt.savefig(os.path.join(self.root, title + task + " .jpg"), bbox_inches='tight', dpi=300)
        plt.show()


logfilepath = "./data/log/dkeffect-3776d-1690317512.json"
car_groundtruth_file = './data/car_30points_option2.csv'
credit_groundtruth_file = './data/credit_30points_option1.csv'

#get estimation data from post survey:

postSurvey = './data/survey'
colIndex = ['Q1', 'Q11', 'Q16', 'Q12_2', 'Q13_1', 'Q17_1', 'Q18_1', 'Q20_1']
domainFaText = ['Not at all familiar', 'Slightly familiar', 'Somewhat familiar', 'Moderately familiar', 'Extremely familiar']
domainFaNum = [1, 2, 3, 4, 5]

#Presurvey Info:
que = [
'Q1',
'Q4_1', 'Q4_2', 'Q4_3', 'Q4_4', 'Q4_5',
'Q5_1', 'Q5_2', 'Q5_3', 'Q5_4', 'Q5_5',
'Q6',
'Q7_1', 'Q7_2', 'Q7_3', 'Q7_4', 'Q7_5',
'Q8_1', 'Q8_2', 'Q8_3', 'Q8_4', 'Q8_5',
'Q9',
'Q10_1', 'Q10_2', 'Q10_3', 'Q10_4', 'Q10_5']

attention_check_que = ['Q1', 'Q6', 'Q9']

visLiteracy_que = ['Q1', 'Q13', 'Q16', 'Q19', 'Q22','Q25', 'Q28', 'Q31']
vis_answer = ['70.5 kg', '197.1 cm', '53.9 - 123.6 kg', '175.3 cm', 'True', 'False', 'False']
plus = ['Q4_1', 'Q4_2', 'Q4_3', 'Q4_4', 'Q4_5', 'Q7_1', 'Q7_2', 'Q7_3', 'Q7_4']
minus = ['Q5_1', 'Q5_2', 'Q5_3', 'Q5_4', 'Q5_5',  'Q7_5', 'Q8_1', 'Q8_2', 'Q8_3', 'Q8_4', 'Q8_5']

choice = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']

plus_score = [1, 2, 3, 4, 5]
minus_score = [5, 4, 3, 2, 1]
Openness = ['Q4_5', 'Q5_5', 'Q7_5', 'Q8_5']
Conscientiousness = ['Q4_3', 'Q5_3', 'Q7_3', 'Q8_3']
Extraversion = ['Q4_1', 'Q5_1', 'Q7_1', 'Q8_1']
Agreeableness = ['Q4_2', 'Q5_2', 'Q7_2', 'Q8_2']
Neuroticism = ['Q4_4', 'Q5_4', 'Q7_4', 'Q8_4']
personalityName= ['Openness','Conscientiousness','Extraversion','Agreeableness','Neuroticism']

opt_5 = ['Q13', 'Q16', 'Q19', 'Q22',]
opt_3 = ['Q25', 'Q28', 'Q31']
est_path = './data/survey/Personality Traits Questionnaire_August 3, 2023_12.50.csv'

#interaction Sequence info:
filePath_car = 'interaction_elapsed_Car.csv'
filePath_credit = 'interaction_elapsed_Credit.csv'

alpha = 0.05
num_test = 13

LogReader = ParsingLogs(logfilepath=logfilepath, car_groundtruth_file=car_groundtruth_file, credit_groundtruth_file=credit_groundtruth_file)
LogReader.dkLine(LogReader.sort_actual_estimation_car, LogReader.sort_actual_estimation_credit)

PresvyReader = PreSurvey(que, attention_check_que, visLiteracy_que, vis_answer, opt_5, opt_3, est_path, LogReader.sort_actual_estimation_credit,
                 plus, minus, Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism, plus_score, minus_score, choice)

TransMatrx = InterSequence(filePath_car, filePath_credit, LogReader.sort_actual_estimation_car, LogReader.sort_actual_estimation_credit,
                           PresvyReader.o_high_IDs, PresvyReader.o_low_IDs, PresvyReader.c_high_IDs, PresvyReader.c_low_IDs, PresvyReader.e_high_IDs,
                           PresvyReader.e_low_IDs, PresvyReader.a_high_IDs, PresvyReader.a_low_IDs, PresvyReader.n_high_IDs, PresvyReader.n_low_IDs,
                           LogReader.bot_userID_car, LogReader.top_userID_car, LogReader.bot_userID_credit, LogReader.top_userID_credit,)

ThinkTime = InterPace(filePath_car, filePath_credit, LogReader.bot_userID_car, LogReader.top_userID_car, LogReader.bot_userID_credit, LogReader.top_userID_credit,
                      PresvyReader.o_high_IDs, PresvyReader.o_low_IDs, PresvyReader.c_high_IDs, PresvyReader.c_low_IDs, PresvyReader.e_high_IDs, PresvyReader.e_low_IDs, PresvyReader.a_high_IDs,
                      PresvyReader.a_low_IDs, PresvyReader.n_high_IDs, PresvyReader.n_low_IDs)

#eyetracking Info:
filePath_car = './result/eyeTracking/Click_car.csv'
filePath_credit = './result/eyeTracking/Click_credit.csv'
backgroundPath_car = "./data/car.png"
backgroundPath_credit = "./data/credit.png"
output = './result/eyeTracking/heatmap/'
map = AttentionMap(LogReader.Uids, LogReader.data, LogReader.bot_userID_car, LogReader.top_userID_car, LogReader.bot_userID_credit, LogReader.top_userID_credit,
                   LogReader.inter_car_df, LogReader.inter_credit_df,
                   PresvyReader.o_high_IDs, PresvyReader.o_low_IDs, PresvyReader.c_high_IDs, PresvyReader.c_low_IDs,
                   PresvyReader.e_high_IDs, PresvyReader.e_low_IDs, PresvyReader.a_high_IDs,
                   PresvyReader.a_low_IDs, PresvyReader.n_high_IDs, PresvyReader.n_low_IDs)

stats = StaTests(alpha, num_test, LogReader.sort_actual_estimation_car, LogReader.sort_actual_estimation_credit, LogReader.differenceScore_car, LogReader.differenceScore_credit,
                 LogReader.bot_userID_car, LogReader.top_userID_car, LogReader.bot_userID_credit, LogReader.top_userID_credit,
                 TransMatrx.botMatrix_car, TransMatrx.topMatrix_car, TransMatrx.botMatrix_credit, TransMatrx.topMatrix_credit,
                 ThinkTime.bot_thinking_car, ThinkTime.top_thinking_car, ThinkTime.bot_thinking_credit, ThinkTime.top_thinking_credit,
                 ThinkTime.bot_carData, ThinkTime.top_carData, ThinkTime.bot_creditData, ThinkTime.top_creditData,
                 map.timeSpent_car, map.timeSpent_credit,
                 TransMatrx.botMatrix_o_car, TransMatrx.topMatrix_o_car, TransMatrx.botMatrix_c_car, TransMatrx.topMatrix_c_car, TransMatrx.botMatrix_e_car,
                 TransMatrx.topMatrix_e_car, TransMatrx.botMatrix_a_car, TransMatrx.topMatrix_a_car, TransMatrx.botMatrix_n_car, TransMatrx.topMatrix_n_car,
                 TransMatrx.botMatrix_o_credit, TransMatrx.topMatrix_o_credit, TransMatrx.botMatrix_c_credit, TransMatrx.topMatrix_c_credit, TransMatrx.botMatrix_e_credit,
                 TransMatrx.topMatrix_e_credit, TransMatrx.botMatrix_a_credit, TransMatrx.topMatrix_a_credit, TransMatrx.botMatrix_n_credit, TransMatrx.topMatrix_n_credit,
                 ThinkTime.highThink_carO, ThinkTime.lowThink_carO, ThinkTime.highThink_carC, ThinkTime.lowThink_carC, ThinkTime.highThink_carE, ThinkTime.lowThink_carE,
                 ThinkTime.highThink_carA, ThinkTime.lowThink_carA, ThinkTime.highThink_carN, ThinkTime.lowThink_carN,
                 ThinkTime.highThink_creditO, ThinkTime.lowThink_creditO, ThinkTime.highThink_creditC, ThinkTime.lowThink_creditC, ThinkTime.highThink_creditE,
                 ThinkTime.lowThink_creditE, ThinkTime.highThink_creditA, ThinkTime.lowThink_creditA, ThinkTime.highThink_creditN, ThinkTime.lowThink_creditN,
                 ThinkTime.df_car, ThinkTime.df_credit,
                 PresvyReader.o_high_IDs, PresvyReader.o_low_IDs, PresvyReader.c_high_IDs, PresvyReader.c_low_IDs, PresvyReader.e_high_IDs,
                 PresvyReader.e_low_IDs, PresvyReader.a_high_IDs, PresvyReader.a_low_IDs, PresvyReader.n_high_IDs, PresvyReader.n_low_IDs,
                 PresvyReader.visScore,
                 PresvyReader.sum_dic,
                 LogReader.familiarity_car, LogReader.familiarity_credit, LogReader.domainFami_agg,
                 LogReader.diffScore_agg)



