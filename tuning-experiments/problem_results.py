
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from scipy.stats import kendalltau

import statistics as stats


class ProblemResults:


    def __init__(self):
        self.results = {}

    @dataclass
    class Result:
        problem:str
        rmse:float
        kt:float
        diff:float
        top1err:float
        top2err:float
        top3err:float

        correct1:int
        correct2:int
        correct3:int

        spgemm_runtime:float
        timings:dict

        bcast_time: float
        local_spgemm_time: float
        merge_time: float


    def add_result(self, problem, y_arr, y_pred_arr, spgemm_runtime, timings,
                   bcast_pred_arr, local_spegmm_pred_arr, merge_pred_arr,
                   params):

        y_arr, y_pred_arr = np.array(y_arr), np.array(y_pred_arr)

        kt = (kendalltau(y_pred_arr, y_arr).statistic)

        rmse = ((((np.linalg.norm(y_pred_arr - y_arr))**(1/2))/len(y_arr))**2)

        is_correct1 = 1 if np.argmin(y_pred_arr)==np.argmin(y_arr) else 0
        is_correct2 = 1 if is_correct1 or np.argpartition(y_pred_arr,1)[1]==np.argmin(y_arr) else 0
        is_correct3 = 1 if is_correct2 or np.argpartition(y_pred_arr,2)[2]==np.argmin(y_arr) else 0

        diff = abs(y_arr[np.argmin(y_pred_arr)] - np.min(y_arr))

        if np.min(y_arr)>0:
            top_1_err = (y_arr[np.argmin(y_pred_arr)] / (np.min(y_arr))) - 1
            top_2_err = min(top_1_err, (y_arr[np.argpartition(y_pred_arr, 1)[1]] / (np.min(y_arr))) - 1) 
            top_3_err = min(top_2_err, (y_arr[np.argpartition(y_pred_arr, 2)[2]] / (np.min(y_arr))) - 1) 
        else:
            top_1_err = None
            top_2_err = None
            top_3_err = None
        
        min_idx = np.argmin(y_pred_arr)
        
        timings["AutotuningSpGEMM"] = y_arr[min_idx]
        bcast, local_spgemm, merge = bcast_pred_arr[params[min_idx]], local_spegmm_pred_arr[params[min_idx]], merge_pred_arr[params[min_idx]]
        
        print(params[min_idx])
        print((bcast + local_spgemm + merge), timings["PredSpGEMMTime"])
        
        

        self.results[problem] = self.Result(problem, rmse, kt, diff, top_1_err, top_2_err, top_3_err, 
                                            is_correct1, is_correct2, is_correct3, spgemm_runtime, timings,
                                            bcast, local_spgemm, merge)
    
    def get_stat_arr(self, stat_name):
        arr = map(lambda r: self.results[r].__dict__[stat_name], self.results)
        return list(filter(lambda x: x!=None, arr)) 

    def get_result_stat(self, problem, stat):
        return self.results[problem].__dict__[stat]

    def output_eval(self):

        kt_arr = self.get_stat_arr("kt")
        rmse_arr = self.get_stat_arr("rmse")
        diff_arr = self.get_stat_arr("diff")
        correct_arr1 = self.get_stat_arr("correct1")
        correct_arr2 = self.get_stat_arr("correct2")
        correct_arr3 = self.get_stat_arr("correct3")
        err_arr1 = self.get_stat_arr("top1err")
        err_arr2 = self.get_stat_arr("top2err")
        err_arr3 = self.get_stat_arr("top3err")
        
        print(f"----AVERAGE KT: {sum(kt_arr)/len(kt_arr)}")
        print(f"----MEDIAN KT: {stats.median(kt_arr)}")
        
        kt_sorted_results = sorted(self.results.values(), key=lambda r:r.kt)
        print(f"----Problems with the 10 worst KT are: ")
        for i in range(0, min(10, len(kt_sorted_results))):
            print(f"{kt_sorted_results[i].problem}")

        print(f"----Problems with the 10 best KT are: ")
        for i in range(1, min(11, len(kt_sorted_results)+1)):
            print(f"{kt_sorted_results[-i].problem}")

        print(f"----AVERAGE DIFF : {sum(diff_arr)/len(diff_arr)}s")
        print(f"----TOTAL DIFF : {sum(diff_arr)}s")
        print(f"----NUMBER CORRECT1 : {sum(correct_arr1)}/{len(correct_arr1)}")
        print(f"----NUMBER CORRECT2 : {sum(correct_arr2)}/{len(correct_arr2)}")
        print(f"----NUMBER CORRECT3 : {sum(correct_arr3)}/{len(correct_arr3)}")
        print(f"----AVERAGE TOP 1 ERROR: {sum(err_arr1)/len(err_arr1)}")
        print(f"----MEDIAN TOP 1 ERROR: {stats.median(err_arr1)}")
        print(f"----AVERAGE TOP 2 ERROR: {sum(err_arr2)/len(err_arr2)}")
        print(f"----MEDIAN TOP 3 ERROR: {stats.median(err_arr2)}")
        print(f"----AVERAGE TOP 3 ERROR: {sum(err_arr3)/len(err_arr3)}")
        print(f"----MEDIAN TOP 3 ERROR: {stats.median(err_arr3)}")

    def plot_eval(self):
        
        problems = []
        err_arr1 = []
        err_arr2 = []
        err_arr3 = []
        for problem in self.results.keys():
            err1 = self.get_result_stat(problem, "top1err")
            err2 = self.get_result_stat(problem, "top2err")
            err3 = self.get_result_stat(problem, "top3err")
            
            if err1 or err2 or err3:
                problems.append(problem.split(".")[0] + ("-permuted" if "permuted" in problem else ""))
                err_arr1.append(err1)
                err_arr2.append(err2)
                err_arr3.append(err3)

        width=2.0
        inds = np.arange(len(problems))*(width*4)
        plt.figure(figsize=(12,6))
        plt.bar(inds, err_arr1, label="Top 1 Error", width=width)
        plt.bar(inds+width, err_arr2, label="Top 2 Error", width=width)
        plt.bar(inds+width*2, err_arr3, label="Top 3 Error", width=width)
        plt.legend()
        plt.xticks(inds+width, labels=problems, rotation=90)
        plt.ylabel("Error")
        plt.title("Top K Errors of Test Matrices")
        plt.savefig(f"{args.label}-plots/{args.plotname}-errs.png", bbox_inches='tight')
        plt.clf()

    def plot_spgemm(self):

        problems = []
        spgemm_times = []
        autotuning_timings = []
        autotuning_spgemm_timings = []
        for problem in self.results.keys():
            spgemm_time = self.get_result_stat(problem, "spgemm_runtime")
            if spgemm_time:
                problems.append(problem.split(".")[0] + ("-permuted" if "permuted" in problem else ""))
                spgemm_times.append(spgemm_time)
                autotuning_timings.append(self.get_result_stat(problem, "timings"))
        
        feature_init_times = list(map(lambda t: t["FeatureInit"], autotuning_timings))
        prediction_times = list(map(lambda t: t["Prediction"], autotuning_timings))
        autotuning_spgemm_times = list(map(lambda t: t["AutotuningSpGEMM"], autotuning_timings))
        tuning_times = list(map(lambda t: t["TuneSpGEMM2D"], autotuning_timings))

        categories = ["Autotuning Runtime", "SpGEMM Runtime"]
        ind = np.arange(len(problems))*1.5
        plt.figure(figsize=(12,6))
        fig, ax = plt.subplots()
        #ax.bar(ind, feature_init_times, width=0.5, label="FeatureInit")
        #ax.bar(ind, prediction_times, width=0.5, label="Prediction", bottom=feature_init_times)
        ax.bar(ind, tuning_times, width=0.5, label="Autotuning Overhead")
        ax.bar(ind, autotuning_spgemm_times, width=0.5, label="Autotuning SpGEMM", bottom=tuning_times)
        ax.bar(ind+0.5, spgemm_times, width=0.5, label="Naive SpGEMM")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"SpGEMM Runtime vs. Autotuning Overhead")
        ax.set_xticks(ind)
        ax.set_xticklabels(problems, rotation=90)
        ax.legend()
        plt.savefig(f"{args.label}-plots/timing.png", bbox_inches='tight')
        plt.clf()

class ProblemPhaseResults: # Don't ask


    def __init__(self):
        self.results = {}

    @dataclass
    class Result:
        problem:str
        rmse:float
        kt:float
        diff:float
        top1err:float
        top2err:float
        top3err:float

        correct1:int
        correct2:int
        correct3:int

        spgemm_runtime:float
        timings:dict

    def add_result(self, problem, y_arr, y_pred_arr, spgemm_runtime, timings):

        y_arr, y_pred_arr = np.array(y_arr), np.array(y_pred_arr)

        kt = (kendalltau(y_pred_arr, y_arr).statistic)

        rmse = ((((np.linalg.norm(y_pred_arr - y_arr))**(1/2))/len(y_arr))**2)

        is_correct1 = 1 if np.argmin(y_pred_arr)==np.argmin(y_arr) else 0
        is_correct2 = 1 if is_correct1 or np.argpartition(y_pred_arr,1)[1]==np.argmin(y_arr) else 0
        is_correct3 = 1 if is_correct2 or np.argpartition(y_pred_arr,2)[2]==np.argmin(y_arr) else 0

        diff = abs(y_arr[np.argmin(y_pred_arr)] - np.min(y_arr))

        if np.min(y_arr)>0:
            top_1_err = (y_arr[np.argmin(y_pred_arr)] / (np.min(y_arr))) - 1
            top_2_err = min(top_1_err, (y_arr[np.argpartition(y_pred_arr, 1)[1]] / (np.min(y_arr))) - 1) 
            top_3_err = min(top_2_err, (y_arr[np.argpartition(y_pred_arr, 2)[2]] / (np.min(y_arr))) - 1) 
        else:
            top_1_err = None
            top_2_err = None
            top_3_err = None

        timings["AutotuningSpGEMM"] = y_arr[np.argmin(y_pred_arr)]
        self.results[problem] = self.Result(problem, rmse, kt, diff, top_1_err, top_2_err, top_3_err, 
                                            is_correct1, is_correct2, is_correct3, spgemm_runtime, timings)
    
    def get_stat_arr(self, stat_name):
        arr = map(lambda r: self.results[r].__dict__[stat_name], self.results)
        return list(filter(lambda x: x!=None, arr)) 

    def get_result_stat(self, problem, stat):
        return self.results[problem].__dict__[stat]

    def output_eval(self):

        kt_arr = self.get_stat_arr("kt")
        rmse_arr = self.get_stat_arr("rmse")
        diff_arr = self.get_stat_arr("diff")
        correct_arr1 = self.get_stat_arr("correct1")
        correct_arr2 = self.get_stat_arr("correct2")
        correct_arr3 = self.get_stat_arr("correct3")
        err_arr1 = self.get_stat_arr("top1err")
        err_arr2 = self.get_stat_arr("top2err")
        err_arr3 = self.get_stat_arr("top3err")
        
        print(f"----AVERAGE KT: {sum(kt_arr)/len(kt_arr)}")
        print(f"----MEDIAN KT: {stats.median(kt_arr)}")
        
        kt_sorted_results = sorted(self.results.values(), key=lambda r:r.kt)
        print(f"----Problems with the 10 worst KT are: ")
        for i in range(0, min(10, len(kt_sorted_results))):
            print(f"{kt_sorted_results[i].problem}")

        print(f"----Problems with the 10 best KT are: ")
        for i in range(1, min(11, len(kt_sorted_results)+1)):
            print(f"{kt_sorted_results[-i].problem}")

        print(f"----AVERAGE DIFF : {sum(diff_arr)/len(diff_arr)}s")
        print(f"----TOTAL DIFF : {sum(diff_arr)}s")
        print(f"----NUMBER CORRECT1 : {sum(correct_arr1)}/{len(correct_arr1)}")
        print(f"----NUMBER CORRECT2 : {sum(correct_arr2)}/{len(correct_arr2)}")
        print(f"----NUMBER CORRECT3 : {sum(correct_arr3)}/{len(correct_arr3)}")
        print(f"----AVERAGE TOP 1 ERROR: {sum(err_arr1)/len(err_arr1)}")
        print(f"----MEDIAN TOP 1 ERROR: {stats.median(err_arr1)}")
        print(f"----AVERAGE TOP 2 ERROR: {sum(err_arr2)/len(err_arr2)}")
        print(f"----MEDIAN TOP 3 ERROR: {stats.median(err_arr2)}")
        print(f"----AVERAGE TOP 3 ERROR: {sum(err_arr3)/len(err_arr3)}")
        print(f"----MEDIAN TOP 3 ERROR: {stats.median(err_arr3)}")

    def plot_eval(self):
        
        problems = []
        err_arr1 = []
        err_arr2 = []
        err_arr3 = []
        for problem in self.results.keys():
            err1 = self.get_result_stat(problem, "top1err")
            err2 = self.get_result_stat(problem, "top2err")
            err3 = self.get_result_stat(problem, "top3err")
            
            if err1 or err2 or err3:
                problems.append(problem.split(".")[0] + ("-permuted" if "permuted" in problem else ""))
                err_arr1.append(err1)
                err_arr2.append(err2)
                err_arr3.append(err3)

        width=2.0
        inds = np.arange(len(problems))*(width*4)
        plt.figure(figsize=(12,6))
        plt.bar(inds, err_arr1, label="Top 1 Error", width=width)
        plt.bar(inds+width, err_arr2, label="Top 2 Error", width=width)
        plt.bar(inds+width*2, err_arr3, label="Top 3 Error", width=width)
        plt.legend()
        plt.xticks(inds+width, labels=problems, rotation=90)
        plt.ylabel("Error")
        plt.title("Top K Errors of Test Matrices")
        plt.savefig(f"{args.label}-plots/{args.plotname}-errs.png", bbox_inches='tight')
        plt.clf()

    def plot_spgemm(self):

        problems = []
        spgemm_times = []
        autotuning_timings = []
        autotuning_spgemm_timings = []
        for problem in self.results.keys():
            spgemm_time = self.get_result_stat(problem, "spgemm_runtime")
            if spgemm_time:
                problems.append(problem.split(".")[0] + ("-permuted" if "permuted" in problem else ""))
                spgemm_times.append(spgemm_time)
                autotuning_timings.append(self.get_result_stat(problem, "timings"))
        
        feature_init_times = list(map(lambda t: t["FeatureInit"], autotuning_timings))
        prediction_times = list(map(lambda t: t["Prediction"], autotuning_timings))
        autotuning_spgemm_times = list(map(lambda t: t["AutotuningSpGEMM"], autotuning_timings))
        tuning_times = list(map(lambda t: t["TuneSpGEMM2D"], autotuning_timings))

        categories = ["Autotuning Runtime", "SpGEMM Runtime"]
        ind = np.arange(len(problems))*1.5
        plt.figure(figsize=(12,6))
        fig, ax = plt.subplots()
        #ax.bar(ind, feature_init_times, width=0.5, label="FeatureInit")
        #ax.bar(ind, prediction_times, width=0.5, label="Prediction", bottom=feature_init_times)
        ax.bar(ind, tuning_times, width=0.5, label="Autotuning Overhead")
        ax.bar(ind, autotuning_spgemm_times, width=0.5, label="Autotuning SpGEMM", bottom=tuning_times)
        ax.bar(ind+0.5, spgemm_times, width=0.5, label="Naive SpGEMM")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"SpGEMM Runtime vs. Autotuning Overhead")
        ax.set_xticks(ind)
        ax.set_xticklabels(problems, rotation=90)
        ax.legend()
        plt.savefig(f"{args.label}-plots/timing.png", bbox_inches='tight')
        plt.clf()
