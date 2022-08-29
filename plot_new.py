import numpy as np
import math
import random
import matplotlib.pyplot as plt
import torch
from scipy.special import comb, gamma
from multiprocessing import Process, Queue
from time import sleep
import datetime
import os


def fix_theta_change_delta_risk_averse(experiment_name, results, path):
    xAxis = []
    yAxis = []
    v2PnL = []
    for job in results:
        xAxis.append(job['delta'])
        yAxis.append(job['best_expected_pnl_projected'])
        v2PnL.append(job['v2_expected_pnl'])

    plt.title('Delta vs PnL')
    plt.xlabel('Delta')
    plt.ylabel('Expected PnL')
    plt.plot(xAxis, yAxis, '-o', label = "V3")
    plt.plot(xAxis, v2PnL, label = "V2", color='grey', linestyle='dotted')
    plt.legend()
    plt.savefig('{}/{}_delta_pnl.pdf'.format(path, experiment_name))
    plt.close()
    
    xAxis = []
    yAxis = []
    v2PnL = []
    for job in results:
        xAxis.append(job['delta'])
        yAxis.append(job['best_utility_projected'])
        v2PnL.append(job['v2_expected_utility'].item())

    plt.title('Delta vs Utility')
    plt.xlabel('Delta')
    plt.ylabel('Expected Utility')
    plt.plot(xAxis, yAxis, '-o', label = "V3")
    plt.plot(xAxis, v2PnL, label = "V2", color='grey', linestyle='dotted')
    plt.legend()
    plt.savefig('{}/{}_delta_utility.pdf'.format(path, experiment_name))
    plt.close()
    
    xAxis = []
    yAxis = []

    for job in results:
        xAxis.append(job['delta'])
        yAxis.append(job['total_crossing'])

    plt.title('Delta vs Total Crossing')
    plt.xlabel('Delta')
    plt.ylabel('Total Expected Crossings')
    plt.plot(xAxis, yAxis, '-o')
    plt.savefig('{}/{}_delta_crossing.pdf'.format(path, experiment_name))
    plt.close()

    xAxis = []
    y1Axis = []
    y2Axis = []
    v2PnL = []
    for job in results:
        xAxis.append(job['delta'])
        y1Axis.append(job['best_utility_projected'])
        y2Axis.append(job['total_crossing'])
        v2PnL.append(job['v2_expected_utility'].item())

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.set_title('Delta vs Expected Utility and Total Crossing')
    ax1.set_xlabel('Delta')
    ax1.set_ylabel('Expected Utility', color='darkblue')
    ax2.set_ylabel('Expected Total Crossings', color='firebrick')
    ax2.plot(xAxis, y2Axis, '-o', color='firebrick', label='V3 Total Crossing')
    ax1.plot(xAxis, y1Axis, '-o', color='darkblue', label = "V3 Utility")
    ax1.plot(xAxis, v2PnL, label = "V2 Utility", color='b', linestyle='dotted')
    ax1.legend(loc='upper center')
    ax2.legend(loc='upper right')
    plt.savefig('{}/{}_delta_utility_and_crossing.pdf'.format(path, experiment_name))
    plt.close()


def change_k(experiment_name, results, path):
    xAxis = []
    yAxis = []
    for job in results:
        xAxis.append(job['num_non_arb'])
        yAxis.append(job['best_expected_pnl_projected'])

    plt.title('K vs PnL')
    plt.xlabel('K')
    plt.ylabel('Expected PnL')
    plt.plot(xAxis, yAxis, '-o')
    plt.savefig('{}/{}_k_pnl.pdf'.format(path, experiment_name))
    plt.close()
    
    xAxis = []
    yAxis = []
    for job in results:
        xAxis.append(job['num_non_arb'])
        yAxis.append(job['best_utility_projected'])

    plt.title('K vs Utility')
    plt.xlabel('K')
    plt.ylabel('Expected Utility')
    plt.plot(xAxis, yAxis, '-o')
    plt.savefig('{}/{}_k_utility.pdf'.format(path, experiment_name))
    plt.close()
    
    xAxis = []
    yAxis = []
    
    for job in results:
        xAxis.append(job['num_non_arb'])
        yAxis.append(job['total_crossing'])

    plt.title('K vs Total Crossing')
    plt.xlabel('K')
    plt.ylabel('Total Expected Crossings')
    plt.plot(xAxis, yAxis, '-o')
    plt.savefig('{}/{}_k_crossing.pdf'.format(path, experiment_name))
    plt.close()

    xAxis = []
    y1Axis = []
    y2Axis = []
    for job in results:
        xAxis.append(job['num_non_arb'])
        y1Axis.append(job['best_utility_projected'])
        y2Axis.append(job['total_crossing'])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.set_title('K vs Expected Utility and Total Crossing')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Expected Utility', color='darkblue')
    ax2.set_ylabel('Expected Total Crossings', color='firebrick')
    ax1.plot(xAxis, y1Axis, '-o', color='darkblue', label = "Utility")
    ax2.plot(xAxis, y2Axis, '-o', color='firebrick', label='Total Crossing')
    ax1.legend(loc='upper center')
    ax2.legend(loc='upper right')
    plt.savefig('{}/{}_k_utility_and_crossing.pdf'.format(path, experiment_name))
    plt.close()


def change_lambda(experiment_name, results, path):
    xAxis = []
    yAxis = []
    for job in results:
        xAxis.append(job['non_arb_lambda'])
        yAxis.append(job['best_expected_pnl_projected'])

    plt.title('lambda vs PnL')
    plt.xlabel('lambda')
    plt.ylabel('Expected PnL')
    plt.plot(xAxis, yAxis, '-o')
    plt.savefig('{}/{}_lambda_pnl.pdf'.format(path, experiment_name))
    plt.close()
    
    xAxis = []
    yAxis = []
    for job in results:
        xAxis.append(job['non_arb_lambda'])
        yAxis.append(job['best_utility_projected'])

    plt.title('lambda vs Utility')
    plt.xlabel('lambda')
    plt.ylabel('Expected Utility')
    plt.plot(xAxis, yAxis, '-o')
    plt.savefig('{}/{}_lambda_utility.pdf'.format(path, experiment_name))
    plt.close()
    
    xAxis = []
    yAxis = []
    
    for job in results:
        xAxis.append(job['non_arb_lambda'])
        yAxis.append(job['total_crossing'])

    plt.title('lambda vs Total Crossing')
    plt.xlabel('lambda')
    plt.ylabel('Total Expected Crossings')
    plt.plot(xAxis, yAxis, '-o')
    plt.savefig('{}/{}_lambda_crossing.pdf'.format(path, experiment_name))
    plt.close()

    xAxis = []
    y1Axis = []
    y2Axis = []
    for job in results:
        xAxis.append(job['non_arb_lambda'])
        y1Axis.append(job['best_utility_projected'])
        y2Axis.append(job['total_crossing'])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.set_title('lambda vs Expected Utility and Total Crossing')
    ax1.set_xlabel('lambda')
    ax1.tick_params(axis='x', labelsize=8)
    ax1.set_ylabel('Expected Utility', color='darkblue')
    ax2.set_ylabel('Expected Total Crossings', color='firebrick')
    ax1.plot(xAxis, y1Axis, '-o', color='darkblue', label = "Utility")
    ax2.plot(xAxis, y2Axis, '-o', color='firebrick', label='Total Crossing')
    ax1.legend(loc='upper center')
    ax2.legend(loc='upper right')
    plt.savefig('{}/{}_lambda_utility_and_crossing.pdf'.format(path, experiment_name))
    plt.close()


def change_both_theta_and_delta_risk_averse(experiment_name, results, path):
    '''
    # get the list of theta values
    theta_list = []
    for job in results:
        theta = job['theta']
        if theta not in theta_list:
            theta_list.append(theta)
    '''

    thetas = [1.002 + i * 0.002 for i in range(10)]
    xAxis = [[] for i in range(10)]
    yAxis = [[] for i in range(10)]
    delta = 1
    thetaIndex = 0
    for job in results:
        xAxis[thetaIndex].append(job['total_crossing'])
        yAxis[thetaIndex].append(job['best_utility_projected'])

        if delta % 20 == 0:
            delta = 1
            thetaIndex += 1
        else:
            delta += 1

    for i in range(len(xAxis)):
        plt.scatter(xAxis[i], yAxis[i], label = "Theta = {}".format(str(thetas[i])))

    # for job in results:
    #     plt.annotate(job['delta'], (job['total_crossing'], job['best_utility_projected']), fontsize=5)

    plt.title('Total Crossing vs Utility.')
    plt.xlabel('Total Crossing')
    plt.ylabel('Expected Utiltiy')
    #plt.xlim(0,40)
    plt.legend()
    plt.savefig('{}/{}_crossing_utility.pdf'.format(path, experiment_name))
    plt.close()
    
    thetas = [1.002 + i * 0.002 for i in range(10)]
    xAxis = [[] for i in range(10)]
    yAxis = [[] for i in range(10)]
    delta = 1
    thetaIndex = 0
    for job in results:
        xAxis[thetaIndex].append(job['total_crossing'])
        yAxis[thetaIndex].append(job['best_utility_projected'])

        if delta % 20 == 0:
            delta = 1
            thetaIndex += 1
        else:
            delta += 1

    fronteirX = []
    fronteirY = []

    fronteirX = [[] for i in range(10)]
    fronteirY = [[] for i in range(10)]

    nonFronteirX = []
    nonFronteirY = []
    points = []
    for p in range(10):
        for i in range(len(xAxis[p])):
            testingPoint = (xAxis[p][i], yAxis[p][i])
            points.append((testingPoint, p))

    #print(points)
    for testP, thetaIndex in points:
    #     fronteirX.append(testP[0])
    #     fronteirY.append(testP[1])
        broken = False
        for otherP, _ in points:
            if testP == otherP:
                continue
            elif otherP[0] <= testP[0] and otherP[1] > testP[1]:
                broken = True
                break
            else:
                continue
        if broken == False:
            fronteirX[thetaIndex].append(testP[0])
            fronteirY[thetaIndex].append(testP[1])
        else:
            nonFronteirX.append(testP[0])
            nonFronteirY.append(testP[1])
            
    plt.scatter(nonFronteirX, nonFronteirY, label= "Non Frontier", color = 'lightgray') 
    for i in range(10):
        plt.scatter(fronteirX[i], fronteirY[i], label = "Theta = {}".format(str(thetas[i])))          
    plt.title('Pareto frontier of gas fees (total crossing) and maximum Utility')
    plt.xlabel('Total Crossing')
    plt.ylabel('Expected Utility')
    # plt.xlim(0,40)
    plt.legend()
    plt.savefig('{}/{}_crossing_utility_frontier.pdf'.format(path, experiment_name))
    plt.close()


def change_theta_delta_and_fee_rate(experiment_name, results, path):
    fee_rates = [0.0005, 0.003, 0.005, 0.01, 0.02]
    xAxis = [[] for i in range(5)]
    yAxis = [[] for i in range(5)]
    inner_idx = 1
    outer_idx = 0
    for job in results:
        xAxis[outer_idx].append(job['total_crossing'])
        yAxis[outer_idx].append(job['best_utility_projected'])

        if inner_idx % 25 == 0:
            inner_idx = 1
            outer_idx += 1
        else:
            inner_idx += 1

    for i in range(len(xAxis)):
        plt.scatter(xAxis[i], yAxis[i], label = "fee_rate = {}".format(str(fee_rates[i])))

    # for job in results:
    #     plt.annotate(job['delta'], (job['total_crossing'], job['best_utility_projected']), fontsize=5)

    plt.title('Total Crossing vs Utility.')
    plt.xlabel('Total Crossing')
    plt.ylabel('Expected Utiltiy')
    #plt.xlim(0,40)
    plt.legend()
    plt.savefig('{}/{}_crossing_utility_fee_rate.pdf'.format(path, experiment_name))
    plt.close()
    
    fee_rates = [0.0005, 0.003, 0.005, 0.01, 0.02]
    xAxis = [[] for i in range(5)]
    yAxis = [[] for i in range(5)]
    inner_idx = 1
    outer_idx = 0
    for job in results:
        xAxis[outer_idx].append(job['total_crossing'])
        yAxis[outer_idx].append(job['best_utility_projected'])

        if inner_idx % 25 == 0:
            inner_idx = 1
            outer_idx += 1
        else:
            inner_idx += 1

    fronteirX = []
    fronteirY = []

    fronteirX = [[] for i in range(5)]
    fronteirY = [[] for i in range(5)]

    nonFronteirX = []
    nonFronteirY = []
    points = []
    for p in range(5):
        for i in range(len(xAxis[p])):
            testingPoint = (xAxis[p][i], yAxis[p][i])
            points.append((testingPoint, p))

    #print(points)
    for testP, thetaIndex in points:
    #     fronteirX.append(testP[0])
    #     fronteirY.append(testP[1])
        broken = False
        for otherP, _ in points:
            if testP == otherP:
                continue
            elif otherP[0] <= testP[0] and otherP[1] > testP[1]:
                broken = True
                break
            else:
                continue
        if broken == False:
            fronteirX[thetaIndex].append(testP[0])
            fronteirY[thetaIndex].append(testP[1])
        else:
            nonFronteirX.append(testP[0])
            nonFronteirY.append(testP[1])

    plt.scatter(nonFronteirX, nonFronteirY, label= "Non Frontier", color = 'lightgray') 
    for i in range(5):
        plt.scatter(fronteirX[i], fronteirY[i], label = "fee_rate = {}".format(str(fee_rates[i])))          
    plt.title('Pareto frontier of gas fees (total crossing) and maximum Utility')
    plt.xlabel('Total Crossing')
    plt.ylabel('Expected Utility')
    # plt.xlim(0,40)
    plt.legend()
    plt.savefig('{}/{}_crossing_utility_frontier_fee_rate.pdf'.format(path, experiment_name))
    plt.close()


def fix_theta_and_delta_change_risk_averse(experiment_name, results, path):
    xAxis = []
    yAxis = []

    for job in results:
        xAxis.append(job['risk_a'])
        yAxis.append(job['best_utility_projected'])

    plt.title('Risk Parameter vs Utility.')
    plt.xlabel('Risk Averseness')
    plt.ylabel('Best Utility')
    plt.scatter(xAxis, yAxis)
    plt.savefig('{}/{}_utility_a.pdf'.format(path, experiment_name))
    plt.close()
    
    xAxis = []
    yAxis = []

    for job in results:
        xAxis.append(job['risk_a'])
        yAxis.append(job['best_expected_pnl_projected'])

    plt.title('Risk Parameter vs Expected PnL.')
    plt.xlabel('Risk Averseness')
    plt.ylabel('Expected PnL')
    plt.plot(xAxis, yAxis, '-o')
    plt.savefig('{}/{}_pnl_a.pdf'.format(path, experiment_name))
    plt.close()
    
    xAxis = []
    yAxis = []

    for job in results:
        xAxis.append(job['risk_a'])
        yAxis.append(job['best_pnl_std_projected'])

    plt.title('Risk Parameter vs Standard Deviation of PnL.')
    plt.xlabel('Risk Averseness')
    plt.ylabel('Std of PnL')
    plt.plot(xAxis, yAxis, '-o')
    plt.savefig('{}/{}_std_pnl_a.pdf'.format(path, experiment_name))
    plt.close()

    xAxis = []
    yAxis = []

    for job in results:
        xAxis.append(job['best_expected_pnl_projected'])
        yAxis.append(job['best_pnl_std_projected'])

    plt.title('Expected PnL vs Standard Deviation of PnL.')
    plt.xlabel('Expected PnL')
    plt.ylabel('Std of PnL')
    plt.plot(xAxis, yAxis, '-o')
    plt.savefig('{}/{}_pnl_std_pnl.pdf'.format(path, experiment_name))
    plt.close()

    xAxis = []
    y1Axis = []
    y2Axis = []
    for job in results:
        xAxis.append(job['risk_a'])
        y1Axis.append(job['best_expected_pnl_projected'])
        y2Axis.append(job['best_pnl_std_projected'])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.set_title('Risk Parameter vs Expected PnL and Std of PnL')
    ax1.set_xlabel('Risk Averseness')
    ax1.set_ylabel('Expected PnL', color='darkblue')
    ax2.set_ylabel('Std of PnL', color='firebrick')
    ax1.plot(xAxis, y1Axis, '-o', color='darkblue', label = "Expected PnL")
    ax2.plot(xAxis, y2Axis, '-o', color='firebrick', label='Std of PnL')
    ax1.legend(loc='upper center')
    ax2.legend(loc='upper right')
    plt.savefig('{}/{}_a_pnl_and_std_pnl.pdf'.format(path, experiment_name))
    plt.close()
    
    # allocation
    fig, axes = plt.subplots(3, 4, figsize=(15,15))
    ax_list = []
    for i in range(3):
        for j in range(4):
            ax_list.append(axes[i][j])
    for i, job in enumerate(results):
        ax_list[i].set_ylim([0.0, 1.0])
        ax_list[i].fill_between(job['bucket_endpoints'][1+30:-2-30], job['best_x_projected'][30:-30], step='pre')
        ax_list[i].set_title('Risk aversion = {:.1f}'.format(job['risk_a']))
        ax_list[i].set_xlabel('Bucket')
        ax_list[i].set_ylabel('Capital Allocation')
    plt.savefig('{}/{}_allocation_portion_a.pdf'.format(path, experiment_name))
    plt.close()
    
    fig, axes = plt.subplots(3, 4, figsize=(15,15))
    ax_list = []
    for i in range(3):
        for j in range(4):
            ax_list.append(axes[i][j])
    for i, job in enumerate(results):
        ax_list[i].set_ylim([0.0, 1000.0])
        ax_list[i].fill_between(job['bucket_endpoints'][1+30:-2-30], job['best_units_of_liq_projected'][30:-30], step='pre')
        ax_list[i].set_title('Risk aversion = {:.1f}'.format(job['risk_a']))
        ax_list[i].set_xlabel('Bucket')
        ax_list[i].set_ylabel('Units of Liquidity')
    plt.savefig('{}/{}_allocation_units_of_liq_a.pdf'.format(path, experiment_name))
    plt.close()



def handle_file(fname):
    batch = np.load(fname, allow_pickle=True)
    batchDict = batch.item()
    path = 'plots/{}'.format(fname)
    try:
        os.mkdir(path)
    except OSError as error:
        pass
    for experiment_name in batchDict:
        results = batchDict[experiment_name]
        
        print('experiment_name:', experiment_name)
        
        if 'fix_theta_change_delta' in experiment_name:
            fix_theta_change_delta_risk_averse(experiment_name, results, path)
        elif 'change_k' in experiment_name:
            change_k(experiment_name, results, path)
        elif 'change_lambda' in experiment_name:
            change_lambda(experiment_name, results, path)
        elif 'change_both_theta_and_delta_risk_averse' in experiment_name:
            change_both_theta_and_delta_risk_averse(experiment_name, results, path)
        elif 'change_theta_delta_and_fee_rate' in experiment_name:
            change_theta_delta_and_fee_rate(experiment_name, results, path)
        elif 'fix_theta_and_delta_change_risk_averse' in experiment_name:
            fix_theta_and_delta_change_risk_averse(experiment_name, results, path)
        else:
            raise Exception('unknown experiment type: {}'.format(experiment_name))


if __name__ == '__main__':
    # plt.rcParams['text.usetex'] = True
    handle_file('result_batch_x.npy')


