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

def fix_theta_change_delta_risk_neutral(experiment_name, results, path):
    xAxis = []
    yAxis = []
    v2PnL = []
    for job in results:
        xAxis.append(job['delta'])
        yAxis.append(job['best_normalized_pnl'])
        v2PnL.append(job['v2_normalized_pnl'])
    
    # plt.title(r'OPT(\cP,\vect{{\mu}}({}, \Delta)), W = {}'.format(job['theta'], job['W']))
    plt.title('Delta vs Expected PnL')
    plt.xlabel('Delta')
    plt.ylabel('Expected PnL')
    plt.plot(xAxis, yAxis, '-o', label = "V3")
    plt.plot(xAxis, v2PnL, label = "V2", color='grey', linestyle='dotted')
    plt.legend()
    plt.savefig('{}/{}_delta_pnl.pdf'.format(path, experiment_name))
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


def fix_theta_change_delta_risk_averse(experiment_name, results, path):
    xAxis = []
    yAxis = []
    v2PnL = []
    for job in results:
        xAxis.append(job['delta'])
        yAxis.append(job['best_expected_pnl_projected'])
        v2PnL.append(job['v2_normalized_pnl'])

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
        v2PnL.append(job['v2_normalized_pnl'])

    plt.title('Delta vs PnL')
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


def change_both_theta_and_delta_risk_neutral(experiment_name, results, path):
    thetas = [round(1.015 + .02*i, 3) for i in range(10)]
    xAxis = [[] for i in range(10)]
    yAxis = [[] for i in range(10)]
    delta = 1
    thetaIndex = 0
    for job in results:
        xAxis[thetaIndex].append(job['total_crossing'])
        yAxis[thetaIndex].append(job['best_normalized_pnl'])

        if delta % 10 == 0:
            delta = 1
            thetaIndex += 1
        else:
            delta += 1

    for i in range(len(xAxis)):
        plt.scatter(xAxis[i], yAxis[i], label = "Theta = {}".format(str(thetas[i])))

    plt.title('Total Crossing vs PNL.')
    plt.xlabel('Total Crossing')
    plt.ylabel('Expected PNL')
    #plt.xlim(0,40)
    plt.legend()
    plt.savefig('{}/{}_crossing_pnl.pdf'.format(path, experiment_name))
    plt.close()
    
    thetas = [round(1.015 + .02*i, 3) for i in range(10)]
    xAxis = [[] for i in range(10)]
    yAxis = [[] for i in range(10)]
    delta = 1
    thetaIndex = 0
    for job in results:
        xAxis[thetaIndex].append(job['total_crossing'])
        yAxis[thetaIndex].append(job['best_normalized_pnl'])

        if delta % 10 == 0:
            delta = 1
            thetaIndex += 1
        else:
            delta += 1

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
            elif otherP[0] <= testP[0] and otherP[1] > testP[1] :
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
            
    plt.scatter(nonFronteirX, nonFronteirY, label= "Non Frontier", color = 'grey') 
    for i in range(10):
        plt.scatter(fronteirX[i], fronteirY[i], label = "Theta = {}".format(str(thetas[i])))          
    plt.title('Pareto frontier of gas fees (total crossing) and maximum PNL')
    plt.xlabel('Total Crossing')
    plt.ylabel('Expected PNL')
    # plt.xlim(0,40)
    plt.legend()
    plt.savefig('{}/{}_crossing_pnl_frontier.pdf'.format(path, experiment_name))
    plt.close()

    
def change_both_theta_and_delta_risk_averse(experiment_name, results, path):
    thetas = [round(1.015 + .02*i, 3) for i in range(10)]
    xAxis = [[] for i in range(10)]
    yAxis = [[] for i in range(10)]
    delta = 1
    thetaIndex = 0
    for job in results:
        xAxis[thetaIndex].append(job['total_crossing'])
        yAxis[thetaIndex].append(job['best_utility_projected'])

        if delta % 10 == 0:
            delta = 1
            thetaIndex += 1
        else:
            delta += 1

    for i in range(len(xAxis)):
        plt.scatter(xAxis[i], yAxis[i], label = "Theta = {}".format(str(thetas[i])))

    plt.title('Total Crossing vs Utility.')
    plt.xlabel('Total Crossing')
    plt.ylabel('Expected Utiltiy')
    #plt.xlim(0,40)
    plt.legend()
    plt.savefig('{}/{}_crossing_utility.pdf'.format(path, experiment_name))
    plt.close()
    
    thetas = [round(1.015 + .02*i, 3) for i in range(10)]
    xAxis = [[] for i in range(10)]
    yAxis = [[] for i in range(10)]
    delta = 1
    thetaIndex = 0
    for job in results:
        xAxis[thetaIndex].append(job['total_crossing'])
        yAxis[thetaIndex].append(job['best_utility_projected'])

        if delta % 10 == 0:
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
            elif otherP[0] <= testP[0] and otherP[1] > testP[1] :
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
            
    plt.scatter(nonFronteirX, nonFronteirY, label= "Non Frontier", color = 'grey') 
    for i in range(10):
        plt.scatter(fronteirX[i], fronteirY[i], label = "Theta = {}".format(str(thetas[i])))          
    plt.title('Pareto frontier of gas fees (total crossing) and maximum Utility')
    plt.xlabel('Total Crossing')
    plt.ylabel('Expected Utility')
    # plt.xlim(0,40)
    plt.legend()
    plt.savefig('{}/{}_crossing_utility_frontier.pdf'.format(path, experiment_name))
    plt.close()


def fix_theta_and_delta_change_risk_averse(experiment_name, results, path):
    xAxis = []
    yAxis = []

    for job in results:
        xAxis.append(job['risk_a'])
        yAxis.append(job['best_utility_projected'])

    plt.title('Risk Paramter vs Utility.')
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

    plt.title('Risk Paramter vs PnL.')
    plt.xlabel('Risk Averseness')
    plt.ylabel('Best PnL')
    plt.plot(xAxis, yAxis, '-o')
    plt.savefig('{}/{}_pnl_a.pdf'.format(path, experiment_name))
    plt.close()
    
    xAxis = []
    yAxis = []

    for job in results:
        xAxis.append(job['risk_a'])
        yAxis.append(job['best_pnl_std_projected'])

    plt.title('Risk Paramter vs Standard Deviation of PnL.')
    plt.xlabel('Risk Averseness')
    plt.ylabel('Std of PnL')
    plt.plot(xAxis, yAxis, '-o')
    plt.savefig('{}/{}_std_pnl_a.pdf'.format(path, experiment_name))
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
        ax_list[i].set_ylim([0.0, 25.0])
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
            
        if 'fix_theta_change_delta' in experiment_name and 'risk_averse' in experiment_name:
            fix_theta_change_delta_risk_averse(experiment_name, results, path)
        elif 'fix_theta_change_delta' in experiment_name:
            fix_theta_change_delta_risk_neutral(experiment_name, results, path)
        elif 'change_both_theta_and_delta_risk_neutral' in experiment_name:
            change_both_theta_and_delta_risk_neutral(experiment_name, results, path)
        elif 'change_both_theta_and_delta_risk_averse' in experiment_name:
            change_both_theta_and_delta_risk_averse(experiment_name, results, path)
        elif 'fix_theta_and_delta_change_risk_averse' in experiment_name:
            fix_theta_and_delta_change_risk_averse(experiment_name, results, path)
        else:
            raise Exception('unknown experiment type: {}'.format(experiment_name))


if __name__ == '__main__':
    handle_file('result_batch_11_new.npy')
