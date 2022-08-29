import numpy as np
import math
import random
import matplotlib.pyplot as plt
import torch
from scipy.special import comb, gamma
from multiprocessing import Process, Queue
from time import sleep
import datetime
import scipy
from pathos.multiprocessing import ProcessingPool


#Delta functions

#For delta_y, arguments are ascending prices
def delta_y(pa, pb):
    return math.sqrt(pb) - math.sqrt(pa)


#For delta_x, arguments are descending prices
def delta_x(pb, pa):
    return 1/math.sqrt(pa) - 1/math.sqrt(pb)


#Value of 1 unit of liquidity in [a,b] bucket
#at p of ETH in contract
#a,b,p are prices
def bucket_value_ETH(a, b, p):
    #Price below bucket value, liquidity entirely in ETH
    if (p < a):
        return delta_x(b, a)
    #Price in bucket, liquidity in ETH and USDC
    elif(p <= b):
        return delta_x(b, p)
    #Price above bucket value, liquidity entirely in USDC
    else:
        return 0


#Value of 1 unit of liquidity in [a,b] bucket
#at p of ETH in contract
#a,b,p are prices
def bucket_value_USDC(a, b, p):
    #Price above bucket value, liquidity entirely in USDC
    if (p > b):
        return delta_y(a, b)
    #Price in bucket, liquidity in ETH and USDC
    elif(p >= a):
        return delta_y(a, p)
    #Price below bucket value, liquidity entirely in ETH
    else:
        return 0

    
#Value of 1 unit of liquidity in [a,b] bucket in USDC value
def valueOfLiquidity(a, b, p):
    if p < a:
        return p * delta_x(b, a)
    elif p > b: 
        return delta_y(a, b)
    else:
        return p * delta_x(b, p) + delta_y(a, p)


def createTransitionMatrix(dim):
    M = np.zeros((dim,dim))
    #Taking care of corner cases
    #left-most state
    # random.seed(3)
    # p = random.uniform(0,1)
    p = .5
    M[0,0] = 1-p
    M[0,1] = p
    #right-most state
    # p = random.uniform(0,1)
    p = .5
    M[dim-1,dim-2] = 1-p
    M[dim-1,dim-1] = p
    
    #All other states
    for i in range(1,dim-1):
        # p = random.uniform(0,1)
        p = .5
        M[i,i-1] = 1-p
        M[i,i+1] = p
    
    return M

def Beta(x,y):
    return (gamma(x) * gamma(y)) / gamma(x + y)

def binomialMatrix(n, w, p):
    #Array to hold the computed binomial expansion coefficients
    coeffs = [0 for i in range(n)]
    matrix = np.zeros((n,n))
    #Compute the coefficients given w, n, p

    for k in range(-w, w+1):
        newC = comb(w * 2, w + k, exact = True) * (p ** (w+k)) * ((1-p) ** (w-k))
        coeffs[k + n//2] = newC

    #Set the desired coefficients to the location in the matrix. matrix[i][j] represents a jump from i to j
    for i in range(len(matrix)):
        for j in range(max(0, i - w), min(len(matrix), i + w + 1)):
            matrix[i][j] = coeffs[n // 2 + (j - i)]

    sum_of_rows = matrix.sum(axis=1)
    normalizedMatrix = matrix / sum_of_rows[:, np.newaxis]
    #print(normalizedMatrix)
    #plt.plot(coeffs)
    return normalizedMatrix


def betaBinomialMatrix(n, w, alpha, beta):
    #Array to hold the computed binomial expansion coefficients
    coeffs = [0 for i in range(n)]
    matrix = np.zeros((n,n))
    #Compute the coefficients given w, n, alpha, beta
    k = 0
    for i in range(max(0, n // 2 - w), min(n // 2 + 1 + w, n)):
        newC = comb(w * 2, k, exact = True) * Beta(k + alpha, w * 2 - k + beta)/Beta(alpha, beta)
        coeffs[i] = newC
        k += 1
    
    #Set the desired coefficients to the location in the matrix. matrix[i][j] represents a jump from i to j
    for i in range(len(matrix)):
        for j in range(max(0, i - w), min(len(matrix), i + w + 1)):
            matrix[i][j] = coeffs[n // 2 + (j - i)]
    

    sum_of_rows = matrix.sum(axis=1)
    normalizedMatrix = matrix / sum_of_rows[:, np.newaxis]
    #print(normalizedMatrix)
    #plt.plot(coeffs)
    return normalizedMatrix


def getPrices(numPrices, p_low=0.0, p_high=10.0):
    p_diff = (p_high - p_low) / (numPrices - 1)
    prices = [p_low + i * p_diff for i in range(numPrices)]
    return prices

def getMultiplicativePrices(numPrices, p_low=0.1, p_high=10.0):
    p_div = (p_high / p_low)**(1/(numPrices - 1))
    prices = [p_low * (p_div ** i) for i in range(numPrices)]
    return prices, p_div


################# Sampling ###################

def riskAverse(x, a):
    if a != 0:
        return (1 - torch.exp(-a*x))/a
    else:
        return x

def createBuckets(bucket_endpoints):
    buckets = []
    
    for bucket_id in range(1, len(bucket_endpoints)):
        newBucket = {'id': bucket_id - 1,'pLow': bucket_endpoints[bucket_id - 1], 
                'pHigh': bucket_endpoints[bucket_id], 'earnedETH': 0, 'earnedUSD': 0}
        buckets.append(newBucket)
    return buckets


#a: left price, b: right price, p: current price  --> value of liquidity
def transactionFeeInStep(a, b, p1, p2, fee_rate):
    #Return USD, ETH
    if (p1 < a and p2 < a) or (p1 > b and p2 > b) or p1 == p2:
        return 0, 0
    if (p1 < p2):
        return fee_rate * delta_y(max(p1, a), min(p2, b)), 0
    elif (p1 > p2):
        return 0, fee_rate * delta_x(min(b, p1), max(p2, a))


def getPriceMovementSample(num_prices, transitionMatrix, p0Index, T=100):
    numPrices = len(transitionMatrix)
    currentIndex = p0Index
    pNextIndex = None
    priceIndices = [currentIndex]
    for i in range(T):
        pNextIndex = np.random.choice(num_prices, 1, p=transitionMatrix[currentIndex])[0]
        priceIndices.append(pNextIndex)
        currentIndex = pNextIndex
    
    return priceIndices


# return USDC, ETH
def bucketAmountLocked(a, b, currentPrice, l):

    if currentPrice < a:
        #all in Eth
        return 0, l*delta_x(b, a)
    elif currentPrice > b:
        return l*delta_y(a, b), 0
    else:
        return l*delta_y(a, currentPrice), l*delta_x(b, currentPrice)


def sequence_pnl_advanced_arb_and_none_arb(prices, priceIndices, buckets, fee_rate, num_non_arb, non_arb_lambda):
    for bucket in buckets:
        bucket['earnedUSD'] = 0.0
        bucket['earnedETH'] = 0.0
        bucket['crossing'] = 0.0
    
    external_price = prices[priceIndices[0]]
    current_pool_price = external_price
    price_seq = [current_pool_price]
    for i in range(1, len(priceIndices)):
        for j in range(num_non_arb):
            if np.random.random() < 0.5:
                current_pool_price *= (1 - non_arb_lambda)
            else:
                current_pool_price /= (1 - non_arb_lambda)
            price_seq.append(current_pool_price)
            
            if current_pool_price < (1.0 - fee_rate) * external_price:
                current_pool_price = (1.0 - fee_rate) * external_price
            elif current_pool_price > external_price / (1.0 - fee_rate):
                current_pool_price = external_price / (1.0 - fee_rate)
            price_seq.append(current_pool_price)
            
        external_price = prices[priceIndices[i]]
        if current_pool_price < (1.0 - fee_rate) * external_price:
            current_pool_price = (1.0 - fee_rate) * external_price
        elif current_pool_price > external_price / (1.0 - fee_rate):
            current_pool_price = external_price / (1.0 - fee_rate)
        price_seq.append(current_pool_price)
        

    for i in range(1, len(price_seq)):
        for bucket in buckets:
            earnedValue = transactionFeeInStep(bucket['pLow'], bucket['pHigh'], price_seq[i - 1], price_seq[i], fee_rate)
            bucket['earnedUSD'] += earnedValue[0]
            bucket['earnedETH'] += earnedValue[1]
            if price_seq[i - 1] == price_seq[i]:
                continue
            if (price_seq[i - 1] <= bucket['pLow'] and price_seq[i] > bucket['pLow']) or \
               (price_seq[i - 1] > bucket['pLow'] and price_seq[i] <= bucket['pLow']):
               bucket['crossing'] += 1

    # endPrice = price_seq[-1]
    # use external price
    endPrice = prices[priceIndices[-1]]
    
    for bucket in buckets:
        #Earned Transation Fee in USDC
        earnedTransaction = (bucket['earnedUSD'] + endPrice * bucket['earnedETH'])
        initialLockedAssets = bucketAmountLocked(bucket['pLow'], bucket['pHigh'], price_seq[0], 1.0)
        finalLockedAssets = bucketAmountLocked(bucket['pLow'], bucket['pHigh'], price_seq[-1], 1.0)
        impermanentLoss = (finalLockedAssets[0] + finalLockedAssets[1]*endPrice) - (initialLockedAssets[0] + initialLockedAssets[1]*endPrice)
        
        profitLoss = earnedTransaction + impermanentLoss 
        
        bucket['earnedTransactionFee'] = earnedTransaction
        bucket['impermanentLoss'] = impermanentLoss
        bucket['profitLoss'] = profitLoss
    
    return buckets


# project x to the closest z where sum(z) <= 1 and z >= 0
def project_to_sum_le_one(x):
    d = x.size
    
    # firstly project x to the first quadrant
    x = np.maximum(x, np.zeros(d))
    
    # check if sum(x) <= 1
    if np.sum(x) <= 1.0:
        return x
    
    x_list = []
    for i in range(d):
        x_list.append((i, x[i]))
    x_list = sorted(x_list, key=lambda t: t[1])
    
    # find the correct K
    v_last = None
    for i in range(d):
        K = i + 1
        if K == 1:
            v_i = (np.sum(x) - 1) / (d - K + 1)
        else:
            v_i = (v_last - x_list[i - 1][1] / (d - K + 2)) * (d - K + 2) / (d - K + 1)
        if (i == 0 or v_i >= x_list[i-1][1]) and (v_i < x_list[i][1]):
            break
        v_last = v_i
    
    z = np.zeros(d)
    for i in range(d):
        if i + 1 < K:
            z[x_list[i][0]] = 0.0
        else:
            z[x_list[i][0]] = x_list[i][1] - v_i
    
    return z


def gradientDescent(prices, bucket_endpoints, transitionMatrix, p0Index, fee_rate, num_non_arb, non_arb_lambda, riskAverseness=1.0, learning_rate=1e-3, 
                    max_training_steps=10000, min_delta=1e-5, patience=10, num_samples=1000, T=100):
    # to compute the expected utility for v2 liquidity
    v2_bucket_endpoints = [0.0, np.inf]
    v2_buckets = createBuckets(v2_bucket_endpoints)
    
    if len(bucket_endpoints) - 1 >= 3:
        # remove the first and last buckets: 0 and inf
        bucket_endpoints = bucket_endpoints[1:][:-1]
    
    buckets = createBuckets(bucket_endpoints)
    numBuckets = len(buckets)
    num_prices = len(prices)
    
    ret = {}

    # get samples of price sequence
    pnlArr = [[] for i in range(num_samples)]
    v2_utility = 0.0
    v2_expected_pnl = 0.0
    v2_price_of_liq = valueOfLiquidity(0.0, np.inf, prices[p0Index])
    total_crossing = 0.0
    for i in range(num_samples):
        priceIndices = getPriceMovementSample(num_prices, transitionMatrix, p0Index, T) # get one sample of price sequence
        pnl_buckets = sequence_pnl_advanced_arb_and_none_arb(prices, priceIndices, buckets, fee_rate, num_non_arb, non_arb_lambda)
        for bucket in pnl_buckets:
            total_crossing += bucket['crossing']
        for pnl_bucket in pnl_buckets:
            pnlArr[i].append(pnl_bucket['profitLoss'])
        # for v2
        v2_pnl_buckets = sequence_pnl_advanced_arb_and_none_arb(prices, priceIndices, v2_buckets, fee_rate, num_non_arb, non_arb_lambda)
        v2_pnl = v2_pnl_buckets[0]['profitLoss']
        v2_normalized_pnl = v2_pnl / v2_price_of_liq
        v2_expected_pnl += v2_normalized_pnl
        v2_utility += riskAverse(torch.tensor([v2_normalized_pnl]), riskAverseness)
    v2_expected_pnl /= num_samples
    v2_utility /= num_samples
    total_crossing /= num_samples
    ret['v2_expected_pnl'] = v2_expected_pnl
    ret['v2_expected_utility'] = v2_utility
    ret['total_crossing'] = total_crossing
    
    inverse_price_of_liq_np = np.array([1 / (valueOfLiquidity(bucket['pLow'], bucket['pHigh'], prices[p0Index])) for bucket in buckets])
    inverse_price_of_liq = torch.from_numpy(inverse_price_of_liq_np)

    ret['learning_rate'] = learning_rate
    ret['max_training_steps'] = max_training_steps
    ret['min_delta'] = min_delta
    ret['patience'] = patience
    ret['num_samples'] = num_samples
    
    dtype = torch.float
    device = torch.device("cpu")
    softmax = torch.nn.Softmax(dim=0)
    
    initial_theta = np.random.rand(numBuckets) #randomly initialized length-K vector # k is the number of buckets
    
    # projected gradient descent
    
    theta = torch.tensor(scipy.special.softmax(initial_theta.copy()), device=device, dtype=dtype, requires_grad=True)
    # optimizer = torch.optim.Adam([theta], lr=learning_rate)
    
    best_loss = np.inf
    best_theta = None
    best_i = None
    patience_count = 0
    
    # loss_list = []
    
    # print('riskAverseness:', riskAverseness)

    # max_training_steps = 10000
    # learning_rate = 1e-2
    for i in range(max_training_steps):
        theta = torch.tensor(project_to_sum_le_one(theta.detach().numpy()).copy(), device=device, dtype=dtype, requires_grad=True)
        
        loss = 0.0
        
        units_of_liq = inverse_price_of_liq * theta
        
        for j in range(num_samples):
            pnl_arr_j = torch.Tensor(pnlArr[j]).double()
            total_pnl = torch.dot(pnl_arr_j, units_of_liq)
            utility = riskAverse(total_pnl, riskAverseness)
            # utility = total_pnl
            loss -= utility
        
        loss /= num_samples
        
        # loss_list.append(loss)
        
        if loss < best_loss - min_delta:
            best_loss = loss
            best_theta = theta
            best_i = i
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
               break
        
        loss.backward()
        
        with torch.no_grad():
            theta = theta - learning_rate * theta.grad

        theta.requires_grad = True

    #     if i % 10 == 0:
    #         print('theta:', theta)
    #         print('i:', i)
    # print('best_theta:', best_theta)
    # print('best_i:', best_i)
    
    best_x = best_theta
    best_units_of_liq = best_theta * inverse_price_of_liq
    best_utility = 0.0
    best_pnl_list = []
    for i in range(num_samples):
        pnl_arr_i = torch.Tensor(pnlArr[i]).double()
        total_pnl = torch.dot(pnl_arr_i, best_units_of_liq)
        utility = riskAverse(total_pnl, riskAverseness)
        best_utility += utility
        best_pnl_list.append(total_pnl.detach().numpy())
    best_utility /= num_samples
    best_expected_pnl = np.mean(best_pnl_list)
    best_pnl_std = np.std(best_pnl_list)
    
    best_theta = best_theta.detach().numpy()
    best_x = best_x.detach().numpy()
    best_units_of_liq = best_units_of_liq.detach().numpy()
    best_utility = best_utility.detach().numpy()
    
    ret['best_loss_projected'] = best_loss
    ret['best_theta_projected'] = best_theta
    ret['best_x_projected'] = best_x
    ret['iteration_count_projected'] = i
    ret['best_utility_projected'] = best_utility
    ret['best_units_of_liq_projected'] = best_units_of_liq
    ret['best_expected_pnl_projected'] = best_expected_pnl
    ret['best_pnl_std_projected'] = best_pnl_std
    # ret['loss_list_projected'] = loss_list
    
    return ret


#####################################################

# p_low
# p_high
# num_prices: number of discrete prices
# multiplicative_prices: True or False
# p0_index
# W: largest possible one-step transition distance
# M_p: Binomial transition matrix parameter
# T: time horizon
# risk_a: risk-averse parameter (a=None only calls the conditional method, a=0 for risk-neutral with sampling, a>0 for risk-averse)
# theta: theta^i decides the virtual price ticks
# delta: bucket spacing
def solve(run_args):
    (job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda) = run_args
    if multiplicative_prices:
        prices, p_div = getMultiplicativePrices(num_prices, p_low=p_low, p_high=p_high)
        if M_p is None:
            # pick the value of M_p which provides the martingale property 
            m = math.log(p_div)
            M_p = -math.sqrt((m**2 + 4)/4/m**2)+(m+2)/2/m
    else:
        prices = getPrices(num_prices, p_low=p_low, p_high=p_high)
        if M_p is None:
            M_p = 0.5
    M = binomialMatrix(num_prices, W, M_p)
    
    ret = {}
    ret['job_id'] = job_id
    ret['p_low'] = p_low
    ret['p_high'] = p_high
    ret['num_prices'] = num_prices
    ret['multiplicative_prices'] = multiplicative_prices
    ret['p0_index'] = p0_index
    ret['W'] = W
    ret['M_p'] = M_p
    ret['T'] = T
    ret['risk_a'] = risk_a
    ret['theta'] = theta
    ret['delta'] = delta
    ret['fee_rate'] = fee_rate
    ret['num_non_arb'] = num_non_arb
    ret['non_arb_lambda'] = non_arb_lambda
    
    # try:
    if True:
        '''
        bucket_endpoints = []
        p = 1.0
        while p > p_low:
            p /= theta ** delta
        while p < p_high:
            bucket_endpoints.append(p)
            p *= theta ** delta
        bucket_endpoints.append(p)
        '''

        if p_low < 1.0:
            bucket_endpoints = []
            p = 1.0
            bucket_endpoints.append(p)
            while p > p_low:
                p /= theta ** delta
                bucket_endpoints.append(p)
            p = 1.0
            while p < p_high:
                p *= theta ** delta
                bucket_endpoints.append(p)
            bucket_endpoints.sort()
        else:
            bucket_endpoints = []
            p = 1.0
            while p * (theta ** delta) < p_low:
                p *= theta ** delta
            bucket_endpoints.append(p)
            while p < p_high:
                p *= theta ** delta
                bucket_endpoints.append(p)

        # append 0 and inf
        bucket_endpoints = [0.0] + bucket_endpoints + [np.inf]
        num_buckets = len(bucket_endpoints) - 1
        
        ret['bucket_endpoints'] = bucket_endpoints

        # calls the sampling method
        if risk_a is not None:
            sampling_ret = gradientDescent(prices, bucket_endpoints, M, p0_index, fee_rate, num_non_arb, non_arb_lambda, risk_a, learning_rate=1e-2, 
                        max_training_steps=10000, min_delta=1e-5, patience=1000, num_samples=1000, T=T)
            ret = {**ret, **sampling_ret}
        else:
            assert False
    # except:
    #    print('job_id: {} error!'.format(job_id))
        
    return ret

    
if __name__ == "__main__":
    experiments = {}
    job_list = []
    job_id_to_idx = {}
    job_idx = 0
    

    '''
    batch beta 14 experiments: learning_rate=1e-2, max_training_steps=10000, min_delta=1e-5, patience=100
    batch beta 15 experiments: learning_rate=5e-3, max_training_steps=10000, min_delta=1e-5, patience=1000
    batch beta 16 experiments: learning_rate=1e-2, max_training_steps=10000, min_delta=1e-5, patience=1000, price range centered around 1

    27: price range centered around 1000
    '''

    omega = 1.0005
    p_low = omega ** (-150)
    p_high = omega ** 150
    num_prices = 301
    p0_index = num_prices // 2
    multiplicative_prices = True
    M_p = None # compute the M_p value that satisfies martingale property

    for W in [3, 5, 7]:
        '''
        # fix theta change delta risk averse 0.0
        experiment_name = 'beta_fix_theta_change_delta_omega_risk_averse_0.0_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a = 0.0
        theta = 1.002
        delta_list = [i + 1 for i in range(40)]
        fee_rate = 0.01 / (1 - 0.01)
        num_non_arb = 10
        non_arb_lambda = 0.00025
        for delta in delta_list:
            job_id = 'beta_fix_theta_{}_change_delta_{}_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, risk_a, W)
            experiments[experiment_name].append(job_id)
            job_id_to_idx[job_id] = job_idx
            job_idx += 1
            job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))

        # fix theta change delta exponential risk averse 0.0
        experiment_name = 'beta_fix_theta_change_delta_exponential_risk_averse_0.0_omega_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a = 0.0
        theta = 1.002
        delta_list = [2**i for i in range(6)]
        fee_rate = 0.01 / (1 - 0.01)
        num_non_arb = 10
        non_arb_lambda = 0.00025
        for delta in delta_list:
            job_id = 'beta_fix_theta_{}_change_delta_{}_exponential_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, risk_a, W)
            experiments[experiment_name].append(job_id)
            job_id_to_idx[job_id] = job_idx
            job_idx += 1
            job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))

        # change K risk averse 0.0
        experiment_name = 'beta_change_k_omega_risk_averse_0.0_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a = 0.0
        theta = 1.002
        delta = 1
        fee_rate = 0.01 / (1 - 0.01)
        num_non_arb_list = [1 + i for i in range(20)]
        non_arb_lambda = 0.00025
        for num_non_arb in num_non_arb_list:
            job_id = 'beta_change_k_{}_risk_a_{}_omega_1.005_W_{}'.format(num_non_arb, risk_a, W)
            experiments[experiment_name].append(job_id)
            job_id_to_idx[job_id] = job_idx
            job_idx += 1
            job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))

        # change lambda risk averse 0.0
        experiment_name = 'beta_change_lambda_omega_risk_averse_0.0_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a = 0.0
        theta = 1.002
        delta = 1
        fee_rate = 0.01 / (1 - 0.01)
        num_non_arb = 10
        non_arb_lambda_list = [0.00005 + i * 0.00002 for i in range(20)]
        for non_arb_lambda in non_arb_lambda_list:
            job_id = 'beta_change_lambda_{}_risk_a_{}_omega_1.005_W_{}'.format(non_arb_lambda, risk_a, W)
            experiments[experiment_name].append(job_id)
            job_id_to_idx[job_id] = job_idx
            job_idx += 1
            job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))

        # fix theta change delta risk averse 20.0
        experiment_name = 'beta_fix_theta_change_delta_omega_risk_averse_20.0_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a = 20.0
        theta = 1.002
        delta_list = [i + 1 for i in range(40)]
        fee_rate = 0.01 / (1 - 0.01)
        num_non_arb = 10
        non_arb_lambda = 0.00025
        for delta in delta_list:
            job_id = 'beta_fix_theta_{}_change_delta_{}_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, risk_a, W)
            experiments[experiment_name].append(job_id)
            job_id_to_idx[job_id] = job_idx
            job_idx += 1
            job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))

        # fix theta change delta exponential risk averse 20.0
        experiment_name = 'beta_fix_theta_change_delta_exponential_risk_averse_20.0_omega_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a = 20.0
        theta = 1.002
        delta_list = [2**i for i in range(6)]
        fee_rate = 0.01 / (1 - 0.01)
        num_non_arb = 10
        non_arb_lambda = 0.00025
        for delta in delta_list:
            job_id = 'beta_fix_theta_{}_change_delta_{}_exponential_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, risk_a, W)
            experiments[experiment_name].append(job_id)
            job_id_to_idx[job_id] = job_idx
            job_idx += 1
            job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))

        # change K risk averse 20.0
        experiment_name = 'beta_change_k_omega_risk_averse_20.0_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a = 20.0
        theta = 1.002
        delta = 1
        fee_rate = 0.01 / (1 - 0.01)
        num_non_arb_list = [1 + i for i in range(20)]
        non_arb_lambda = 0.00025
        for num_non_arb in num_non_arb_list:
            job_id = 'beta_change_k_{}_risk_a_{}_omega_1.005_W_{}'.format(num_non_arb, risk_a, W)
            experiments[experiment_name].append(job_id)
            job_id_to_idx[job_id] = job_idx
            job_idx += 1
            job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))

        # change lambda risk averse 20.0
        experiment_name = 'beta_change_lambda_omega_risk_averse_20.0_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a = 20.0
        theta = 1.002
        delta = 1
        fee_rate = 0.01 / (1 - 0.01)
        num_non_arb = 10
        non_arb_lambda_list = [0.00005 + i * 0.00002 for i in range(20)]
        for non_arb_lambda in non_arb_lambda_list:
            job_id = 'beta_change_lambda_{}_risk_a_{}_omega_1.005_W_{}'.format(non_arb_lambda, risk_a, W)
            experiments[experiment_name].append(job_id)
            job_id_to_idx[job_id] = job_idx
            job_idx += 1
            job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))
    
        # change theta, delta and fee_rate, risk_averse = 20.0
        experiment_name = 'beta_change_theta_delta_and_fee_rate_risk_averse_20.0_omega_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a = 20.0
        theta_list = [1.002 + i * 0.002 for i in range(5)]
        delta_list = [1 + i * 2 for i in range(5)]
        fee_rate_list = [0.0005 / (1 - 0.0005), 0.003 / (1 - 0.003), 0.005 / (1 - 0.005), 0.01 / (1 - 0.01), 0.02 / (1 - 0.02)]
        num_non_arb = 10
        non_arb_lambda = 0.00025
        for fee_rate in fee_rate_list:
            for theta in theta_list:
                for delta in delta_list:
                    job_id = 'beta_change_theta_{}_delta_{}_and_fee_rate{}_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, fee_rate, risk_a, W)
                    experiments[experiment_name].append(job_id)
                    job_id_to_idx[job_id] = job_idx
                    job_idx += 1
                    job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))

        # change both theta and delta, risk_averse = 10.0
        experiment_name = 'beta_change_both_theta_and_delta_risk_averse_10.0_omega_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a = 10.0
        theta_list = [1.002 + i * 0.002 for i in range(10)]
        delta_list = [i + 1 for i in range(20)]
        fee_rate = 0.01 / (1 - 0.01)
        num_non_arb = 10
        non_arb_lambda = 0.00025
        for theta in theta_list:
            for delta in delta_list:
                job_id = 'beta_change_both_theta_{}_and_delta_{}_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, risk_a, W)
                experiments[experiment_name].append(job_id)
                job_id_to_idx[job_id] = job_idx
                job_idx += 1
                job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))
    
        # change theta, delta and fee_rate, risk_averse = 10.0
        experiment_name = 'beta_change_theta_delta_and_fee_rate_risk_averse_10.0_omega_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a = 10.0
        theta_list = [1.002 + i * 0.002 for i in range(5)]
        delta_list = [1 + i * 2 for i in range(5)]
        fee_rate_list = [0.0005 / (1 - 0.0005), 0.003 / (1 - 0.003), 0.005 / (1 - 0.005), 0.01 / (1 - 0.01), 0.02 / (1 - 0.02)]
        num_non_arb = 10
        non_arb_lambda = 0.00025
        for fee_rate in fee_rate_list:
            for theta in theta_list:
                for delta in delta_list:
                    job_id = 'beta_change_theta_{}_delta_{}_and_fee_rate{}_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, fee_rate, risk_a, W)
                    experiments[experiment_name].append(job_id)
                    job_id_to_idx[job_id] = job_idx
                    job_idx += 1
                    job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))

        # change both theta and delta, risk_averse = 0.0
        experiment_name = 'beta_change_both_theta_and_delta_risk_averse_0.0_omega_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a = 0.0
        theta_list = [1.002 + i * 0.002 for i in range(10)]
        delta_list = [i + 1 for i in range(20)]
        fee_rate = 0.01 / (1 - 0.01)
        num_non_arb = 10
        non_arb_lambda = 0.00025
        for theta in theta_list:
            for delta in delta_list:
                job_id = 'beta_change_both_theta_{}_and_delta_{}_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, risk_a, W)
                experiments[experiment_name].append(job_id)
                job_id_to_idx[job_id] = job_idx
                job_idx += 1
                job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))

        # change theta, delta and fee_rate, risk_averse = 0.0
        experiment_name = 'beta_change_theta_delta_and_fee_rate_risk_averse_0.0_omega_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a = 0.0
        theta_list = [1.002 + i * 0.002 for i in range(10)]
        delta_list = [i + 1 for i in range(10)]
        fee_rate_list = [0.01 / (1 - 0.01)]
        num_non_arb = 10
        non_arb_lambda = 0.00025
        for fee_rate in fee_rate_list:
            for theta in theta_list:
                for delta in delta_list:
                    job_id = 'beta_change_theta_{}_delta_{}_and_fee_rate{}_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, fee_rate, risk_a, W)
                    experiments[experiment_name].append(job_id)
                    job_id_to_idx[job_id] = job_idx
                    job_idx += 1
                    job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))
        
        # fix theta and delta, change risk_averse
        experiment_name = 'beta_fix_theta_and_delta_change_risk_averse_omega_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a_list = [i * 2.0 for i in range(12)]
        theta = 1.002
        delta = 1
        fee_rate = 0.01 / (1 - 0.01)
        num_non_arb = 10
        non_arb_lambda = 0.00025
        for risk_a in risk_a_list:
            job_id = 'beta_fix_theta_{}_and_delta_{}_change_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, risk_a, W)
            experiments[experiment_name].append(job_id)
            job_id_to_idx[job_id] = job_idx
            job_idx += 1
            job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))
        '''
        pass

    for W, num_non_arb, non_arb_lambda in [(3, 5, 0.00020), (7, 15, 0.00030)]:
        # change both theta and delta, risk_averse = 20.0
        experiment_name = 'beta_change_both_theta_and_delta_risk_averse_20.0_omega_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a = 20.0
        theta_list = [1.002 + i * 0.002 for i in range(10)]
        delta_list = [i + 1 for i in range(20)]
        fee_rate = 0.01 / (1 - 0.01)
        for theta in theta_list:
            for delta in delta_list:
                job_id = 'beta_change_both_theta_{}_and_delta_{}_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, risk_a, W)
                experiments[experiment_name].append(job_id)
                job_id_to_idx[job_id] = job_idx
                job_idx += 1
                job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))

        # change both theta and delta, risk_averse = 0.0
        experiment_name = 'beta_change_both_theta_and_delta_risk_averse_0.0_omega_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a = 0.0
        theta_list = [1.002 + i * 0.002 for i in range(10)]
        delta_list = [i + 1 for i in range(20)]
        fee_rate = 0.01 / (1 - 0.01)
        for theta in theta_list:
            for delta in delta_list:
                job_id = 'beta_change_both_theta_{}_and_delta_{}_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, risk_a, W)
                experiments[experiment_name].append(job_id)
                job_id_to_idx[job_id] = job_idx
                job_idx += 1
                job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))

        # fix theta and delta, change risk_averse
        experiment_name = 'beta_fix_theta_and_delta_change_risk_averse_omega_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        T = 100
        risk_a_list = [i * 2.0 for i in range(12)]
        theta = 1.002
        delta = 1
        fee_rate = 0.01 / (1 - 0.01)
        for risk_a in risk_a_list:
            job_id = 'beta_fix_theta_{}_and_delta_{}_change_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, risk_a, W)
            experiments[experiment_name].append(job_id)
            job_id_to_idx[job_id] = job_idx
            job_idx += 1
            job_list.append((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta, fee_rate, num_non_arb, non_arb_lambda))


    pool = ProcessingPool(nodes=96)
    job_output_list = pool.map(solve, job_list)
    
    # process result files
    result = {}
    for experiment_name in experiments:
        job_id_list = experiments[experiment_name]
        result[experiment_name] = []
        for job_id in job_id_list:
            result[experiment_name].append(job_output_list[job_id_to_idx[job_id]])
    
    np.save('results/result_batch_beta_x.npy', result)


