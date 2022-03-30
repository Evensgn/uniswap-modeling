#Importing relevant libraries
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

#We want to come up with a black-box that takes as input the following:
#Underlying universe instance:
#-Price ticks
#-Bucket info, in the form of:
#   -Endpoints of buckets in a list
#Specific LP instance:
#Transition matrix M
#Risk-averseness a = 0 (We only do the risk-neutral)
#Budget W


#Important auxilary functions:

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


#In this function, we take the transition matrix and initial state of the system
#And output the expected transaction fees (in USDC)
#Inputs:
#-Transition matrix M
#-Width of matrix, i.e. for each price p_i, how many p_j does it maximally transition to: W 
#-Initial price distribution: x0
#-Time horizon: T
#-Bucket indices
#-Reward matrix
#
#Outputs:
#Expected transaction fees in USDC


def generate_fees_tensor(M,W,R,traversals,x0,T,prices):
    
############################################################

    #This stuff used to be computed in the function
    #Now this is all passed explicitly as argument

    #Generating reward matrix
    #R = construct_rewards_tensor(prices,bucket_indices)

############################################################

    # print("This is traversals tensor we're given")
    # print(traversals.dtype)
    # print(traversals[0,15])
    # print()

    #State space cardinality (Number of prices) 
    k = M.shape[0]

    #Declaring number of buckets
    #Remember that R should have shape (k x k x |B|)
    num_buckets = R.shape[-1]
    
    #This tensor will hold price distributions at given time
    #at indices [:,t] it holds the distribution of prices at time t
    price_dist_time = np.zeros((k,T+1))
    
    #Now we populate the price at time t tensor
    price_dist_time[:,0] = x0
    for t in range(1,T+1):
        price_dist_time[:,t] = M @ price_dist_time[:,t-1]
    
    #Now we declare the total fee tensor (in USDC)
    #the (t,z)-th entry represents the following:
    #Expected fee rewards in the transition from t -> t+1 conditional on final price index being z
    #Note conditionining on z allows us to translate ETH to USDC
    #This final fees is a B vector, i.e. giving us the fee per bucket
    fees = np.zeros((T,k,num_buckets))
    
    #Now we declare the tensor that holds endpoints traveresed in expectation
    #the (t,z)-th entry represents the following:
    #Expected bucket endpoints traveresed in the transition from t -> t+1 conditional on final price index being z
    #This final cond_traversal[i,j] is a B vector, i.e. giving us expected traversals of left endpoint of B-th bucket
    cond_traversal = np.zeros((T,k,num_buckets))
    
    #Placeholder for what will be M^{T-t} for a given t = T-1,...,0
    rem_M = np.identity(k)
    
    #Placeholder for conditional price distributions
    conditional = np.zeros((k,))
    
    #Now we iterate for t =  T-1,T-2,...,0
    for t in range(T-1,-1,-1):
        #Updating the remaining transition matrix
        #Invariant is that rem_M = M^{T-t}
        rem_M  =  M @ rem_M
        
        #Iterating over end price (indices) for process
        for z in range(0,k):
            
            #Updating vector for conditional distribution at time t given final price
            for i in range(0,k):
                #Use Bayes:
                #Formula is P(i,t | z) = P(i,t) * Prob transition from i to z over T-t turns / P(z,T)
                conditional[i] = price_dist_time[i,t]*rem_M[z,i]/price_dist_time[z,T]
                
                
            #Given conditional distribution, we can compute expected fees over t -> t+1
            #price increases give USDC fees and decreases ETH fees
            #But we know end price is z, so we can compute in USDC
            
            #Iterating over price at t (P_i)
            for i in range(0,k):
                #We use the width to make computational savings
                #We are using the fact that only W prices left and right can be transition into
                lower_j_index = max(0,i-W)
                upper_j_index = min(k,i+W+1)

                #If the sparsity factor, W, is not too small, we can comment the above out, and
                #Revert to for j in range(0,k):

                #Iterating over price at t+1 (P_j)
                #for j in range(0,k):
                for j in range(lower_j_index,upper_j_index):
                    #Rewards for i to i are by default 0, so we need to only consider movements
                    if(j != i):
                        #Computing the probability of this transition
                        transition_prob = conditional[i]*M[j,i]
                        
                        #First the case where i < j (Price increased)
                        #Fees are in USDC, so don't need to translate
                        if (i<j):
                            fees[t,z,:] += transition_prob*R[j,i,:]
                            
                            # if(np.max(traversals[j,i,:]) > 0):
                            #     print("Adding traversals: ",traversals[j,i,:])
                            #     print("With probability: ", transition_prob*traversals[j,i,:])
                            #     print("Before update", cond_traversal[t,z,:])
                            #Adding expected traversals
                            cond_traversal[t,z,:] += transition_prob*traversals[j,i,:]
                            
                            # if(np.max(traversals[j,i,:]) > 0):
                            #     print("After update", cond_traversal[t,z,:])
                            #     print()
                            
                        #Now case where i > j (Price decreased)
                        #Fees are in ETH, so need to translate to USDC
                        else:
                            fees[t,z,:] += prices[z]*transition_prob*R[j,i,:]
                            
                            # if(np.max(traversals[j,i,:]) > 0):
                            #     print("Adding traversals: ",traversals[j,i,:])
                            #     print("With probability: ", transition_prob*traversals[j,i,:])
                            #     print("Before update", cond_traversal[t,z,:])
                            #Adding expected traversals
                            cond_traversal[t,z,:] += transition_prob*traversals[j,i,:]
                            
                            # if(np.max(traversals[j,i,:]) > 0):
                            #     print("After update", cond_traversal[t,z,:])
                            #     print()

    #At this point, fees[t,z,b] holds the expected fees (in USDC) earned in going from 
    #t -> t+1
    #Conditional on end price being z
    #For bucket b 
    #Similar for cond_traversal

    #Now we compute the total expected fees and conditional traversals
    
    #This holds the expected fees (summed over all time-steps t), conditional on end index z
    #In other words, what is left is a |B| x |P| dimensional vector, where the [b,:] entry holds the
    #a vector with expected fees of bucket b, conditional on end price being z, (one entry for each end price)
    conditional_summed_fees = np.sum(fees,axis = 0).transpose()
    
    conditional_summed_traversals = np.sum(cond_traversal,axis = 0).transpose()

    #To get the final fees, we matmul with xT (distribution at time T)
    total_fees = conditional_summed_fees @ price_dist_time[:,T]

    total_traversals = conditional_summed_traversals @ price_dist_time[:,T]
    
    #We also pass the final price distribution to help with final value distribution
    return total_fees, price_dist_time[:,T], total_traversals 


#In this function, we take the transition matrix and initial state of the system
#And output the expected value of 1 bucket 
#Inputs:
#- p0_index, the initial price (index) used to purchase liquidity
#- End price distribution (After T turns)
#- Bucket endpoints
#- Liquidity distribution
# (Implicitly) R = reward matrix
#
#Outputs:
#Expected value of 1 unit of liquidity in each bucket
#Expected value of holding assets worth 1 unit of liquidity purchased at price p0


def IL_per_bucket(p0_index,price_dist,prices,bucket_endpoints):
    
######################################################################

#This is now passed to the function as an argument

    #First we compute the distribution of time T prices
    #price_dist = x0
    #for i in range(0,T):
        #price_dist = M @ price_dist

######################################################################

    #number of buckets
    num_buckets = len(bucket_endpoints)-1
    
    #number of prices
    k = price_dist.shape[0]
    
    #Declaring the value matrix
    #(b,j)-th entry has value of 1 unit of liquidity for b-th bucket at price p[j]
    value_matrix = np.zeros((num_buckets,k))
    hold_matrix = np.zeros((num_buckets,k))

    #Populating the value/holding matrix
    for bucket in range(0,num_buckets):

        #Bucket characteristics
        a = bucket_endpoints[bucket]
        b = bucket_endpoints[bucket+1]

        #Getting initial holdings in ETH and UDSDC
        initial_eth_value = bucket_value_ETH(a,b,prices[p0_index])
        initial_USDC_value = bucket_value_USDC(a,b,prices[p0_index])

        #Iterating over price indices
        for price_index in range(0,k):
            p = prices[price_index]

            #Populating holding matrix when end price is p
            hold_matrix[bucket,price_index] = p*initial_eth_value + initial_USDC_value 

            #Populating the value_matrix

            #Price below bucket, liquidity is in ETH only
            if (p <= a):
                #Accrue the required end value, Multiply by probability of p
                #Transfer to USDC
                value_matrix[bucket,price_index] = p*delta_x(b,a)
                
            #Price above bucket, liquidity in USDC only
            elif (p >= b):
                #Accrue the required end value, Multiply by probability of p
                value_matrix[bucket,price_index] = delta_y(a,b)
                
            #Price in bucket, liquidity in both ETH and USDC
            else:
                #Accrue the required end value, Multiply by probability of p
                value_matrix[bucket,price_index] = p*delta_x(b,p) + delta_y(a,p)

    #The overall list of values/holding is simply the product of the value/hold matrix
    #with the distribution of prices at time T
    value_list = value_matrix @ price_dist
    hold_list = hold_matrix @ price_dist

    return hold_list, value_list


#GRAND FUNCTION
#Puts everything together
#Input:
#- Price list: prices
#- Bucket characteristics: bucket_indices
#- Current/Initial price index: p0
#- Transition matrix: M
#- Sparseness factor of matrix: W (max degree of markov chain as graph)
#- Time horizon T 
#Returns (all values for 1 unit of liquidity in bucket):
#- fee_vector: vector of all expected fees per bucket
#- expected_final_value: vector of all expected final values per bucket
#- initial_values: vector of initial value per bucket

def PnL(prices,bucket_endpoints, M, W, T, p0_index):
    
    #First we compute the rewards tensor and traversals tensor
    #This can be replaced by any other way of computing the reward tensor
    #R = construct_rewards_tensor(prices,bucket_endpoints)    
    R, traversals = construct_rewards_transversals_tensors(prices,bucket_endpoints)

    #We compute the initial price vector
    pre_initial = [0]*len(prices)
    pre_initial[p0_index] = 1
    x0 = np.array(pre_initial)

    #Now we create the vector of fees, and get passed the final price distribution
    fee_vector, price_dist, traversals_vector = generate_fees_tensor(M,W,R,traversals,x0,T,prices)

    #Now we compute the expected hold/final value of the buckets
    expected_hold_values, expected_final_values  = IL_per_bucket(p0_index,price_dist,prices,bucket_endpoints)

    return(fee_vector, expected_hold_values, expected_final_values, traversals_vector)


#Function to take a price index and bucket characteristics
#to return which bucket(s) a price belongs to
#Input:
#price list
#bucket list. B_i corresponds to [buckets[i],buckets[i+1]]
#Need to have buckets[0] = smallest price and buckets[1] = largest price
#Output:
#list for each price which bucket it belongs to
def which_bucket(prices,buckets):
    #Relevant parameters
    num_buckets = len(buckets)-1
    num_prices = len(prices)
    
    #Declaring the array which will hold solution
    #in_bucket[j] tells us which buckets prices[j] belongs to
    in_bucket = []
    
    bucket_index = 0
    for i in range(num_prices):
        while prices[i] > buckets[bucket_index + 1]:
            bucket_index += 1
        in_bucket.append([bucket_index])
    
    return in_bucket


#function to construct reward tensor AND TRAVERSAL TENSOR
#Takes as input:
#-Price list (k = n+m+1 prices)
#-Bucket endpoint list (Not limited to price tick endpoints!)
#Outputs:
#-Reward Matrix tensor (kxkxB) where R_{ij} is reward in going
#from P_j -> P_i
#Price increase means USDC, otherwise ETH
#R_{ij} is a B vector, which holds per-bucket rewards
#Assumed that 1 unit of liquidity in each bucket
#-Traversal tensor (kxkxB) where T_{ij} is indicator vector
#for which bucket endpoints are traversed. Index k represents
#the left endpoint of a bucket, so it being set to 1 means that a_k was traversed

def construct_rewards_transversals_tensors(prices,buckets):
    
    #obtaining number of states
    num_prices = len(prices)
    #print("Number of prices: ",num_prices)
    #print()
    
    #obtaining number of buckets
    num_buckets = len(buckets)-1
    #print("Number of buckets: ",num_buckets)
    #print()
     
    #Declaring the reward tensor to return
    rewards = np.zeros((num_prices,num_prices,num_buckets))
    
    #Declaring the traversal tensor to return
    traversals = np.zeros((num_prices,num_prices,num_buckets))
    
    #Declaring the bucket map from before
    bucket_map = which_bucket(prices,buckets)
    #print("Bucket map: ",bucket_map)
    #print()
    
    ##################################################################
    #First we flesh out rewards for adjacent price movements
    #print("Fleshing out adjacent price movements")
    #print()
    ##################################################################
    
    #Iterating over prices
    for price_index in range(0,num_prices):
        
        #print("Current price index: ",price_index)
        #print()
        
        ######################################  
        #Special cases for first and last price
        ###################################### 
        
        ################### 
        #First price
        ################### 
        if(price_index == 0):
            
            #print("Price index is 0! Only need to consider upward movement")
            #print()
            
            ####################################### 
            #Upward movement (only at this price)
            ####################################### 
            #Declaring next prices for upward movement
            next_price_index = price_index + 1
            #print("Next price index:", next_price_index)
            
            #initial bucket for upward movement
            #If initial price in two buckets, take right one
            # current_buck = bucket_map[price_index][-1]
            #print("bucket of current price: ",current_buck)
            current_buck = bucket_map[price_index][0]
                    
            #If next price in two buckets, take left one
            target_buck = bucket_map[next_price_index][0]
            #print("bucket of target price: ",target_buck)
            #print()
            
            #Initial price for movement and target price (above initial)
            current_price = prices[price_index]
            target_price = prices[next_price_index]
            #print("current price",current_price)
            #print("target_price", target_price)
            #print()
            
            
            #In case jump traverses multiple buckets
            #We first cover all complete bucket jumps
            while(current_buck < target_buck):
                #Moving price from current price to next left bucket endpoint
                #And accumulating fees in correct bucket as we go
                
                #print("Still at least one bucket apart!")
                #print()
                
                #Right end point of the current bucket, which must be smaller than target price
                #(by virtue of while loop condition)
                current_buck_end_price = buckets[current_buck + 1]
                #print("Right endpoint of current bucket", current_buck_end_price)
                #print()
                
                #Accumulating movement from current price to the bucket end
                #print("Accumulating movement from current price to bucket end")
                #print("Before setting reward",rewards[next_price_index,price_index])
                rewards[next_price_index,price_index,current_buck] += delta_y(current_price, current_buck_end_price)
                #print("After setting reward",rewards[next_price_index,price_index])
                #print()
                
                #print("Reassigning price")
                #Re-assigning price
                current_price = current_buck_end_price
                #Moving to the next bucket
                current_buck += 1
                #print("new price: ", current_price)
                #print("current bucket: ", current_buck)
                #print()
                
                #################
                #TRAVERSAL COUNT#
                #################
                
                #print("Counting traversals")
                #Counting the traversal of the bucket endpoint
                #The left endpoint of this new bucket was traversed
                #print("Traversed from bucket {} to bucket {}".format(current_buck-1, current_buck))
                #print("Before updating: ", traversals[next_price_index,price_index])
                traversals[next_price_index,price_index,current_buck] += 1
                #print("After updating: ", traversals[next_price_index,price_index])
                
            #Now we pick up the final price increase
            #Move from current_price to target_price
            #print("Getting rewards from final price movement (in same bucket)")
            #print("Current_buck: ", current_buck)
            #print("Target buck: ", target_buck)
            #print("Before updating: ", rewards[next_price_index,price_index])
            rewards[next_price_index,price_index,current_buck] += delta_y(current_price, target_price)
            #print("After updating: ",rewards[next_price_index,price_index])
            #print()
            
        ###################     
        #Last price
        ################### 
        elif(price_index == num_prices-1):
            
            #print("Price index is last! Only need to consider downward movement")
            #print()
            
            ####################################### 
            #Downward movement (only at this price)
            ####################################### 
            #Declaring next prices for downward movement
            next_price_index = price_index - 1
            #print("Next price index:", next_price_index)
            
            #initial bucket for downward movement
            #If initial price in two buckets, take left one
            current_buck = bucket_map[price_index][0]
            #print("bucket of current price: ",current_buck)
        
            #If next price in two buckets, take right one
            # target_buck = bucket_map[next_price_index][-1]
            #print("bucket of target price: ",target_buck)
            #print()
            target_buck = bucket_map[next_price_index][0]
            
            #Initial price for movement and target price (above initial)
            current_price = prices[price_index]
            target_price = prices[next_price_index]
            #print("current price",current_price)
            #print("target_price", target_price)
            #print()
            
            #In case jump traverses multiple buckets
            #We first cover all complete bucket jumps
            while(current_buck > target_buck):
                #Moving price from current price to next left bucket endpoint
                #And accumulating fees in correct bucket as we go
                
                #print("Still at least one bucket apart!")
                #print()
                
                #left end point of the current bucket, which must be larger than target price
                #(by virtue of while loop condition)
                current_buck_end_price = buckets[current_buck]
                #print("Left endpoint of current bucket", current_buck_end_price)
                #print()
                
                #Accumulating movement from current price to the bucket end (downward movement)
                #print("Accumulating movement from current price to bucket end")
                #print("Before setting reward",rewards[next_price_index,price_index])
                rewards[next_price_index,price_index,current_buck] += delta_x(current_price, current_buck_end_price)
                #print("After setting reward",rewards[next_price_index,price_index])
                #print()
                
                #################
                #TRAVERSAL COUNT#
                #################
                
                #print("Counting traversals")
                #Counting the traversal of the bucket endpoint
                #The left endpoint of the current bucket is traversed
                #print("Traversed from bucket {} to bucket {}".format(current_buck, current_buck-1))
                #print("Before updating: ", traversals[next_price_index,price_index])
                traversals[next_price_index,price_index,current_buck] += 1
                #print("After updating: ", traversals[next_price_index,price_index])
                #print()
                
                #print("Reassigning price")
                #Re-assigning price
                current_price = current_buck_end_price
                #Moving to the next bucket
                current_buck -= 1
                #print("new price: ", current_price)
                #print("current bucket: ", current_buck)
                #print()
                
            #Now we pick up the final price increase
            #Move from current_price to target_price
            #print("Getting rewards from final price movement (in same bucket)")
            #print("Current_buck: ", current_buck)
            #print("Target buck: ", target_buck)
            #print("Before updating: ", rewards[next_price_index,price_index])
            rewards[next_price_index,price_index,current_buck] += delta_x(current_price, target_price)
            #print("After updating: ",rewards[next_price_index,price_index])
            #print()
            
        #########################################################      
        #Intermediate price, both upward and downward movement
        #########################################################  
        else:
            
            #print("Intermediate Price! Need to consider upwards and downwards movement")
            #print()
            
            #print("Considering upward movement")
            #print()
            ####################
            #Upward movement
            ####################
            #Declaring next prices for upward movement
            next_price_index = price_index + 1
            #print("Next price index:", next_price_index)
            
            #initial bucket for upward movement
            #If initial price in two buckets, take right one
            # current_buck = bucket_map[price_index][-1]
            #print("bucket of current price: ",current_buck)
            current_buck = bucket_map[price_index][0]
        
            #If next price in two buckets, take left one
            target_buck = bucket_map[next_price_index][0]
            #print("bucket of target price: ",target_buck)
            #print()
            
            #Initial price for movement and target price (above initial)
            current_price = prices[price_index]
            target_price = prices[next_price_index]
            #print("current price",current_price)
            #print("target_price", target_price)
            #print()
            
            #In case jump traverses multiple buckets
            #We first cover all complete bucket jumps
            while(current_buck < target_buck):
                #Moving price from current price to next left bucket endpoint
                #And accumulating fees in correct bucket as we go
                
                #print("Still at least one bucket apart!")
                #print()
                
                #Right end point of the current bucket, which must be smaller than target price
                #(by virtue of while loop condition)
                current_buck_end_price = buckets[current_buck + 1]
                #print("Right endpoint of current bucket", current_buck_end_price)
                #print()
                
                #Accumulating movement from current price to the bucket end
                #print("Accumulating movement from current price to bucket end")
                #print("Before setting reward",rewards[next_price_index,price_index])
                rewards[next_price_index,price_index,current_buck] += delta_y(current_price, current_buck_end_price)
                #print("After setting reward",rewards[next_price_index,price_index])
                #print()
                
                #print("Reassigning price")
                #Re-assigning price
                current_price = current_buck_end_price
                #Moving to the next bucket
                current_buck += 1
                #print("new price: ", current_price)
                #print("current bucket: ", current_buck)
                #print()
                
                #################
                #TRAVERSAL COUNT#
                #################
                
                #print("Counting traversals")
                #Counting the traversal of the bucket endpoint
                #The left endpoint of this new bucket was traversed
                #print("Traversed from bucket {} to bucket {}".format(current_buck-1, current_buck))
                #print("Before updating: ", traversals[next_price_index,price_index])
                traversals[next_price_index,price_index,current_buck] += 1
                #print("After updating: ", traversals[next_price_index,price_index])
                
            #Now we pick up the final price increase
            #Move from current_price to target_price
            #print("Getting rewards from final price movement (in same bucket)")
            #print("Current_buck: ", current_buck)
            #print("Target buck: ", target_buck)
            #print("Before updating: ", rewards[next_price_index,price_index])
            rewards[next_price_index,price_index,current_buck] += delta_y(current_price, target_price)
            #print("After updating: ",rewards[next_price_index,price_index])
            #print()
            
            #print("Considering downward movement")
            #print()
            ####################
            #Downward movement
            ####################
            #Declaring next prices for upward movement
            next_price_index = price_index - 1
            #print("Next price index:", next_price_index)
            
            #initial bucket for downward movement
            #If initial price in two buckets, take left one
            current_buck = bucket_map[price_index][0]
            #print("bucket of current price: ",current_buck)
        
            #If next price in two buckets, take right one
            # target_buck = bucket_map[next_price_index][-1]
            #print("bucket of target price: ",target_buck)
            #print()
            target_buck = bucket_map[next_price_index][0]
            
            #Initial price for movement and target price (above initial)
            current_price = prices[price_index]
            target_price = prices[next_price_index]
            #print("current price",current_price)
            #print("target_price", target_price)
            #print()
            
            #In case jump traverses multiple buckets
            #We first cover all complete bucket jumps
            while(current_buck > target_buck):
                #Moving price from current price to next left bucket endpoint
                #And accumulating fees in correct bucket as we go
                
                #print("Still at least one bucket apart!")
                #print()
                
                #left end point of the current bucket, which must be larger than target price
                #(by virtue of while loop condition)
                current_buck_end_price = buckets[current_buck]
                #print("Left endpoint of current bucket", current_buck_end_price)
                #print()
                
                #Accumulating movement from current price to the bucket end (downward movement)
                #print("Accumulating movement from current price to bucket end")
                #print("Before setting reward",rewards[next_price_index,price_index])
                rewards[next_price_index,price_index,current_buck] += delta_x(current_price, current_buck_end_price)
                #print("After setting reward",rewards[next_price_index,price_index])
                #print()
                
                #################
                #TRAVERSAL COUNT#
                #################
                
                #print("Counting traversals")
                #Counting the traversal of the bucket endpoint
                #The left endpoint of the current bucket is traversed
                #print("Traversed from bucket {} to bucket {}".format(current_buck, current_buck-1))
                #print("Before updating: ", traversals[next_price_index,price_index])
                traversals[next_price_index,price_index,current_buck] += 1
                #print("After updating: ", traversals[next_price_index,price_index])
                #print()
                
                #print("Reassigning price")
                #Re-assigning price
                current_price = current_buck_end_price
                #Moving to the next bucket
                current_buck -= 1
                #print("new price: ", current_price)
                #print("current bucket: ", current_buck)
                #print()
                
                
            #Now we pick up the final price increase
            #Move from current_price to target_price
            #print("Getting rewards from final price movement (in same bucket)")
            #print("Current_buck: ", current_buck)
            #print("Target buck: ", target_buck)
            #print("Before updating: ", rewards[next_price_index,price_index])
            rewards[next_price_index,price_index,current_buck] += delta_x(current_price, target_price)
            #print("After updating: ",rewards[next_price_index,price_index])
            #print()
    
    
    ###############################################################
    #Now we flesh out price movements that are larger than adjacent
    ###############################################################

            
    #Now we flesh out rewards for arbitrary upward price movements
    #We iterate over base price index, P_i
    #print("Working on upward price movements")
    #print()
    for i in range(0,num_prices-1):
        #print("Current price index", i)
        #print()
        #We iterate over what price we move to, P_j
        #Declaring cumulative reward FOR ALL BUCKETS (As B vector)
        cumulative_reward = rewards[i+1,i,:]
        #print("current cumulative rewards for {} to {}: {}".format(i,i+1,cumulative_reward))
        
        #####
        #OLD
        #####
        #Declaring cumulative traversals FOR ALL BUCKETS (As B vector)
        #cumulative_traversals = traversals[i+1,i,:]
        #print("current cumulative traversals for {} to {}: {}".format(i,i+1,cumulative_traversals))
        #####
        #OLD
        #####
        
        #invariant: at beginning of code in loop, cumulative_reward holds rewards[j-1,i,:], i.e. vector of rewards for each bucket in moving from price i to price j-1
        #same thing for cumulative_traversals vector
        for j in range(i+2,num_prices):
            #print("target price index", j)
            #print()
            #print("Cumulative rewards from {} to {}: {}".format(i, j-1, cumulative_reward))
            #print("Rewards from {} to {}: {}".format(j-1, j, rewards[j,j-1,:]))
            rewards[j,i,:] = cumulative_reward + rewards[j,j-1,:]
            #updating cumulative reward
            #print("New total rewards: ", rewards[j,i,:])
            cumulative_reward = rewards[j,i,:]
            #print("These are now new cumulative rewards")
            #print()
            
            #####
            #OLD
            #####
            #print("Cumulative traversals from {} to {}: {}".format(i, j-1,cumulative_traversals))
            #print("Traversals from {} to {}: {}".format(j-1, j, traversals[j,j-1,:]))
            #traversals[j,i,:] = cumulative_traversals + traversals[j,j-1,:]
            #print("New total traversals: ",traversals[j,i,:])
            #updating cumulative reward
            #cumulative_traversals = traversals[j,i,:]
            #print("These are now new cumulative traversals")
            #print()
            #####
            #OLD
            #####
    
            #New way of computing traversals
            #print("This is the index of bucket of {}-th price:".format(i))
            # left_bucket_index = bucket_map[i][-1]
            left_bucket_index = bucket_map[i][0]
            #print(left_bucket_index)
            #print()
            #print("This is the index of bucket of {}-th price:".format(j))
            right_bucket_index = bucket_map[j][0]
            #print(right_bucket_index)
            #print()
            
            if (left_bucket_index < right_bucket_index):
                #print("Bucket traversals to count!")
                #print("traversals before",traversals[j,i])
                for b in range(left_bucket_index,right_bucket_index):
                    #print("Counting move from bucket {} to {}".format(b,b+1))
                    traversals[j,i,b+1] = 1
                #print("traversals after", traversals[j,i])
                
            #else:
                #print("Same bucket movement, no traversals to count")
                
    
            
    #Now we flesh out rewards for arbitrary downward price movements
    #We iterate over base price index, P_i
    #print()
    #print("Working on downward price movements")
    #print()
    
    for i in range(num_prices-1,1,-1):
        #print("Current price index", i)
        #print()
        #We iterate over what price we move to, P_j: P_i > P_j
        #Declaring cumulative reward
        cumulative_reward = rewards[i-1,i,:]
        #print("current cumulative rewards for {} to {}: {}".format(i,i-1,cumulative_reward))
        
        #####
        #OLD
        #####
        #Declaring cumulative traversals
        #cumulative_traversals = traversals[i-1,i,:]
        #print("current cumulative traversals for {} to {}: {}".format(i,i-1,cumulative_traversals))
        #####
        #OLD
        #####
        
        #invariant: at beginning of code in loop, cumulative_reward holds rewards[j+1,i]
        #same thing for cumulative_traversals vector
        for j in range(i-2,-1,-1):
            #print("target price index", j)
            #print()
            #print("Cumulative rewards from {} to {} (downwards): {}".format(i, j+1, cumulative_reward))
            #print("Rewards from {} to {} (downwards): {}".format(j+1, j, rewards[j,j+1,:]))
            rewards[j,i,:] = cumulative_reward + rewards[j,j+1,:]
            #updating cumulative reward
            #print("New total rewards: ", rewards[j,i,:])
            cumulative_reward = rewards[j,i,:]
            #print("These are now new cumulative rewards")
            #print()
            
            #####
            #OLD
            #####
            #print("Cumulative traversals from {} to {} (downwards): {}".format(i, j+1,cumulative_traversals))
            #print("Traversals from {} to {} (downwards): {}".format(j+1, j, traversals[j,j+1,:]))
            #traversals[j,i,:] = cumulative_traversals + traversals[j,j+1,:]
            #print("New total traversals: ",traversals[j,i,:])
            #updating cumulative reward
            #cumulative_traversals = traversals[j,i,:]
            #print("These are now new cumulative traversals")
            #print()
            #####
            #OLD
            #####
            
            #New way of computing traversals
            #print("This is the index of bucket of {}-th price:".format(i))
            left_bucket_index = bucket_map[i][0]
            #print(left_bucket_index)
            #print()
            #print("This is the index of bucket of {}-th price:".format(j))
            # right_bucket_index = bucket_map[j][-1]
            right_bucket_index = bucket_map[j][0]
            #print(right_bucket_index)
            #print()
            
            if (left_bucket_index > right_bucket_index):
                #print("Bucket traversals to count!")
                #print("traversals before",traversals[j,i])
                for b in range(left_bucket_index,right_bucket_index,-1):
                    #print("Counting move from bucket {} to {}".format(b,b-1))
                    traversals[j,i,b] = 1
                #print("traversals after", traversals[j,i])
                
            #else:
                #print("Same bucket movement, no traversals to count")

    
    return rewards, traversals


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
def transactionFeeInStep(a, b, p1, p2):
    #Return USD, ETH
    if (p1 < a and p2 < a) or (p1 > b and p2 > b) or p1 == p2:
        return 0, 0
    if (p1 < p2):
        return delta_y(max(p1, a), min(p2, b)), 0
    elif (p1 > p2):
        return 0, delta_x(min(b, p1), max(p2, a))

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

    
def sequence_pnl(prices, priceIndices, buckets):
    for bucket in buckets:
        bucket['earnedUSD'] = 0.0
        bucket['earnedETH'] = 0.0
    
    for i in range(1, len(priceIndices)):
        for bucket in buckets:
            earnedValue = transactionFeeInStep(bucket['pLow'], bucket['pHigh'], prices[priceIndices[i - 1]], prices[priceIndices[i]])
            bucket['earnedUSD'] += earnedValue[0]
            bucket['earnedETH'] += earnedValue[1]
    
    endPrice = prices[priceIndices[-1]]
    
    for bucket in buckets:
        #Earned Transation Fee in USDC
        earnedTransaction = (bucket['earnedUSD'] + endPrice * bucket['earnedETH'])
        initialLockedAssets = bucketAmountLocked(bucket['pLow'], bucket['pHigh'], prices[priceIndices[0]], 1.0)
        finalLockedAssets = bucketAmountLocked(bucket['pLow'], bucket['pHigh'], prices[priceIndices[-1]], 1.0)
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


def gradientDescent(prices, bucket_endpoints, transitionMatrix, p0Index, riskAverseness=1.0, learning_rate=1e-3, 
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
    
    # get samples of price sequence
    pnlArr = [[] for i in range(num_samples)]
    v2_utility = 0.0
    v2_price_of_liq = valueOfLiquidity(0.0, np.inf, prices[p0Index])
    for i in range(num_samples):
        priceIndices = getPriceMovementSample(num_prices, transitionMatrix, p0Index, T) # get one sample of price sequence
        pnl_buckets = sequence_pnl(prices, priceIndices, buckets)
        for pnl_bucket in pnl_buckets:
            pnlArr[i].append(pnl_bucket['profitLoss'])
        # for v2
        v2_pnl_buckets = sequence_pnl(prices, priceIndices, v2_buckets)
        v2_pnl = v2_pnl_buckets[0]['profitLoss']
        v2_normalized_pnl = v2_pnl / v2_price_of_liq
        v2_utility += riskAverse(torch.tensor([v2_normalized_pnl]), riskAverseness)
    v2_utility /= num_samples
    
    inverse_price_of_liq_np = np.array([1 / (valueOfLiquidity(bucket['pLow'], bucket['pHigh'], prices[p0Index])) for bucket in buckets])
    inverse_price_of_liq = torch.from_numpy(inverse_price_of_liq_np)
    
    ret = {}
    ret['learning_rate'] = learning_rate
    ret['max_training_steps'] = max_training_steps
    ret['min_delta'] = min_delta
    ret['patience'] = patience
    ret['num_samples'] = num_samples
    
    dtype = torch.float
    device = torch.device("cpu")
    softmax = torch.nn.Softmax(dim=0)
    
    initial_theta = np.random.rand(numBuckets) #randomly initialized length-K vector # k is the number of buckets
    
    '''
    # gradient descent
    
    theta = torch.tensor(initial_theta.copy(), device=device, dtype=dtype, requires_grad=True)
    
    # print("Original Theta: ", theta)
    # optimizer = torch.optim.Adam([theta], lr=learning_rate)
    
    best_loss = np.inf
    best_theta = None
    patience_count = 0
    
    # loss_list = []
    for i in range(max_training_steps):
        # print("I Index: ", i)
        # print("Theta: ", softmax(theta))
        # optimizer.zero_grad()
        loss = 0.0
        
        x = softmax(theta)
        units_of_liq = inverse_price_of_liq * x
        
        for j in range(num_samples):
            pnl_arr_j = torch.Tensor(pnlArr[j]).double()
            total_pnl = torch.dot(pnl_arr_j, units_of_liq)
            utility = riskAverse(total_pnl, riskAverseness)
            loss -= utility
        
        loss /= num_samples
        
        # loss_list.append(loss)
        
        if loss < best_loss - min_delta:
            best_loss = loss
            best_theta = theta
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break
        
        loss.backward()
        with torch.no_grad():
            theta = theta - learning_rate * theta.grad
        theta.requires_grad = True
        # optimizer.step()
    
    best_x = softmax(best_theta)
    best_units_of_liq = best_theta * inverse_price_of_liq
    best_utility = 0.0
    best_pnl_list = []
    for i in range(num_samples):
        pnl_arr_i = torch.Tensor(pnlArr[i]).double()
        total_pnl = torch.dot(pnl_arr_i, best_units_of_liq)
        utility = riskAverse(total_pnl, riskAverseness)
        best_utility += utility
        best_pnl_list.append(total_pnl)
    best_utility /= num_samples
    best_expected_pnl = np.mean(best_pnl_list)
    best_pnl_std = np.std(best_pnl_list)
 
    best_theta = best_theta.detach().numpy()
    best_x = best_x.detach().numpy()
    best_units_of_liq = best_units_of_liq.detach().numpy()
    best_utility = best_utility.detach().numpy()
    
    ret['best_loss'] = best_loss
    ret['best_theta'] = best_theta
    ret['best_x'] = best_x
    ret['iteration_count'] = i
    ret['best_utility'] = best_utility
    ret['best_units_of_liq'] = best_units_of_liq
    ret['best_expected_pnl'] = best_expected_pnl
    ret['best_pnl_std'] = best_pnl_std
    # ret['loss_list'] = loss_list
    ret['v2_utility'] = v2_utility
    '''
    
    # projected gradient descent
    
    theta = torch.tensor(scipy.special.softmax(initial_theta.copy()), device=device, dtype=dtype, requires_grad=True)
    # optimizer = torch.optim.Adam([theta], lr=learning_rate)
    
    best_loss = np.inf
    best_theta = None
    patience_count = 0
    
    # loss_list = []
    
    for i in range(max_training_steps):
        theta = torch.tensor(project_to_sum_le_one(theta.detach().numpy()).copy(), device=device, dtype=dtype, requires_grad=True)
        
        loss = 0.0
        
        units_of_liq = inverse_price_of_liq * theta
        
        for j in range(num_samples):
            pnl_arr_j = torch.Tensor(pnlArr[j]).double()
            total_pnl = torch.dot(pnl_arr_j, units_of_liq)
            utility = riskAverse(total_pnl, riskAverseness)
            loss -= utility
        
        loss /= num_samples
        
        # loss_list.append(loss)
        
        if loss < best_loss - min_delta:
            best_loss = loss
            best_theta = theta
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break
        
        loss.backward()
        
        with torch.no_grad():
            theta = theta - learning_rate * theta.grad
        theta.requires_grad = True
    
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
def solve(job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta):
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
    
    try:
    # if True:
        virtual_price_ticks = []
        p = 1.0
        while p < p_high:
            if p >= p_low:
                virtual_price_ticks.append(p)
            p *= theta
        bucket_endpoints = []
        for i in range(0, len(virtual_price_ticks), delta):
            bucket_endpoints.append(virtual_price_ticks[i])
        # append 0 and inf
        bucket_endpoints = [0.0] + bucket_endpoints + [np.inf]
        num_buckets = len(bucket_endpoints) - 1

        result = PnL(prices, bucket_endpoints, M, W, T, p0_index)
        pnl_arr = result[0] + result[2] - result[1]
        transaction_fee_arr = result[0]
        impermanent_loss_arr = result[2] - result[1]
        bucket_crossing_arr = result[3]
        price_of_liq = [valueOfLiquidity(bucket_endpoints[i], bucket_endpoints[i+1], prices[p0_index]) for i in range(num_buckets)]
        pnl_normalized_arr = [pnl_arr[i] / price_of_liq[i] for i in range(num_buckets)]
        best_normalized_pnl, best_normalized_bucket = np.max(pnl_normalized_arr), np.argmax(pnl_normalized_arr)

        # compute the PnL of [0, inf) v2 liquidity
        v2_bucket_endpoints = [0.0, np.inf]
        v2_result = PnL(prices, v2_bucket_endpoints, M, W, T, p0_index)
        v2_pnl = v2_result[0] + v2_result[2] - v2_result[1]
        v2_transaction_fee = v2_result[0]
        v2_impermanent_loss = v2_result[2] - v2_result[1]
        price_of_v2_liq = valueOfLiquidity(0.0, np.inf, prices[p0_index])
        v2_normalized_pnl = v2_pnl / price_of_v2_liq
        
        ret['bucket_endpoints'] = bucket_endpoints
        ret['virtual_price_ticks'] = virtual_price_ticks

        # the following are only for the risk-neutral case
        ret['best_normalized_pnl'] = best_normalized_pnl
        ret['best_normalized_bucket'] = best_normalized_bucket
        ret['pnl_normalized_arr'] = pnl_normalized_arr
        ret['pnl_arr'] = pnl_arr
        ret['price_of_liq'] = price_of_liq
        ret['transaction_fee_arr'] = transaction_fee_arr
        ret['impermanent_loss_arr'] = impermanent_loss_arr
        
        ret['v2_pnl'] = v2_pnl
        ret['v2_normalized_pnl'] = v2_normalized_pnl
        ret['v2_transaction_fee'] = v2_transaction_fee
        ret['v2_impermanent_loss'] = v2_impermanent_loss
        ret['price_of_v2_liq'] = price_of_v2_liq
        
        ret['bucket_crossing_arr'] = bucket_crossing_arr
        ret['total_crossing'] = sum(bucket_crossing_arr)
        
        # calls the sampling method
        if risk_a is not None:
            sampling_ret = gradientDescent(prices, bucket_endpoints, M, p0_index, risk_a, learning_rate=1e-3, 
                        max_training_steps=1000, min_delta=1e-3, patience=100, num_samples=1000, T=T)
            ret = {**ret, **sampling_ret}
    
    except:
        print('job_id: {} error!'.format(job_id))
        
    return ret


# multiprocessing Worker and dispatchWorker based on https://gist.github.com/prempv/717270a4470a10146c6776820e8e3cbc
class Worker(Process):
    def __init__(self, input_queue):
        Process.__init__(self)
        self.input_queue = input_queue

    def run(self):
        while True:
            if self.input_queue.empty():
                print('job queue is empty')
                break
            else:
                args = self.input_queue.get()
                ret = solve(*args)
                np.save('results/' + args[0] + '.npy', ret)
                print('job_id: {} finished at {}'.format(args[0], datetime.datetime.now()))
        return

    
if __name__ == "__main__":
    task_q = Queue()
    
    experiments = {}
    
    ''' batch 11 experiments '''
    for W in [3, 4, 5, 6, 7]:
        experiment_name = 'b11_fix_theta_change_delta_exponential_omega_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        p_low = 1000.0
        p_high = 1162.0
        num_prices = 301
        p0_index = num_prices // 2
        multiplicative_prices = True
        M_p = None # compute the M_p value that satisfies martingale property
        T = 100
        risk_a = None # use conditional only
        theta = 1.002
        delta_list = [2**i for i in range(6)]
        for delta in delta_list:
            job_id = 'b11_fix_theta_{}_change_delta_{}_exponential_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, risk_a, W)
            experiments[experiment_name].append(job_id)
            task_q.put_nowait((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta))

        # change both theta and delta, risk_averse = 1.0
        experiment_name = 'b11_change_both_theta_and_delta_risk_averse_1.0_omega_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        p_low = 1000.0
        p_high = 1162.0
        num_prices = 301
        p0_index = num_prices // 2
        multiplicative_prices = True
        M_p = None # compute the M_p value that satisfies martingale property
        T = 100
        risk_a = 1.0
        theta_list = [1.002 + i * 0.002 for i in range(10)]
        delta_list = [i + 1 for i in range(10)]
        for theta in theta_list:
            for delta in delta_list:
                job_id = 'b11_change_both_theta_{}_and_delta_{}_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, risk_a, W)
                experiments[experiment_name].append(job_id)
                task_q.put_nowait((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta))

        # change both theta and delta, risk_averse = 0.0
        experiment_name = 'b11_change_both_theta_and_delta_risk_averse_0.0_omega_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        p_low = 1000.0
        p_high = 1162.0
        num_prices = 301
        p0_index = num_prices // 2
        multiplicative_prices = True
        M_p = None # compute the M_p value that satisfies martingale property
        T = 100
        risk_a = 0.0
        theta_list = [1.002 + i * 0.002 for i in range(10)]
        delta_list = [i + 1 for i in range(10)]
        for theta in theta_list:
            for delta in delta_list:
                job_id = 'b11_change_both_theta_{}_and_delta_{}_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, risk_a, W)
                experiments[experiment_name].append(job_id)
                task_q.put_nowait((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta))

        # change both theta and delta, risk neutral
        experiment_name = 'b11_change_both_theta_and_delta_risk_neutral_omega_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        p_low = 1000.0
        p_high = 1162.0
        num_prices = 301
        p0_index = num_prices // 2
        multiplicative_prices = True
        M_p = None # compute the M_p value that satisfies martingale property
        T = 100
        risk_a = None
        theta_list = [1.002 + i * 0.002 for i in range(10)]
        delta_list = [i + 1 for i in range(10)]
        for theta in theta_list:
            for delta in delta_list:
                job_id = 'b11_change_both_theta_{}_and_delta_{}_risk_neutral_omega_1.005_W_{}'.format(theta, delta, W)
                experiments[experiment_name].append(job_id)
                task_q.put_nowait((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta))

        # fix theta and delta, change risk_averse, (invalid run, incorrect job_id defined)
        experiment_name = 'b11_fix_theta_and_delta_change_risk_averse_omega_1.005_W_{}'.format(W)
        experiments[experiment_name] = []
        p_low = 1000.0
        p_high = 1162.0
        num_prices = 301
        p0_index = num_prices // 2
        multiplicative_prices = True
        M_p = None # compute the M_p value that satisfies martingale property
        T = 100
        risk_a_list = [i * 0.1 for i in range(12)]
        theta = 1.002
        delta = 1
        for risk_a in risk_a_list:
            job_id = 'b11_fix_theta_{}_and_delta_{}_change_risk_a_{}_omega_1.005_W_{}'.format(theta, delta, risk_a, W)
            experiments[experiment_name].append(job_id)
            task_q.put_nowait((job_id, p_low, p_high, num_prices, multiplicative_prices, p0_index, W, M_p, T, risk_a, theta, delta))
    
    # print('total jobs:', task_q.qsize())
    
    processes = [
        Worker(task_q)
        for i in range(94)
    ]

    for proc in processes:
        proc.start()
    
    for proc in processes: 
        print('waiting to join')
        proc.join()
    
    # process result files
    result = {}
    for experiment_name in experiments:
        job_id_list = experiments[experiment_name]
        result[experiment_name] = []
        for job_id in job_id_list:
            ret = np.load('results/' + job_id + '.npy', allow_pickle=True)
            ret = ret.item()
            result[experiment_name].append(ret)
    
    np.save('results/result_batch_11_new.npy', result)


