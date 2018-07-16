from __future__ import print_function
import numpy as np
import itertools as it
import cPickle as pkl
from os.path import isdir, isfile, join
from os import mkdir
import networkx as nx
from mpi4py import MPI

from exploit_core import ExploitCore
from stateNetwork import state


def scatterInputArr(comm, inputArr):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        np.random.shuffle(inputArr)
        chunks = [[] for _ in range(size)]
        for i, chunk in enumerate(inputArr):
            chunks[i % size].append(chunk)

    else:
        chunks = None

    scatteredInput = comm.scatter(chunks, root=0)
    return scatteredInput


def run_networkGrid(parameterList,
                    networkGerneratingFunction,
                    sampleSize=200,
                    savePath='/p/tmp/geier/default',
                    redo=False, networkParameterUser=[600, 0.1, 0.1], networkParameterGa=[60, 0.1, 0.1]):

        # Check constantVariable to initilize the to be varied parameters

        for parameterSet in parameterList:
            constantVariable, constantValue, QoI, runNumber = parameterSet

            if constantVariable == 'phi':  # phi                     deltaT                 lambda
                variParameter = it.product([constantValue], np.linspace(0.02, 2., 100), np.linspace(0.0, 1., 101))

            elif constantVariable == 'deltaT':  # phi                     deltaT                 lambda
                variParameter = it.product(np.linspace(0.0, 1., 101), [constantValue], np.linspace(0.0, 1., 101))

            elif constantVariable == 'lambda':  # phi                     deltaT                 lambda
                variParameter = it.product(np.linspace(0.0, 1., 101), np.linspace(0.02, 2., 100), [constantValue])

            else:
                print('Problem with the constantVariable! Typo?')  # Hopefully this is not needed!

            savePathNameFracUser = get_savePathName(constantVariable=constantVariable, constantValue=constantValue, QoI='fracSus', sampleRun=runNumber, savePath=savePath)
            savePathNameFracGA = get_savePathName(constantVariable=constantVariable, constantValue=constantValue, QoI='fracSusGa', sampleRun=runNumber, savePath=savePath)
            savePathNameTime = get_savePathName(constantVariable=constantVariable, constantValue=constantValue, QoI='consensTime', sampleRun=runNumber, savePath=savePath)

            susUser = []
            susGa = []
            conTime = []

            for parameter in variParameter:
                #print("computing ", parameter)

                if isfile(savePathNameTime) and not redo:
                    continue

                userData, gaData, time = calc_finalValue(parameter[0], parameter[1], parameter[2], networkParameterUser=networkParameterUser,
                                                         networkParameterGa=networkParameterGa, networkGerneratingFunction=networkGerneratingFunction)

                susUser.append(userData)
                susGa.append(gaData)
                conTime.append(time)

            susUser = np.array(susUser)
            susGa = np.array(susGa)
            conTime = np.array(conTime)

            # ppData = [np.mean(userPP), np.mean(gaPP)]
            # pkl.dump(ppData, open(savePathName, 'wb'))

            print("Saving: ", constantValue, runNumber)
            print("Shape of susUser: " + str(susUser.shape))
            print("Shape of susGa: " + str(susGa.shape))
            print("Shape of conTime: " + str(conTime.shape))
            #np.save(savePathNameFracUser, susUser)
            #np.save(savePathNameFracGA, susGa)
            #np.save(savePathNameTime, conTime)
            pkl.dump(susUser, open(savePathNameFracUser, "wb"))
            pkl.dump(susGa, open(savePathNameFracGA, "wb"))
            pkl.dump(conTime, open(savePathNameTime, "wb"))
            print("Saved")


def calc_finalValue(phi=0.3,
                    tau=0.3,
                    taxation_factor=0.6,
                    networkParameterUser=[600, 0.1, 0.1],
                    networkParameterGa=[60, 0.1, 0.1],
                    networkGerneratingFunction=nx.watts_strogatz_graph,
                    verbose=False):

    # from link density to mean degree
    #networkParameterUser = list(networkParameterUser)
    #networkParameterUser[1] = int(networkParameterUser[1] * networkParameterUser[0])
    #networkParameterGa = list(networkParameterGa)
    #networkParameterGa[1] = int(networkParameterGa[1] * networkParameterGa[0])

    adjacency_matrix_agents = nx.adj_matrix(networkGerneratingFunction(*networkParameterUser)).toarray()

    adjacency_matrix_states = nx.adj_matrix(networkGerneratingFunction(*networkParameterGa)).toarray()
    N_agents = adjacency_matrix_agents.shape[0]
    N_states = adjacency_matrix_states.shape[0]

    rewiring_prob_agents = phi
    rewiring_prob_states = phi
    update_timescale_agents = tau
    update_timescale_states = tau * (N_agents / N_states)**2
    taxation_factor = taxation_factor

    if type(adjacency_matrix_agents[0][0]) != np.int64:
        #print('The type of the adjacency matrix aAgents is casted to np.int64.')
        adjacency_matrix_agents = adjacency_matrix_agents.astype(int)
    if type(adjacency_matrix_agents) != np.int64:
        #print('The type of the adjacency matrix aAgents is casted to np.int64.')
        adjacency_matrix_states = adjacency_matrix_states.astype(int)

    strategies_agents = np.random.randint(2, size=N_agents)  # Agent's Strategies
    stocks_agents = np.ones(N_agents)                    # Agent's Stocks
    max_stocks_agents = np.ones(N_agents)                 # Agent's Maximum Stocks (Capacities)

    growth_rates_agents = np.ones(N_agents)                    # Agent's Growth Rates
    rationalities_agents = np.ones(N_agents)

    strategies_states = np.random.randint(2, size=N_states)
    rationalities_states = np.ones(N_states)

    # ######Initializing the System#############
    states = state(adjacency_matrix_states, strategies_states, rationalities_states,
                   taxation_factor, rewiring_prob_states, update_timescale_states, N_agents, N_states)

    exploit = ExploitCore(adjacency_matrix_agents, strategies_agents, stocks_agents,
                        max_stocks_agents, growth_rates_agents, rationalities_agents,
                        rewiring_prob_agents, update_timescale_agents, N_agents, N_states)

    a_stateID = np.append(np.arange(0, N_states), np.random.randint(N_states, size=N_agents - N_states))
    np.random.shuffle(a_stateID)

    exploit.set_stateID(a_stateID)
    states.set_stateID(a_stateID)

    harvest = np.zeros(N_agents)

    # ######### Run it#######

    while True:
        # stockVec= exploit.get_stocks()
        states.set_states_stocks(exploit.get_stocks())

        update_time_state = states.update_system()
        if exploit.get_time_min() > update_time_state:
            continue

        action_time = exploit.get_time() + update_time_state

        taxation_vec = states.create_taxation_vec()

        exploit.set_taxation(taxation_vec)

        consensus, harvest = exploit.run(action_time=action_time)

        if consensus == -1 or consensus == 1:
            break

    return [np.mean(exploit.get_strategies()), np.mean(states.get_strategies()), exploit.get_time()]


def get_savePathName(constantVariable='dummy', constantValue=100, QoI='dummy', sampleRun=999,
                    savePath='/p/tmp/geier/default'):

        fileName = constantVariable + '_' + str(constantValue) + '_' + QoI + '_run_' + str(sampleRun) + '.pkl'
        fileName.replace('.', 'o')
        savePath = join(savePath, fileName)
        return savePath


def avoidDoubleCalculation(parameterList, savePath, redo=False):
    '''
        Checks if a parameter setting/sampling run has already been computed to avoid douple computations.

    '''
    newParameterList = []
    for parameter in parameterList:
        constantVariable, constantValue, QuI, runNumber = parameter
        savePathName = get_savePathName(constantVariable, constantValue, 'consensusTime', runNumber, savePath)
        if not isfile(savePathName) or redo:                               # Ugly but it should work. Make sure that consensusTime file is saved last
            newParameterList.append(parameter)

    return newParameterList


if __name__ == '__main__':
    comm = MPI.COMM_WORLD

    # ##################### savePath NEEDS TO BE ADAPTED ###################################
    savePath = '/p/tmp/barfuss/Taxploit/constantDeltaT'
    # savePath = '/home/fabian/Berlin-Potsdam/Masterarbeit/Code/Taxploit/'
    # savePath = 'savePathLambda'
    userNodes_arr = 600  # Btw.: len(range(225, 701, 25)) == 20
    gaNodes_arr = 60
    
    # TESTING TESTING TESTING ====================== ##
    userNodes_arr = 200
    gaNodes_arr = 20
    # savePath = "../TestData/constantDeltaT"
    # ============================================== ##
    
    link_density = 0.05
    # initalRew = 0.5

    # constantVariable = ['deltaT']  # one possiblity of three!
    # constantValue = [0., 0.5, 1.]  # np.linspace(0.02, 2., 100)

    # also has to be computed
    constantVariable = ['deltaT']
    constantValue = [0.5, 1.0, 1.5]  # np.linspace(0.0, 1., 101)

    # also has to be computed
    # constantVariable = ['phi']
    # constantValue = [0., 0.4, 0.8]  # np.linspace(0.0, 1., 101)

    QoI = ['fracSus', 'consensTime']  # This might not be a good idea as they are computed in the same run.

    sampleRuns = np.arange(100)  # 100 sample runs

    # TESTING TESTING TESTING ====================== ##
    # sampleRuns = np.arange(3)  # 100 sample runs
    # ============================================== ##

    networkGerneratingFunction = nx.watts_strogatz_graph
    networkGerneratingFunction = nx.erdos_renyi_graph
    
    # list(it.product(phi_arr, tau_arr, tax_arr, userNetworkParameter_Iterator, gaNetworkParameter_Iterator))
    parameterList = list(it.product(constantVariable, constantValue, QoI, sampleRuns))
    # constantVariable, constantValue, QoI, runNumber = parameter

    if comm.Get_rank() == 0:
        parameterList = avoidDoubleCalculation(parameterList, savePath)
        print("Number of datapoints that need to be simulated: {}".format(len(parameterList)))
    else:
        parameterList = None

    parameterList = scatterInputArr(comm, parameterList)  # Distribute parameterList to the different processes
    # print("Rank: {}; ParameterList: {}".format(comm.Get_rank(), parameterList))
    print (comm.rank, len(parameterList))

    print("Starting the simulation; Rank: {}".format(comm.Get_rank()))
    nwParamUser=[userNodes_arr, link_density]
    nwParamGa=[gaNodes_arr, link_density]
    run_networkGrid(parameterList, networkGerneratingFunction=networkGerneratingFunction, savePath=savePath,
                    networkParameterUser=nwParamUser, networkParameterGa=nwParamGa, redo=True)
