#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random as r
import numpy as np
import matplotlib.pyplot as plt
import csv
import collections as col
import os
import time


# In[ ]:


# FUNCTIONS ---------------------------------------------------------------------------------------

# Reads in data from a given .csv file
def readData(file):
    inFile = open(file,'r')
    reader = csv.reader(inFile,delimiter = ',')

    polished = [line for line in reader if (line != [] and line != ['',''])]
            
    inFile.close()
    
    del(polished[0])
    
    polished.sort()
    
    return polished


# Updates partisan asymmetry objective function
def update_PA(FD):
    FD_copy = FD.copy()
    curve_Dem = []
    all_won = False
    all_lost = False
    
    # Increase average vote-share until dems win all seats, recording breakpoints for seat gains
    while not all_won:
        change_increase = [(0.5 - fd) for fd in FD_copy if fd < 0.5]
        if len(change_increase) == 0:
            all_won = True
            voteShare_Dem = (sum(FD_copy))/(len(FD_copy))
            seatShare_Dem = (len(FD_copy) - len(change_increase))/(len(FD_copy))
            curve_Dem.append([voteShare_Dem,seatShare_Dem])
            continue
            
        delta_increase = min(change_increase)
        
        voteShare_Dem = (sum(FD_copy))/(len(FD_copy))
        seatShare_Dem = (len(FD_copy) - len(change_increase))/(len(FD_copy))
        curve_Dem.append([voteShare_Dem,seatShare_Dem])
        
        FD_copy = [(fd + delta_increase) for fd in FD_copy] # Increase district vote-shares by enough for dems to win one more seat
        
    current = curve_Dem[0]
    del(curve_Dem[0])
    FD_copy = FD.copy()
    
    # Indicator to not record point in first loop iteration, since it's not a breakpoint
    goAhead = False
    
    # Decrease average vote-share until dems lose all seats, recording breakpoints for seat losses
    while not all_lost:
        change_decrease = [(fd - 0.5) for fd in FD_copy if fd > 0.5]
        if len(change_decrease) == 0:
            all_lost = True
            voteShare_Dem = (sum(FD_copy))/(len(FD_copy))
            seatShare_Dem = (len(change_decrease)+1)/(len(FD_copy))
            curve_Dem.append([voteShare_Dem,seatShare_Dem])
            continue
            
        delta_decrease = min(change_decrease)
        
        voteShare_Dem = (sum(FD_copy))/(len(FD_copy))
        seatShare_Dem = (len(change_decrease)+1)/(len(FD_copy))
        if goAhead:
            curve_Dem.append([voteShare_Dem,seatShare_Dem])
            
        goAhead = True
        
        FD_copy = [(fd - delta_decrease) for fd in FD_copy]
    
    
    curve_Dem.sort()
    
    # Create rep curve from dem curve
    curve_Rep = [[1-cd[0],1-cd[1]+(1/len(FD_copy))] for cd in curve_Dem]
    curve_Rep.sort()
    
    # Calculate area between curves
    integ = 0
    for i in range(0,len(curve_Dem)):
        dem = curve_Dem[i]
        rep = curve_Rep[i]
        rect = (1.0/(len(FD_copy)))*(abs(dem[0]-rep[0]))
        integ += rect
        
        
    # Plot dem and rep curves
#     x = [cd[0] for cd in curve_Dem]
#     y = [cd[1]-1.0/(len(FD_copy)) for cd in curve_Dem]
#     x.append(1.0)
#     y.append(1.0)
#     plt.step(x,y,'b',linewidth=.5,label='Party A')
    
#     x = [cd[0] for cd in curve_Rep]
#     y = [cd[1]-1.0/(len(FD_copy)) for cd in curve_Rep]
#     x.append(1.0)
#     y.append(1.0)
#     plt.step(x,y,'r--',linewidth=.5,label='Party B')
#     plt.axis([0,1,0,1])
#     plt.legend()
#     plt.xlabel('Average Vote-Share')
#     plt.ylabel('Seat-Share')
#     plt.show()
    
    return (integ)


# # Updates nominal Efficiency Gap value -- slower
# def update_EG(FD,TV):
    
#     Dem = []
#     Rep = []
#     wastedVotes = []
    
#     for i in range(0,len(TV)):
#         Dem.append(FD[i]*TV[i])
#         Rep.append((1-FD[i])*TV[i])
        
#     for i in range(0,len(Dem)):
#         if Dem[i] >= Rep[i]:
#             wastedVotes.append(.5*(Dem[i]-3*Rep[i]))
#         else:
#             wastedVotes.append(.5*(3*Dem[i]-Rep[i]))
            
#     return (round(abs(sum(wastedVotes))/totalVotes,10))



# Initializes dictionary of wasted votes in each district (nominal)
def initialize_wasted(ND,NR):
    
    # Store wasted votes for each district
    WASTED_NOMINAL = {}

    for i in range(0,len(ND)):
        if ND[i] >= NR[i]:
            WASTED_NOMINAL[i+1] = .5*(ND[i]-3*NR[i])
        else:
            WASTED_NOMINAL[i+1] = .5*(3*ND[i]-NR[i])
                
    return WASTED_NOMINAL


# Faster update of nominal efficiency gap objective function
def update_EG_fast():

    sumWasted = 0.0
    for i in range(1,numDistricts):
        sumWasted += wastedNominal[i]

    return (round(abs(sumWasted)/totalVotes,10))


# Update of nominal efficiency gap objective function during an iteration
def update_EG_fast_it(From,Too,FDFrom,FDToo,TVFrom,TVToo):
    
    NDFrom = FDFrom*TVFrom
    NRFrom = (1-FDFrom)*TVFrom
    
    NDToo = FDToo*TVToo
    NRToo = (1-FDToo)*TVToo
    
    if NDFrom >= NRFrom:
        wastedFrom = .5*(NDFrom-3*NRFrom)
    else:
        wastedFrom = .5*(3*NDFrom-NRFrom)
        
        
    if NDToo >= NRToo:
        wastedToo = .5*(NDToo-3*NRToo)
    else:
        wastedToo = .5*(3*NDToo-NRToo)
        

    sumWasted = 0.0
    for i in range(1,numDistricts):
        if i == From:
            sumWasted += wastedFrom
        elif i == Too:
            sumWasted += wastedToo
        else:
            sumWasted += wastedNominal[i]
            

    return (round(abs(sumWasted)/totalVotes,10)),wastedFrom,wastedToo


# initializes dictionary of wasted votes in each district for each vote-share shift
def initialize_wastedShift(FD,TV):
    
     # Shifted Efficiency Gap objective function (MO)
    percentShiftsNeg = [-0.05,-0.04,-0.03,-0.02,-0.01,0.0]
    percentShiftsPos = [0.01,0.02,0.03,0.04,0.05]
    #allVotes = sum(TV)
    WASTED_ALL = {}

    for s in percentShiftsNeg:
        fracDem_shift = [max(fd+s,0.0) for fd in FD]
        numDem_shift = []
        numRep_shift = []
        for i in range(0,len(TV)):
            numDem_shift.append(fracDem_shift[i]*TV[i])
            numRep_shift.append((1-fracDem_shift[i])*TV[i])
        
        #wasted_shift = []
        for i in range(0,len(numDem_shift)):
            if numDem_shift[i] >= numRep_shift[i]:
                #wasted_shift.append(.5*(numDem_shift[i]-3*numRep_shift[i]))
                WASTED_ALL[(i+1,s)] = .5*(numDem_shift[i]-3*numRep_shift[i])
            else:
                #wasted_shift.append(.5*(3*numDem_shift[i]-numRep_shift[i]))
                WASTED_ALL[(i+1,s)] = .5*(3*numDem_shift[i]-numRep_shift[i])
        
 
    for s in percentShiftsPos:
        fracDem_shift = [min(fd+s,1.0) for fd in FD]
        numDem_shift = []
        numRep_shift = []
        for i in range(0,len(TV)):
            numDem_shift.append(fracDem_shift[i]*TV[i])
            numRep_shift.append((1-fracDem_shift[i])*TV[i])
            
        #wasted_shift = []
        for i in range(0,len(numDem_shift)):
            if numDem_shift[i] >= numRep_shift[i]:
                #wasted_shift.append(.5*(numDem_shift[i]-3*numRep_shift[i]))
                WASTED_ALL[(i+1,s)] = .5*(numDem_shift[i]-3*numRep_shift[i])
            else:
                #wasted_shift.append(.5*(3*numDem_shift[i]-numRep_shift[i]))
                WASTED_ALL[(i+1,s)] = .5*(3*numDem_shift[i]-numRep_shift[i])
                
    return WASTED_ALL


# Faster update of shifted efficiency gap objective function
def update_ShiftedEG_fast():
    
    percentShifts = [-0.05,-0.04,-0.03,-0.02,-0.01,0.0,0.01,0.02,0.03,0.04,0.05]
    EG_shift = []
    
    for s in percentShifts:
        sumWasted = 0.0
        for i in range(1,numDistricts):
            sumWasted += wastedShift[(i,s)]
            
        EG_shift.append(round(abs(sumWasted)/totalVotes,10))
        
#     print('-----')
#     for val in EG_shift:
#         print(val)
        
    return max(EG_shift)
    #return EG_shift[5] # Quick change to get nominal EG value!


# Returns updated efficiency gap value during an iteration
def update_ShiftedEG_fast_it(From,Too,FDFrom,FDToo,TVFrom,TVToo):
    
    percentShiftsNeg = [-0.05,-0.04,-0.03,-0.02,-0.01,0.0]
    percentShiftsPos = [0.01,0.02,0.03,0.04,0.05]
    percentShifts = [-0.05,-0.04,-0.03,-0.02,-0.01,0.0,0.01,0.02,0.03,0.04,0.05]
    EG_shift = []
    
    wastedFrom = {}
    wastedToo = {}
    
    for s in percentShiftsNeg:
        numDemFrom_shift = (max(FDFrom+s,0.0))*TVFrom
        numRepFrom_shift = (1-max(FDFrom+s,0.0))*TVFrom
        
        numDemToo_shift = (max(FDToo+s,0.0))*TVToo
        numRepToo_shift = (1-max(FDToo+s,0.0))*TVToo
        
        if numDemFrom_shift >= numRepFrom_shift:
            wastedFrom[s] = .5*(numDemFrom_shift-3*numRepFrom_shift)
        else:
            wastedFrom[s] = .5*(3*numDemFrom_shift-numRepFrom_shift)
            
        if numDemToo_shift >= numRepToo_shift:
            wastedToo[s] = .5*(numDemToo_shift-3*numRepToo_shift)
        else:
            wastedToo[s] = .5*(3*numDemToo_shift-numRepToo_shift)
            
            
    for s in percentShiftsPos:
        numDemFrom_shift = (min(FDFrom+s,1.0))*TVFrom
        numRepFrom_shift = (1-min(FDFrom+s,1.0))*TVFrom
        
        numDemToo_shift = (min(FDToo+s,1.0))*TVToo
        numRepToo_shift = (1-min(FDToo+s,1.0))*TVToo
        
        if numDemFrom_shift >= numRepFrom_shift:
            wastedFrom[s] = .5*(numDemFrom_shift-3*numRepFrom_shift)
        else:
            wastedFrom[s] = .5*(3*numDemFrom_shift-numRepFrom_shift)
            
        if numDemToo_shift >= numRepToo_shift:
            wastedToo[s] = .5*(numDemToo_shift-3*numRepToo_shift)
        else:
            wastedToo[s] = .5*(3*numDemToo_shift-numRepToo_shift)
            
            
    for s in percentShifts:
        sumWasted = 0.0
        for i in range(1,numDistricts):
            if i == From:
                sumWasted += wastedFrom[s]
            elif i == Too:
                sumWasted += wastedToo[s]
            else:
                sumWasted += wastedShift[(i,s)]
            
        EG_shift.append(round(abs(sumWasted)/totalVotes,10))
        
    return max(EG_shift),wastedFrom,wastedToo
    #return EG_shift[5],wastedFrom,wastedToo # Quick change to get nominal EG value!


# # Updates shifted efficiency gap objective function -- slower
# def update_ShiftedEG(FD,TV):
    
#     # Shifted Efficiency Gap objective function (MO)
#     percentShiftsNeg = [-0.05,-0.04,-0.03,-0.02,-0.01,0.0]
#     percentShiftsPos = [0.01,0.02,0.03,0.04,0.05]
#     allVotes = sum(TV)
#     EG_shift = []

#     for s in percentShiftsNeg:
#         fracDem_shift = [max(fd+s,0.0) for fd in FD]
#         numDem_shift = []
#         numRep_shift = []
#         for i in range(0,len(TV)):
#             numDem_shift.append(fracDem_shift[i]*TV[i])
#             numRep_shift.append((1-fracDem_shift[i])*TV[i])
        
#         wasted_shift = []
#         for i in range(0,len(numDem_shift)):
#             if numDem_shift[i] >= numRep_shift[i]:
#                 wasted_shift.append(.5*(numDem_shift[i]-3*numRep_shift[i]))
#             else:
#                 wasted_shift.append(.5*(3*numDem_shift[i]-numRep_shift[i]))
                
#         EG_shift.append(round(abs(sum(wasted_shift))/allVotes,10))
        
        
#     for s in percentShiftsPos:
#         fracDem_shift = [min(fd+s,1.0) for fd in FD]
#         numDem_shift = []
#         numRep_shift = []
#         for i in range(0,len(TV)):
#             numDem_shift.append(fracDem_shift[i]*TV[i])
#             numRep_shift.append((1-fracDem_shift[i])*TV[i])
            
#         wasted_shift = []
#         for i in range(0,len(numDem_shift)):
#             if numDem_shift[i] >= numRep_shift[i]:
#                 wasted_shift.append(.5*(numDem_shift[i]-3*numRep_shift[i]))
#             else:
#                 wasted_shift.append(.5*(3*numDem_shift[i]-numRep_shift[i]))
                
#         EG_shift.append(round(abs(sum(wasted_shift))/allVotes,10))
        

# #     print('-----')
# #     for val in EG_shift:
# #         print(val)

#     return max(EG_shift) # For max EG shift
#     #return EG_shift[5] # For regular EG



# Creates spanning tree (Wilson's loop-erased random walk method)
def spanningTree(nodes,rt):
    
    InTree = {} # Whether a unit has been added to tree or not
    Next = {} # Successor to the index unit
    discoveredStack = col.deque() # Stack to keep track of nodes in the tree in case loop is formed
    nodes_set = set(nodes)
    
    for n in nodes:
        InTree[n] = False
        
    InTree[rt] = True
    Next[rt] = 'Null'
    
    # For every node not yet in the tree, begin a path starting at that node, moving to a neighbor at random until reach the tree
    for n in nodes:
        unit = n
        while not InTree[unit]:
            if unit in discoveredStack: # If a loop is formed, it's erased
                popped = discoveredStack.pop()
                while popped != unit:
                    del Next[popped]
                    popped = discoveredStack.pop()
                    
            discoveredStack.append(unit)
            both = list(nodes_set.intersection(set(neighborhoods[unit]))) # Only choose neighbors that are also in the two chosen districts
            Next[unit] = both[int(len(both)*r.random())]
            unit = Next[unit]
            
        # Once path reaches the tree, update tree membership of all nodes in path
        unit = n
        while not InTree[unit]:
            InTree[unit] = True
            unit = Next[unit]
            
    return Next


# Colors a given tree according to given node to split on
def colorTree(tree,nodes,split):
    
    # To color units, according to split. Initially color all nodes 'blue'
    labels = {}
    for n in nodes:
        labels[n] = 'blue'

    # Color split 'red', then color all paths that end at split 'red'    
    labels[split] = 'red'
    for n in nodes:
        skip = False
        unit = n
        while labels[unit] == 'blue':
            unit = tree[unit]
            if unit == 'Null':
                skip = True
                break

        if not skip:
            value = labels[unit]
            unit = n
            while labels[unit] == 'blue':
                labels[unit] = value
                unit = tree[unit]
                if unit == 'Null':
                    break
                    
    return labels



# Chooses edge to cut in spanning tree that bipartitions graph while satisfying constraints and improving objective (if there is one)
def bipartition(nodes,From,Too,passed):
    
    satisfied = False
    countTrees = 0
    fail = False

    while not satisfied:
        
        countTrees += 1
        
        root = nodes[int(len(nodes)*r.random())] # Choose root uniformly at random
        
        Next = spanningTree(nodes,root) # Obtain spanning tree
        viableNodes = [] # Stores nodes at which the tree can be cut to satisfy constraints and improve objective
    
        # Examine how tree splits for each node
        for n in nodes:
            
            if n != 'Null' and n != root: # Can't split tree at root, since Next[root] = 'Null'
                
                # Color tree according to splitNode
                splitNode = n
                colors = colorTree(Next,nodes,splitNode)
                
                # Gather nodes by color and determine new district populations, border nodes, and perimeters, etc.
                first,second,popFirst,popSecond,votesFirst,votesSecond,wFirst,wSecond,bFirst,bSecond,lFirst,lSecond = gather_pop_votes_demo(nodes,colors)


                # Use bool to keep track of whether objective/constraints are satisfied
                checks = True
                
                # Check pop objective/constraint, if applicable
                if (objective == 'pop') or ('pop' in constraints):
                    
                    # Calculate each district's percent difference from expected population
                    dPP = [abs(1-(dp)/(meanDistPop)) for dp in [popFirst,popSecond]]
                    dev = max(dPP)
                    
                    if dev > passed['pop']:
                        checks = False
                    elif objective == 'pop':
                        obj = dev
                        
                # If previous checks pass, check demo constraint, if applicable
                if checks:
                    if 'demo' in constraints:
                        
                        countB = 0
                        countL = 0
                        countW = 0
                        
                        if wFirst < 0.5:
                            if bFirst > max(wFirst,lFirst):
                                countB += 1
                            elif lFirst > max(wFirst,bFirst):
                                countL += 1
                            else:
                                countW += 1
                            
                        if wSecond < 0.5:
                            if bSecond > max(wSecond,lSecond):
                                countB += 1
                            elif lSecond > max(wSecond,bSecond):
                                countL += 1
                            else:
                                countW += 1
                                
                        if countB != passed['numB'] or countL != passed['numL'] or countW != passed['numW']:
                            checks = False
                        
                        
                # If previous checks pass, check perim objective/constraint, if applicable
                if checks:
                    if (objective == 'perim') or ('perim' in constraints):
                        
                        # Gather new border nodes and perimeters
                        bNFirst,bNSecond,perimFirst,perimSecond = gather_border_perim(first,second,colors)
                
                        newPerimeter = perimFirst + perimSecond
                        
                        if newPerimeter > passed['perim']:
                            checks = False
                        elif objective == 'perim':
                            obj = newPerimeter
                         
                        
                # If previous checks pass, check cmpttv objective/constraint, if applicable
                if checks:
                    if (objective == 'cmpttv') or ('cmpttv' in constraints):
                        
                        # Calculate margins of victory in both districts
                        if sum(votesFirst) != 0.0 and sum(votesSecond) != 0.0:
                            cmpttvFirst = abs(votesFirst[0]-votesFirst[1])/(votesFirst[0]+votesFirst[1])
                            cmpttvSecond = abs(votesSecond[0]-votesSecond[1])/(votesSecond[0]+votesSecond[1])
                        else:
                            cmpttvFirst = 1.5
                            cmpttvSecond = 1.5
                        
                        if 'cmpttv' in constraints:
                            
                            # Check whether competitive districts are maintained (<= 10% margin of victory)
                            if passed['cmpttv'][0] <= 0.1 and passed['cmpttv'][1] <= 0.1:

                                if cmpttvFirst > 0.1 or cmpttvSecond > 0.1:
                                    checks = False


                            elif passed['cmpttv'][0] <= 0.1 or passed['cmpttv'][1] <= 0.1:

                                if cmpttvFirst > 0.1 and cmpttvSecond > 0.1:
                                    checks = False
                                    
                        else:
                            
                            if max(cmpttvFirst,cmpttvSecond) > passed['cmpttv']:
                                checks = False
                            else:
                                obj = max(cmpttvFirst,cmpttvSecond)
                   
                
                # If previous checks pass, check mm objective/constraint, if applicable
                if checks:
                    if (objective == 'mm') or ('mm' in constraints):
                        
                        if sum(votesFirst) != 0.0 and sum(votesSecond) != 0.0:
                            newFD = [fd for fd in fracDem]
                            newFD[From] = votesFirst[0]/(sum(votesFirst))
                            newFD[Too] = votesSecond[0]/(sum(votesSecond))
                                    
                            mm = abs((np.median(newFD[1:])) - (sum(newFD[1:])/(numDistricts-1)))
                            
                        else:
                            mm = 1.5
                            
                        if mm > passed['mm']:
                            checks = False
                        elif objective == 'mm':
                            obj = mm
                
                
                # If previous checks pass, check eg objective/constraint, if applicable
                if checks:
                    if (objective == 'eg') or ('eg' in constraints):
                        
                        if sum(votesFirst) != 0.0 and sum(votesSecond) != 0.0:
                            newFDFrom = votesFirst[0]/(sum(votesFirst))
                            newFDToo = votesSecond[0]/(sum(votesSecond))
                            newTVFrom = sum(votesFirst)
                            newTVToo = sum(votesSecond)
                            eg,wFrom,wToo = update_EG_fast_it(From,Too,newFDFrom,newFDToo,newTVFrom,newTVToo)

                        else:
                            eg = 1.5
                            
                        if eg > passed['eg']:
                            checks = False
                        elif objective == 'eg':
                            obj = eg
                
                
                # If previous checks pass, check eg_shift objective/constraint, if applicable
                if checks:
                    if (objective == 'eg_shift') or ('eg_shift' in constraints):
                        
                        if sum(votesFirst) != 0.0 and sum(votesSecond) != 0.0:
                            newFDFrom = votesFirst[0]/(sum(votesFirst))
                            newFDToo = votesSecond[0]/(sum(votesSecond))
                            newTVFrom = sum(votesFirst)
                            newTVToo = sum(votesSecond)
                            seg,wFrom,wToo = update_ShiftedEG_fast_it(From,Too,newFDFrom,newFDToo,newTVFrom,newTVToo)

                        else:
                            seg = 1.5
                            
                            
                        if seg > passed['eg_shift']:
                            checks = False
                        elif objective == 'eg_shift':
                            obj = seg
                   
                
                # If previous checks pass, check whole constraint, if applicable
                if checks:
                    if 'whole' in constraints:
                        
                        unitsFirst = [n[0:5] for n in first]
                        unitsSecond = [n[0:5] for n in second]

                        numCountiesFirst = len(list(set(unitsFirst)))
                        numCountiesSecond = len(list(set(unitsSecond)))

                        if passed['whole'][0] == 1 and passed['whole'][1] == 1:
                            if numCountiesFirst != 1 or numCountiesSecond != 1:
                                checks = False
                                
                        elif passed['whole'][0] == 1 or passed['whole'][1] == 1:
                            if numCountiesFirst != 1 and numCountiesSecond != 1:
                                checks = False
                   
                          
                   
                # If previous checks pass, check pa objective/constraint, if applicable
                if checks:
                    if (objective == 'pa') or ('pa' in constraints):
                        
                        if sum(votesFirst) != 0.0 and sum(votesSecond) != 0.0:
                            newFD = [fd for fd in fracDem]
                            newFD[From] = votesFirst[0]/(sum(votesFirst))
                            newFD[Too] = votesSecond[0]/(sum(votesSecond))
                            pa = update_PA(newFD[1:])
                        else:
                            pa = 1.5
                
                
                        if pa > passed['pa']:
                            checks = False
                        elif objective == 'pa':
                            obj = pa
                
                
                # If all checks pass, record viable node and corresponding objective value
                if checks:
                    if objective == 'none':
                        viableNodes.append(splitNode)
                    else:
                        viableNodes.append([splitNode,obj])
                
                    
        if len(viableNodes) > 0: # If there exists a node at which to split the tree that satisfies constraints and improve objective (if there is one)
            
            # If no objective, choose split node uniformly at random. Otherwise, choose viable node that most improves objective
            if objective == 'none':
                splitNode = viableNodes[int(len(viableNodes)*r.random())]
            else:
                splitNode = viableNodes[0][0]
                minObj = viableNodes[0][1]
                for val in viableNodes:
                    if val[1] < minObj:
                        splitNode = val[0]
                        minObj = val[1]
                        
                
            satisfied = True
            
            # Color tree accordingly and get updates
            colors = colorTree(Next,nodes,splitNode)

            first,second,popFirst,popSecond,bNFirst,bNSecond,perimFirst,perimSecond,votesFirst,votesSecond,wFirst,wSecond,bFirst,bSecond,lFirst,lSecond = gather_pop_border_perim_votes_demo(nodes,colors)

                        
        if countTrees >= 25:
            satisfied = True
            fail = True
            
    #print('# trees = ',countTrees)        
    if fail:
        return [],[],-1,-1,[],[],-1,-1,[],[],-1,-1,-1,-1,-1,-1
    else:
        return first,second,popFirst,popSecond,bNFirst,bNSecond,perimFirst,perimSecond,votesFirst,votesSecond,wFirst,wSecond,bFirst,bSecond,lFirst,lSecond 
    

# Determines district membership, populations, border nodes, perimeters, votes, and % white, Black, Lat/Hisp after Recom move
def gather_pop_border_perim_votes_demo(nodes,colors):
    
    # Gather units by color, sum populations, etc.
    first = []
    second = []
    popFirst = 0
    popSecond = 0
    bNFirst = []
    bNSecond = []
    perimFirst = 0.0
    perimSecond = 0.0
    votesFirst = [0.0,0.0]
    votesSecond = [0.0,0.0]
    wFirst = 0
    wSecond = 0
    bFirst = 0
    bSecond = 0
    lFirst = 0
    lSecond = 0
    
    for n in nodes:
        if colors[n] == 'red':
            first.append(n)
            popFirst += pop[n]
            votesFirst[0] += votes[n][0]
            votesFirst[1] += votes[n][1]
            wFirst += popW[n]
            bFirst += popB[n]
            lFirst += popL[n]
                
            for nb in neighborhoods[n]:
                if nb in colors:
                    if colors[nb] == 'blue':
                        bNFirst.append(n)
                        perimFirst += edgesLength[(n,nb)]
                        
                else:
                    bNFirst.append(n)
                    perimFirst += edgesLength[(n,nb)]
            
        else:
            second.append(n)
            popSecond += pop[n]
            votesSecond[0] += votes[n][0]
            votesSecond[1] += votes[n][1]
            wSecond += popW[n]
            bSecond += popB[n]
            lSecond += popL[n]
                
            for nb in neighborhoods[n]:
                if nb in colors:
                    if colors[nb] == 'red':
                        bNSecond.append(n)
                        perimSecond += edgesLength[(n,nb)]

                else:
                    bNSecond.append(n)
                    perimSecond += edgesLength[(n,nb)]
                        
    
    if popFirst != 0:
        wFirst = float(wFirst)/popFirst
        bFirst = float(bFirst)/popFirst
        lFirst = float(lFirst)/popFirst
    else:
        wFirst = 0.0
        bFirst = 0.0
        lFirst = 0.0
        
    if popSecond != 0:
        wSecond = float(wSecond)/popSecond
        bSecond = float(bSecond)/popSecond
        lSecond = float(lSecond)/popSecond
    else:
        wSecond = 0.0
        bSecond = 0.0
        lSecond = 0.0
    
                        
    return first,second,popFirst,popSecond,bNFirst,bNSecond,perimFirst,perimSecond,votesFirst,votesSecond,wFirst,wSecond,bFirst,bSecond,lFirst,lSecond

    

# Determines district membership, populations, votes, and % white, Black, Lat/Hisp after Recom move
def gather_pop_votes_demo(nodes,colors):
    
    # Gather units by color, sum populations, etc.
    first = []
    second = []
    popFirst = 0
    popSecond = 0
    votesFirst = [0.0,0.0]
    votesSecond = [0.0,0.0]
    wFirst = 0
    wSecond = 0
    bFirst = 0
    bSecond = 0
    lFirst = 0
    lSecond = 0
    
    for n in nodes:
        if colors[n] == 'red':
            first.append(n)
            popFirst += pop[n]
            votesFirst[0] += votes[n][0]
            votesFirst[1] += votes[n][1]
            wFirst += popW[n]
            bFirst += popB[n]
            lFirst += popL[n]
                
        else:
            second.append(n)
            popSecond += pop[n]
            votesSecond[0] += votes[n][0]
            votesSecond[1] += votes[n][1]
            wSecond += popW[n]
            bSecond += popB[n]
            lSecond += popL[n]
                                 
    
    if popFirst != 0:
        wFirst = float(wFirst)/popFirst
        bFirst = float(bFirst)/popFirst
        lFirst = float(lFirst)/popFirst
    else:
        wFirst = 0.0
        bFirst = 0.0
        lFirst = 0.0
        
    if popSecond != 0:
        wSecond = float(wSecond)/popSecond
        bSecond = float(bSecond)/popSecond
        lSecond = float(lSecond)/popSecond
    else:
        wSecond = 0.0
        bSecond = 0.0
        lSecond = 0.0
    
                        
    return first,second,popFirst,popSecond,votesFirst,votesSecond,wFirst,wSecond,bFirst,bSecond,lFirst,lSecond


# Determines border nodes and perimeters after Recom move
def gather_border_perim(nodesFirst,nodesSecond,colors):
    
    # Identify border nodes and sum perimeters
    bNFirst = []
    bNSecond = []
    perimFirst = 0.0
    perimSecond = 0.0
    
    for n in nodesFirst:
        for nb in neighborhoods[n]:
            if nb in colors:
                if colors[nb] == 'blue':
                    bNFirst.append(n)
                    perimFirst += edgesLength[(n,nb)]

            else:
                bNFirst.append(n)
                perimFirst += edgesLength[(n,nb)]


    for n in nodesSecond:
        for nb in neighborhoods[n]:
            if nb in colors:
                if colors[nb] == 'red':
                    bNSecond.append(n)
                    perimSecond += edgesLength[(n,nb)]

            else:
                bNSecond.append(n)
                perimSecond += edgesLength[(n,nb)]
            
                        
    return bNFirst,bNSecond,perimFirst,perimSecond
            

# READ IN ALL PARAMETERS --------------------------------------------------------------------------------

# Record start time
start = time.time()

# Create boolean variable to report whether user input is valid
VALID = True

# Read in parameters
parameterFile = open('ReCom_PARAMETERS_MO.csv','r')
readerParam = csv.reader(parameterFile,delimiter=',')
parameters = [line for line in readerParam]

# Folder of district plan files
BigFolder = parameters[0][1]
print('Folder with maps: ',BigFolder)

if os.path.isdir(BigFolder):
    maps = [p for p in os.listdir(BigFolder+'/') if not p.startswith('.')]
    maps.sort()
else:
    VALID = False
    print('\n-----Folder does not exist-----\n')

if len(parameters[0]) > 2 and os.path.isdir(BigFolder):
    if parameters[0][2] != '':
        maps = []
        for val in parameters[0][2:]:
            if val == '':
                continue
            elif os.path.isfile(BigFolder+'/'+val):
                maps.append(val)
            else:
                VALID = False
                print('\n-----File does not exist-----\n')
                break

        print('Maps:\n',maps)
    

# Folder with state data
StateFolder = parameters[1][1]
print('Folder with state data: ',StateFolder)

if not os.path.isdir(StateFolder):
    VALID = False
    print('\n-----Folder does not exist-----\n')
else:
    files = [t for t in os.listdir(StateFolder+'/') if not t.startswith('.')]
    files.sort()


# Folder for output
OutputFolder = parameters[2][1]
print('Folder for output: ',OutputFolder)

if not os.path.isdir(OutputFolder):
    VALID = False
    print('\n-----Folder does not exist-----\n')


# Assign number of iterations
K = int(parameters[3][1])
itType = parameters[3][2]

if itType == 'it':
    print('Number of iterations: ',K)
else:
    print('Number of cycles: ',K)


# Determine objective function
objective = parameters[4][1]

ValidObjectives = ['none','pop','perim','eg','eg_shift','mm','pa','cmpttv']

if objective in ValidObjectives:
    print('Objective: ',objective)
else:
    VALID = False
    print('\n-----',objective,' is not a valid objective-----\n')
    
if objective == 'pop':
    print('Note: the program automatically uses single iterations for pop objective.')

    
# Determine constraints
constraints = parameters[5][1:]
constraints = [c for c in constraints if (c != '')]
print('Constraints:\n',constraints)

ValidConstraints = ['pop','perim','eg','eg_shift','mm','pa','cmpttv','demo','whole']

for c in constraints:
    if c not in ValidConstraints:
        VALID = False
        print('\n-----',c,' is not a valid constraint-----\n')


# Assign thresholds
popMinThresh = float(parameters[6][1])
perimAdd = float(parameters[7][1])
EGThresh = float(parameters[8][1])
EGShiftThresh = float(parameters[9][1])
MMThresh = float(parameters[10][1])
PAThresh = float(parameters[11][1])

if 'pop' in constraints:
    print('Population threshold: ',popMinThresh)
    
if 'perim' in constraints:
    print('Value added to current perimeter for perimeter threshold: ',perimAdd)
    
if 'eg' in constraints:
    print('EG threshold: ',EGThresh)
    
if 'eg_shift' in constraints:
    print('EGShift threshold: ',EGShiftThresh)
    
if 'mm' in constraints:
    print('MM threshold: ',MMThresh)
    
if 'pa' in constraints:
    print('PA threshold: ',PAThresh)
    

# Assign edge weights
edgeBonus = float(parameters[12][1])
edgePenalty = float(parameters[13][1])

print('Edge bonus: ',edgeBonus)
print('Edge penalty: ',edgePenalty)


# Determine if user wants objective convergence
converge = parameters[14][1]

if converge == 'yes' and (objective == 'pop' or objective == 'none'):
    converge = False
    print('Convergence is only an option for non-pop/non-none objectives, convergence choice ignored.')
elif converge == 'yes':
    converge = True
    print('Convergence')
    if len(parameters[14]) > 2:
        if parameters[14][2] != '':
            epsilon = float(parameters[14][2])
        else:
            VALID = False
            print('\n----- Did not provide convergence threshold -----\n')
    else:
        VALID = False
        print('\n----- Did not provide convergence threshold -----\n')
elif converge == 'no':
    converge = False
    print('No convergence')
else:
    VALID = False
    print('\n-----',converge,' is not a valid convergence option-----\n')

    

# Generate and report random seed
SEED = 0
while (SEED % 2 == 0):
    SEED = r.randrange(1000000001,9999999999)
#SEED = 1909110927
r.seed(SEED)
print('\nSeed = ', SEED)


# If user input is invalid, stop program
if not VALID:
    print('\nPROGRAM STOPPED DUE TO INVALID USER INPUT')
    
# If all user input is valid, run the program
else:

# STATE INFO -----------------------------------------------------------------------------------------------

    # Read in state info
    adj = readData(StateFolder+'/'+files[0])       # Unit adjacency
    unitInfo = readData(StateFolder+'/'+files[1])  # Unit info
    if len(files) > 2:
        forbidden = readData(StateFolder+'/'+files[2]) # Forbidden unit adjacency
    else:
        forbidden = []


    #forbidden = []


    # Gather dictionaries of unit info
    ids = [] # GEOIDs
    pop = {} # total population
    votes = {} # dem/rep votes
    popW = {} # white population
    popB = {} # Black/African American population
    popL = {} # Lat/Hisp population
    
    for u in unitInfo:
        ids.append(u[0])
        pop[u[0]] = int(u[1])
        votes[u[0]] = [float(u[2]),float(u[3])]
        popW[u[0]] = int(u[4])
        popB[u[0]] = int(u[5])
        popL[u[0]] = int(u[6])

            
    print('Number of units: ',len(ids))

    # Insert dummy node 0
    ids.insert(0,'0')
    pop['0'] = 0
    votes['0'] = [0.0,0.0]
    popW['0'] = 0
    popB['0'] = 0
    popL['0'] = 0

    print('State Info')
    

# ADJACENCY ------------------------------------------------------------------------------------------------

    # Make cleaned adjacency dictionary
    edgesLength = {}

    # Populate adjacency matrix with length of shared segment for normal adjacency and skip aug adjacency

    # Normal adjacency
    for e in adj:
        if e[2] != '0' and e[2] != '0.0' and e[0] != 'other' and e[1] != 'other':
            if [e[0][0:5],e[1][0:5]] not in forbidden and [e[1][0:5],e[0][0:5]] not in forbidden and [e[0],e[1]] not in forbidden and [e[1],e[0]] not in forbidden:
                edgesLength[(e[0],e[1])] = float(e[2])
                edgesLength[(e[1],e[0])] = float(e[2]) # Record both so don't have to check each time that the edge pair is in the dictionary


    # BONUS - decrease length of shared segment if units are from two different counties (to encourage districts to follow county lines)
    # PENALTY - increase length of shared segment if units are from the same county (to discourage districts from straddling county lines)
    # Only affects county-tract or tract-tract adjacency (i.e., not county-county)
    for e in edgesLength:
        if len(e[0]) > 5 or len(e[1]) > 5:
        #if len(e[0]) > 5 and len(e[1]) > 5:
            if e[0][0:5] != e[1][0:5]:
                edgesLength[e] = edgeBonus*edgesLength[e]
            else:
                edgesLength[e] = edgePenalty*edgesLength[e]


    print('Adjacency')


# NEIGHBORHOODS -----------------------------------------------------------------------------------

    # Gather a list of neighbors for every unit
    neighborhoods = {}

    for i in ids:
        neighborhoods[i] = []

    for pair in edgesLength:
        neighborhoods[pair[0]].append(pair[1]) # Now that an edge is in edgesLength as both (a,b) and (b,a), don't need double neighborhood statements
        #neighborhoods[pair[1]].append(pair[0])


    print('Neighborhoods')


# ITERATE THROUGH EACH DISTRICT PLAN --------------------------------------------------------------

    for plan in maps:

        # Record start time for a particular map
        startMap = time.time()

        print('Given district map: ',plan)

# DISTRICTS ---------------------------------------------------------------------------------------

        # Read in plan data, clean data
        data_raw = readData(BigFolder+'/'+plan)
        data = {}
        numDistricts = 0
        for d in data_raw:
            data[d[0]] = int(d[1])
            if int(d[1]) > numDistricts:
                numDistricts = int(d[1])

        # Insert dummy node 0 in dummy district 0 (to represent outside the district)
        data['0'] = 0
        numDistricts += 1

        # Group units by part
        partition = [[] for p in range(0,numDistricts)]
        
        for i in ids:
            partition[data[i]].append(i)
        
        # Count number of counties spanned by each district
        if 'whole' in constraints:
        
            numCounties = []
            for p in partition:
                temp = []
                for i in p:
                    temp.append(i[0:5])

                temp = list(set(temp))
                numCounties.append(len(temp))

        #     for i in range(0,len(numCounties)):
        #         print('District: ',i,' Counties: ',numCounties[i])


# BORDER NODES ------------------------------------------------------------------------------------

        # Gather units on the border of each district
        borderNodes = {}
        for i in range(1,numDistricts):
            borderNodes[i] = []

        for node in ids:
            dist = data[node]
            for nb in neighborhoods[node]:
                otherDist = data[nb]
                if otherDist != dist and otherDist != 0 and dist != 0: # Don't want nodes to move to dummy district 0, DON'T need to count units multiple times
                    borderNodes[dist].append(node)
                    break # Exit neighborhood loop if node is already identified as on the border


# POPULATION --------------------------------------------------------------------------------------

        # Calculate population of each district

        distPop = [0 for i in range(0,numDistricts)]
        
        for i in ids:
            distPop[data[i]] += pop[i]

        meanDistPop = sum(distPop)/(numDistricts-1) #numDistricts-1 bc don't want to include dummy district
        distPop[0] = meanDistPop # Change dummy districts population to the expected

        # Calculate each district's percent difference from expected population
        distPopPercent = [abs(1-(dp)/(meanDistPop)) for dp in distPop]

        popInitial = max(distPopPercent)
    #     print('Pop dev for each district:\n',distPopPercent)
        print('Initial max population deviation: ',popInitial)

        valuesPop = [popInitial]
        popDev = popInitial

# COMPACTNESS -------------------------------------------------------------------------------------

        # Calculate district perimeters
        distPerimeter = [0.0 for i in range(0,numDistricts)]
        distPerimeter[0] = 0.0 # Dummy district perimeter

        for i in range(1,numDistricts):
            for b in borderNodes[i]:
                for n in neighborhoods[b]:
                    if data[n] != data[b]:
                        distPerimeter[i] += edgesLength[(n,b)]

        
        perimInitial = sum(distPerimeter)
        print('Initial total perimeter: ',perimInitial)
        valuesPerim = [perimInitial]
        totalPerim = perimInitial
        if objective == 'perim':
            objValue = totalPerim


# VOTES -------------------------------------------------------------------------------------------

        # Calculate fraction of dem, and number of dem/rep in every district
        fracDem = [0.0 for i in range(0,numDistricts)]
        numDem = [0.0 for i in range(0,numDistricts)]
        numRep = [0.0 for i in range(0,numDistricts)]
        
        for d in data:
            numDem[data[d]] += votes[d][0]
            numRep[data[d]] += votes[d][1]
                    
        for i in range(1,numDistricts):
            fracDem[i] = numDem[i]/(numDem[i]+numRep[i])

        totalVotesByDistrict = [numDem[i]+numRep[i] for i in range(0,len(numDem))]
        totalVotes = sum(totalVotesByDistrict)


# FAIRNESS METRICS --------------------------------------------------------------------------------

        # Calculate initial Efficiency Gap value
        if (objective == 'eg') or ('eg' in constraints):
            
            wastedNominal = initialize_wasted(numDem[1:],numRep[1:])
            EGInitial = update_EG_fast()
            print('Initial EG: ',EGInitial)
            valuesEGNominal = [EGInitial]
            EGNominal = EGInitial
            
            if objective == 'eg':
                objValue = EGNominal

        
        # Calculate initial Shifted Efficiency Gap value
        if (objective == 'eg_shift') or ('eg_shift' in constraints):

            wastedShift = initialize_wastedShift(fracDem[1:],totalVotesByDistrict[1:])
            EGShiftInitial = update_ShiftedEG_fast()
            print('Initial Max EG Shift: ',EGShiftInitial)
            valuesEGShift = [EGShiftInitial]
            EGShift = EGShiftInitial
            
            if objective == 'eg_shift':
                objValue = EGShift

            # Slower calculation of EGShift
        #     EGShiftInitial = update_ShiftedEG(fracDem[1:],totalVotesByDistrict[1:])
        #     print('Initial Max EG Shift OLD: ',EGShiftInitial)
        #     valuesEGShift = [EGShiftInitial]
        #     EGShift = EGShiftInitial

        
        # Calculate initial Median-Mean value
        if (objective == 'mm') or ('mm' in constraints):
            
            MMInitial = abs((np.median(fracDem[1:])) - (sum(fracDem[1:])/(numDistricts-1))) # abs(Median(FD) - Mean(FD))
            print('Initial MM: ',MMInitial)
            valuesMM = [MMInitial]
            MM = MMInitial
            
            if objective == 'mm':
                objValue = MM
        
        
        # Calculate initial Partisan Asymmetry value
        if (objective == 'pa') or ('pa' in constraints):
        
            PAInitial = update_PA(fracDem[1:])
            print('Initial PA: ',PAInitial)
            valuesPA = [PAInitial]
            PA = PAInitial
            
            if objective == 'pa':
                objValue = PA

            
        # Calculate initial competitiveness
        if (objective == 'cmpttv') or ('cmpttv' in constraints):
        
            compFrac = [abs(2*fd - 1) for fd in fracDem[1:]]
            compFrac.insert(0,0.0)
            CmpttvInitial = sum(compFrac)/(len(compFrac)-1)
            print('Initial avg margin (Cmpttv): ',CmpttvInitial)
            print('Initial max margin: ',max(compFrac))
            
            countCmpttv = 0
            for cp in compFrac[1:]:
                if cp <= 0.1:
                    countCmpttv += 1

            print('Initial number within 10% margin: ',countCmpttv)
            
            valuesCmpttv = [CmpttvInitial]
            Cmpttv = CmpttvInitial
            
            if objective == 'cmpttv':
                objValue = Cmpttv


# MAJORITY-MINORITY -------------------------------------------------------------------------------

        # Calculate the fraction of district pop that is white/Black/Lat
        fracW = [0.0 for i in range(0,numDistricts)]
        fracB = [0.0 for i in range(0,numDistricts)]
        fracL = [0.0 for i in range(0,numDistricts)]

        for d in data:
            fracW[data[d]] += popW[d]
            fracB[data[d]] += popB[d]
            fracL[data[d]] += popL[d]

        for i in range(0,numDistricts):
            fracW[i] = float(fracW[i])/distPop[i]
            fracB[i] = float(fracB[i])/distPop[i]
            fracL[i] = float(fracL[i])/distPop[i]

#         for i in range(0,numDistricts):
#             print(fracW[i])

        numPlurB = 0
        numPlurL = 0
        numPlurW = 0

        for i in range(1,numDistricts):
            if fracW[i] < 0.5:
                if fracB[i] > fracW[i] and fracB[i] > fracL[i]:
                    numPlurB += 1
                elif fracL[i] > fracW[i] and fracL[i] > fracB[i]:
                    numPlurL += 1
                else:
                    numPlurW += 1
                    

        if 'demo' in constraints:
            print('Initial # plurality-Black MM districts: ',numPlurB)
            print('Initial # plurality-Lat/Hisp MM districts: ',numPlurL)
            print('Initial # plurality-white MM districts: ',numPlurW)


# ITERATIONS --------------------------------------------------------------------------------------
    
        # For pop objective, alternate between choosing random district and district with max pop deviation
        # This is much quicker for achieving pop balance
        
        if objective == 'pop':

            for k in range(1,K+1):

                print(k) # Give user idea of algorithm progress
                
    #             if (k % 25) == 0:
    #                 print(k) # Give user idea of algorithm progress

                if (k % 2) == 0:
                    # Choose district with largest pop dev
                    distFrom = distPopPercent.index(popDev)
                else:
                    # Randomly choose a district and a unit on its border
                    distFrom = int((numDistricts-1)*r.random())+1


                # Randomly choose unit on border of distFrom
                index = int(len(borderNodes[distFrom])*r.random())
                node = borderNodes[distFrom][index]

                # Determine other district that chosen unit is adjacent to 
                candidates = []

                for nb in neighborhoods[node]:
                    otherDist = data[nb]
                    if nb != '0' and otherDist != distFrom and otherDist not in candidates:
                        candidates.append(otherDist)

                index = int(len(candidates)*r.random())
                distToo = candidates[index]

                #print('distFrom: ',distFrom,' distToo: ',distToo)

                # Combine units from the two adjacent districts
                merged = partition[distFrom] + partition[distToo]

                # Choose root of spanning tree uniformly at random
                #root = merged[int(len(merged)*r.random())]
                #print('root: ',root)
                
                # Create dictionary of values to pass bipartition function
                valuesToPass = {}
                
                # We know the objective is pop
                valuesToPass['pop'] = max(distPopPercent[distFrom],distPopPercent[distToo])
                
                # Prep values to pass to bipartition function
                if 'perim' in constraints:
                    valuesToPass['perim'] = perimInitial + perimAdd - (totalPerim - distPerimeter[distFrom] - distPerimeter[distToo])
                if 'eg' in constraints:
                    valuesToPass['eg'] = EGThresh
                if 'eg_shift' in constraints:
                    valuesToPass['eg_shift'] = EGShiftThresh
                if 'mm' in constraints:
                    valuesToPass['mm'] = MMThresh
                if 'pa' in constraints:
                    valuesToPass['pa'] = PAThresh
                if 'cmpttv' in constraints:
                    valuesToPass['cmpttv'] = [compFrac[distFrom],compFrac[distToo]]
                if 'whole' in constraints:
                    valuesToPass['whole'] = [numCounties[distFrom],numCounties[distToo]]
                if 'demo' in constraints:
                    numW = 0
                    numB = 0
                    numL = 0
                    
                    if fracW[distFrom] < 0.5:
                        if fracB[distFrom] > max(fracW[distFrom],fracL[distFrom]):
                            numB += 1
                        elif fracL[distFrom] > max(fracW[distFrom],fracB[distFrom]):
                            numL += 1
                        else:
                            numW += 1
                            
                    if fracW[distToo] < 0.5:
                        if fracB[distToo] > max(fracW[distToo],fracL[distToo]):
                            numB += 1
                        elif fracL[distToo] > max(fracW[distToo],fracB[distToo]):
                            numL += 1
                        else:
                            numW += 1
                            
                    valuesToPass['numW'] = numW
                    valuesToPass['numB'] = numB
                    valuesToPass['numL'] = numL
                    
                    
                # Call bipartition function
                partFrom,partToo,newPopFrom,newPopToo,newBNFrom,newBNToo,newPerimFrom,newPerimToo,newVotesFrom,newVotesToo,newWFrom,newWToo,newBFrom,newBToo,newLFrom,newLToo = bipartition(merged,distFrom,distToo,valuesToPass)

                # If a change has actually been made (i.e., move is successful)
                if newPopFrom >= 0:

                    # Update partition
                    partition[distFrom] = partFrom.copy()
                    partition[distToo] = partToo.copy()

                    # Update district membership (and possibly count number of counties spanned by the two districts)
                    if 'whole' in constraints:
                        
                        countiesFrom = []
                        countiesToo = []
                        
                        for unit in partition[distFrom]:
                            data[unit] = distFrom
                            countiesFrom.append(unit[0:5])

                        for unit in partition[distToo]:
                            data[unit] = distToo
                            countiesToo.append(unit[0:5])

                        numCounties[distFrom] = len(list(set(countiesFrom)))
                        numCounties[distToo] = len(list(set(countiesToo)))
                        
                    else:
                    
                        for unit in partition[distFrom]:
                            data[unit] = distFrom
                        for unit in partition[distToo]:
                            data[unit] = distToo

                    # Update district populations
                    distPop[distFrom] = newPopFrom
                    distPop[distToo] = newPopToo

                    # Update district percent difference from expected population
                    distPopPercent[distFrom] = abs(1-(distPop[distFrom])/(meanDistPop))
                    distPopPercent[distToo] = abs(1-(distPop[distToo])/(meanDistPop))

                    # Record pop objective value
                    popDev = max(distPopPercent)
                    valuesPop.append(popDev)

                    # Update borderNodes
                    borderNodes[distFrom] = list(set(newBNFrom.copy()))
                    borderNodes[distToo] = list(set(newBNToo.copy()))

                    # Update compactness
                    distPerimeter[distFrom] = newPerimFrom
                    distPerimeter[distToo] = newPerimToo

                    totalPerim = sum(distPerimeter)
                    valuesPerim.append(totalPerim)

                    # Update votes
                    fracDem[distFrom] = newVotesFrom[0]/(sum(newVotesFrom))
                    fracDem[distToo] = newVotesToo[0]/(sum(newVotesToo))

                    numDem[distFrom] = newVotesFrom[0]
                    numDem[distToo] = newVotesToo[0]

                    numRep[distFrom] = newVotesFrom[1]
                    numRep[distToo] = newVotesToo[1]

                    totalVotesByDistrict[distFrom] = numDem[distFrom]+numRep[distFrom]
                    totalVotesByDistrict[distToo] = numDem[distToo]+numRep[distToo]
                    
                    # Update % white, Black, Lat/Hisp
                    fracW[distFrom] = newWFrom
                    fracB[distFrom] = newBFrom
                    fracL[distFrom] = newLFrom

                    fracW[distToo] = newWToo
                    fracB[distToo] = newBToo
                    fracL[distToo] = newLToo



        # For objectives other than pop, single iterations
        elif itType == 'it':

            finished = False
            k = 0
            numIt = 0

            while not finished:

                # Give user idea of algorithm progress
                k += 1
                print(k)

                # Check convergence, or complete a pre-determined number of iterations
                if converge and k > 1:
                    if (objValue_Old - objValue < epsilon):
                        numIt += 1
                        print('\nobjValue_Old: ',objValue_Old)
                        print('objValue: ',objValue)
                    else:
                        numIt = 0

                    if numIt >= 10:
                        finished = True
                        continue

                elif (not converge) and (k >= K):
                    finished = True
                    
                
                if converge:
                    objValue_Old = objValue
                    
                    
                    
                # Randomly choose a district
                distFrom = int((numDistricts-1)*r.random())+1
                
                # Randomly choose unit on border of distFrom
                index = int(len(borderNodes[distFrom])*r.random())
                node = borderNodes[distFrom][index]

                # Determine other district that chosen unit is adjacent to 
                candidates = []

                for nb in neighborhoods[node]:
                    otherDist = data[nb]
                    if nb != '0' and otherDist != distFrom and otherDist not in candidates:
                        candidates.append(otherDist)

                index = int(len(candidates)*r.random())
                distToo = candidates[index]

                #print('distFrom: ',distFrom,' distToo: ',distToo)

                # Combine units from the two adjacent districts
                merged = partition[distFrom] + partition[distToo]

                # Choose root of spanning tree uniformly at random
                #root = merged[int(len(merged)*r.random())]
                #print('root: ',root)

                # Create dictionary of values to pass bipartition function
                valuesToPass = {}

                # Prep values to pass to bipartition function

                # We know pop isn't the objective
                if 'pop' in constraints:
                    valuesToPass['pop'] = popMinThresh

                # Prep values to pass to bipartition function
                if objective == 'perim':
                    valuesToPass['perim'] = distPerimeter[distFrom]+distPerimeter[distToo]
                elif 'perim' in constraints:
                    valuesToPass['perim'] = perimInitial + perimAdd - (totalPerim - distPerimeter[distFrom] - distPerimeter[distToo])

                if objective == 'eg':
                    valuesToPass['eg'] = EGNominal
                elif 'eg' in constraints:
                    valuesToPass['eg'] = EGThresh

                if objective == 'eg_shift':
                    valuesToPass['eg_shift'] = EGShift
                elif 'eg_shift' in constraints:
                    valuesToPass['eg_shift'] = EGShiftThresh

                if objective == 'mm':
                    valuesToPass['mm'] = MM
                elif 'mm' in constraints:
                    valuesToPass['mm'] = MMThresh

                if objective == 'pa':
                    valuesToPass['pa'] = PA
                elif 'pa' in constraints:
                    valuesToPass['pa'] = PAThresh

                if objective == 'cmpttv':
                    valuesToPass['cmpttv'] = max(compFrac[distFrom],compFrac[distToo])
                elif 'cmpttv' in constraints:
                    valuesToPass['cmpttv'] = [compFrac[distFrom],compFrac[distToo]]

                if 'whole' in constraints:
                    valuesToPass['whole'] = [numCounties[distFrom],numCounties[distToo]]

                if 'demo' in constraints:
                    numW = 0
                    numB = 0
                    numL = 0

                    if fracW[distFrom] < 0.5:
                        if fracW[distFrom] > max(fracB[distFrom],fracL[distFrom]):
                            numW += 1
                        elif fracB[distFrom] > max(fracW[distFrom],fracL[distFrom]):
                            numB += 1
                        elif fracL[distFrom] > max(fracW[distFrom],fracB[distFrom]):
                            numL += 1

                    if fracW[distToo] < 0.5:
                        if fracW[distToo] > max(fracB[distToo],fracL[distToo]):
                            numW += 1
                        elif fracB[distToo] > max(fracW[distToo],fracL[distToo]):
                            numB += 1
                        elif fracL[distToo] > max(fracW[distToo],fracB[distToo]):
                            numL += 1

                    valuesToPass['numW'] = numW
                    valuesToPass['numB'] = numB
                    valuesToPass['numL'] = numL


                # Call bipartition function
                partFrom,partToo,newPopFrom,newPopToo,newBNFrom,newBNToo,newPerimFrom,newPerimToo,newVotesFrom,newVotesToo,newWFrom,newWToo,newBFrom,newBToo,newLFrom,newLToo = bipartition(merged,distFrom,distToo,valuesToPass)

                # If a change has actually been made (i.e., move is successful)
                if newPopFrom >= 0:

                    # Update partition
                    partition[distFrom] = partFrom.copy()
                    partition[distToo] = partToo.copy()

                    # Update district membership (and possibly count number of counties spanned by the two districts)
                    if 'whole' in constraints:

                        countiesFrom = []
                        countiesToo = []

                        for unit in partition[distFrom]:
                            data[unit] = distFrom
                            countiesFrom.append(unit[0:5])

                        for unit in partition[distToo]:
                            data[unit] = distToo
                            countiesToo.append(unit[0:5])

                        numCounties[distFrom] = len(list(set(countiesFrom)))
                        numCounties[distToo] = len(list(set(countiesToo)))

                    else:

                        for unit in partition[distFrom]:
                            data[unit] = distFrom
                        for unit in partition[distToo]:
                            data[unit] = distToo

                    # Update district populations
                    distPop[distFrom] = newPopFrom
                    distPop[distToo] = newPopToo

                    # Update district percent difference from expected population
                    distPopPercent[distFrom] = abs(1-(distPop[distFrom])/(meanDistPop))
                    distPopPercent[distToo] = abs(1-(distPop[distToo])/(meanDistPop))

                    # Record pop objective value
                    popDev = max(distPopPercent)
                    valuesPop.append(popDev)

                    # Update borderNodes
                    borderNodes[distFrom] = list(set(newBNFrom.copy()))
                    borderNodes[distToo] = list(set(newBNToo.copy()))

                    # Update compactness
                    distPerimeter[distFrom] = newPerimFrom
                    distPerimeter[distToo] = newPerimToo

                    totalPerim = sum(distPerimeter)
                    valuesPerim.append(totalPerim)

                    if objective == 'perim':
                        objValue = totalPerim

                    # Update votes
                    fracDem[distFrom] = newVotesFrom[0]/(sum(newVotesFrom))
                    fracDem[distToo] = newVotesToo[0]/(sum(newVotesToo))

                    numDem[distFrom] = newVotesFrom[0]
                    numDem[distToo] = newVotesToo[0]

                    numRep[distFrom] = newVotesFrom[1]
                    numRep[distToo] = newVotesToo[1]

                    totalVotesByDistrict[distFrom] = numDem[distFrom]+numRep[distFrom]
                    totalVotesByDistrict[distToo] = numDem[distToo]+numRep[distToo]

                    # Update EG
                    if (objective == 'eg') or ('eg' in constraints):

                        EGNominal,newWastedFrom,newWastedToo = update_EG_fast_it(distFrom,distToo,fracDem[distFrom],fracDem[distToo],totalVotesByDistrict[distFrom],totalVotesByDistrict[distToo])
                        wastedNominal[distFrom] = newWastedFrom
                        wastedNominal[distToo] = newWastedToo

                        valuesEGNominal.append(EGNominal)

                        if objective == 'eg':
                            objValue = EGNominal

                    # Updated Shifted EG
                    if (objective == 'eg_shift') or ('eg_shift' in constraints):

                        EGShift,newWastedFrom,newWastedToo = update_ShiftedEG_fast_it(distFrom,distToo,fracDem[distFrom],fracDem[distToo],totalVotesByDistrict[distFrom],totalVotesByDistrict[distToo])
                        for s in [-0.05,-0.04,-0.03,-0.02,-0.01,0.0,0.01,0.02,0.03,0.04,0.05]:
                            wastedShift[(distFrom,s)] = newWastedFrom[s]
                            wastedShift[(distToo,s)] = newWastedToo[s]

                        valuesEGShift.append(EGShift)

                        if objective == 'eg_shift':
                            objValue = EGShift

                    # Update MM
                    if (objective == 'mm') or ('mm' in constraints):

                        MM = abs((np.median(fracDem[1:])) - (sum(fracDem[1:])/(numDistricts-1))) 
                        valuesMM.append(MM)

                        if objective == 'mm':
                            objValue = MM

                    # Update PA
                    if (objective == 'pa') or ('pa' in constraints):

                        PA = update_PA(fracDem[1:])
                        valuesPA.append(PA)

                        if objective == 'PA':
                            objValue = PA

                    # Update Cmpttv
                    if (objective == 'cmpttv') or ('cmpttv' in constraints):

                        compFrac[distFrom] = abs(2*fracDem[distFrom] - 1)
                        compFrac[distToo] = abs(2*fracDem[distToo] - 1)
                        Cmpttv = sum(compFrac)/(len(compFrac)-1)
                        valuesCmpttv.append(Cmpttv)

                        if objective == 'cmpttv':
                            objValue = Cmpttv


                    # Update % white, Black, Lat/Hisp
                    fracW[distFrom] = newWFrom
                    fracB[distFrom] = newBFrom
                    fracL[distFrom] = newLFrom

                    fracW[distToo] = newWToo
                    fracB[distToo] = newBToo
                    fracL[distToo] = newLToo


                    
                    
        # For objectives other than pop, cycles through the districts
        else:    

            distPermutation = [i for i in range(1,numDistricts)]

            finished = False
            k = 0
            numCycles = 0
            
            while not finished:

                # Give user idea of algorithm progress
                k += 1
                print(k)

                # Check convergence, or complete a pre-determined number of cycles
                if converge and k > 1:
                    if (objValue_Old - objValue < epsilon):
                        numCycles += 1
                        print('\nobjValue_Old: ',objValue_Old)
                        print('objValue: ',objValue)
                    else:
                        numCycles = 0

                    if numCycles >= 10:
                        finished = True
                        continue

                elif (not converge) and (k >= K):
                    finished = True
                    
                
                if converge:
                    objValue_Old = objValue
                

                # For objectives other than pop, cycle through a random permutation of the districts
                r.shuffle(distPermutation)

                for distFrom in distPermutation:

                    # Randomly choose unit on border of distFrom
                    index = int(len(borderNodes[distFrom])*r.random())
                    node = borderNodes[distFrom][index]

                    # Determine other district that chosen unit is adjacent to 
                    candidates = []

                    for nb in neighborhoods[node]:
                        otherDist = data[nb]
                        if nb != '0' and otherDist != distFrom and otherDist not in candidates:
                            candidates.append(otherDist)

                    index = int(len(candidates)*r.random())
                    distToo = candidates[index]

                    #print('distFrom: ',distFrom,' distToo: ',distToo)

                    # Combine units from the two adjacent districts
                    merged = partition[distFrom] + partition[distToo]

                    # Choose root of spanning tree uniformly at random
                    #root = merged[int(len(merged)*r.random())]
                    #print('root: ',root)
                    
                    # Create dictionary of values to pass bipartition function
                    valuesToPass = {}
                    
                    # Prep values to pass to bipartition function

                    # We know pop isn't the objective
                    if 'pop' in constraints:
                        valuesToPass['pop'] = popMinThresh

                    # Prep values to pass to bipartition function
                    if objective == 'perim':
                        valuesToPass['perim'] = distPerimeter[distFrom]+distPerimeter[distToo]
                    elif 'perim' in constraints:
                        valuesToPass['perim'] = perimInitial + perimAdd - (totalPerim - distPerimeter[distFrom] - distPerimeter[distToo])
                    
                    if objective == 'eg':
                        valuesToPass['eg'] = EGNominal
                    elif 'eg' in constraints:
                        valuesToPass['eg'] = EGThresh
                        
                    if objective == 'eg_shift':
                        valuesToPass['eg_shift'] = EGShift
                    elif 'eg_shift' in constraints:
                        valuesToPass['eg_shift'] = EGShiftThresh
                        
                    if objective == 'mm':
                        valuesToPass['mm'] = MM
                    elif 'mm' in constraints:
                        valuesToPass['mm'] = MMThresh
                        
                    if objective == 'pa':
                        valuesToPass['pa'] = PA
                    elif 'pa' in constraints:
                        valuesToPass['pa'] = PAThresh
                        
                    if objective == 'cmpttv':
                        valuesToPass['cmpttv'] = max(compFrac[distFrom],compFrac[distToo])
                    elif 'cmpttv' in constraints:
                        valuesToPass['cmpttv'] = [compFrac[distFrom],compFrac[distToo]]
                        
                    if 'whole' in constraints:
                        valuesToPass['whole'] = [numCounties[distFrom],numCounties[distToo]]
                        
                    if 'demo' in constraints:
                        numW = 0
                        numB = 0
                        numL = 0

                        if fracW[distFrom] < 0.5:
                            if fracW[distFrom] > max(fracB[distFrom],fracL[distFrom]):
                                numW += 1
                            elif fracB[distFrom] > max(fracW[distFrom],fracL[distFrom]):
                                numB += 1
                            elif fracL[distFrom] > max(fracW[distFrom],fracB[distFrom]):
                                numL += 1
                                
                        if fracW[distToo] < 0.5:
                            if fracW[distToo] > max(fracB[distToo],fracL[distToo]):
                                numW += 1
                            elif fracB[distToo] > max(fracW[distToo],fracL[distToo]):
                                numB += 1
                            elif fracL[distToo] > max(fracW[distToo],fracB[distToo]):
                                numL += 1

                        valuesToPass['numW'] = numW
                        valuesToPass['numB'] = numB
                        valuesToPass['numL'] = numL
                        
                        
                    # Call bipartition function
                    partFrom,partToo,newPopFrom,newPopToo,newBNFrom,newBNToo,newPerimFrom,newPerimToo,newVotesFrom,newVotesToo,newWFrom,newWToo,newBFrom,newBToo,newLFrom,newLToo = bipartition(merged,distFrom,distToo,valuesToPass)

                    # If a change has actually been made (i.e., move is successful)
                    if newPopFrom >= 0:

                        # Update partition
                        partition[distFrom] = partFrom.copy()
                        partition[distToo] = partToo.copy()

                        # Update district membership (and possibly count number of counties spanned by the two districts)
                        if 'whole' in constraints:

                            countiesFrom = []
                            countiesToo = []

                            for unit in partition[distFrom]:
                                data[unit] = distFrom
                                countiesFrom.append(unit[0:5])

                            for unit in partition[distToo]:
                                data[unit] = distToo
                                countiesToo.append(unit[0:5])

                            numCounties[distFrom] = len(list(set(countiesFrom)))
                            numCounties[distToo] = len(list(set(countiesToo)))

                        else:

                            for unit in partition[distFrom]:
                                data[unit] = distFrom
                            for unit in partition[distToo]:
                                data[unit] = distToo

                        # Update district populations
                        distPop[distFrom] = newPopFrom
                        distPop[distToo] = newPopToo

                        # Update district percent difference from expected population
                        distPopPercent[distFrom] = abs(1-(distPop[distFrom])/(meanDistPop))
                        distPopPercent[distToo] = abs(1-(distPop[distToo])/(meanDistPop))

                        # Record pop objective value
                        popDev = max(distPopPercent)
                        valuesPop.append(popDev)

                        # Update borderNodes
                        borderNodes[distFrom] = list(set(newBNFrom.copy()))
                        borderNodes[distToo] = list(set(newBNToo.copy()))

                        # Update compactness
                        distPerimeter[distFrom] = newPerimFrom
                        distPerimeter[distToo] = newPerimToo

                        totalPerim = sum(distPerimeter)
                        valuesPerim.append(totalPerim)
                        
                        if objective == 'perim':
                            objValue = totalPerim

                        # Update votes
                        fracDem[distFrom] = newVotesFrom[0]/(sum(newVotesFrom))
                        fracDem[distToo] = newVotesToo[0]/(sum(newVotesToo))

                        numDem[distFrom] = newVotesFrom[0]
                        numDem[distToo] = newVotesToo[0]

                        numRep[distFrom] = newVotesFrom[1]
                        numRep[distToo] = newVotesToo[1]

                        totalVotesByDistrict[distFrom] = numDem[distFrom]+numRep[distFrom]
                        totalVotesByDistrict[distToo] = numDem[distToo]+numRep[distToo]

                        # Update EG
                        if (objective == 'eg') or ('eg' in constraints):
                            
                            EGNominal,newWastedFrom,newWastedToo = update_EG_fast_it(distFrom,distToo,fracDem[distFrom],fracDem[distToo],totalVotesByDistrict[distFrom],totalVotesByDistrict[distToo])
                            wastedNominal[distFrom] = newWastedFrom
                            wastedNominal[distToo] = newWastedToo

                            valuesEGNominal.append(EGNominal)
                            
                            if objective == 'eg':
                                objValue = EGNominal
                            
                        # Updated Shifted EG
                        if (objective == 'eg_shift') or ('eg_shift' in constraints):

                            EGShift,newWastedFrom,newWastedToo = update_ShiftedEG_fast_it(distFrom,distToo,fracDem[distFrom],fracDem[distToo],totalVotesByDistrict[distFrom],totalVotesByDistrict[distToo])
                            for s in [-0.05,-0.04,-0.03,-0.02,-0.01,0.0,0.01,0.02,0.03,0.04,0.05]:
                                wastedShift[(distFrom,s)] = newWastedFrom[s]
                                wastedShift[(distToo,s)] = newWastedToo[s]

                            valuesEGShift.append(EGShift)
                            
                            if objective == 'eg_shift':
                                objValue = EGShift

                        # Update MM
                        if (objective == 'mm') or ('mm' in constraints):
                            
                            MM = abs((np.median(fracDem[1:])) - (sum(fracDem[1:])/(numDistricts-1))) 
                            valuesMM.append(MM)
                            
                            if objective == 'mm':
                                objValue = MM
            
                        # Update PA
                        if (objective == 'pa') or ('pa' in constraints):
        
                            PA = update_PA(fracDem[1:])
                            valuesPA.append(PA)
                
                            if objective == 'PA':
                                objValue = PA

                        # Update Cmpttv
                        if (objective == 'cmpttv') or ('cmpttv' in constraints):
                        
                            compFrac[distFrom] = abs(2*fracDem[distFrom] - 1)
                            compFrac[distToo] = abs(2*fracDem[distToo] - 1)
                            Cmpttv = sum(compFrac)/(len(compFrac)-1)
                            valuesCmpttv.append(Cmpttv)
                            
                            if objective == 'cmpttv':
                                objValue = Cmpttv


                        # Update % white, Black, Lat/Hisp
                        fracW[distFrom] = newWFrom
                        fracB[distFrom] = newBFrom
                        fracL[distFrom] = newLFrom

                        fracW[distToo] = newWToo
                        fracB[distToo] = newBToo
                        fracL[distToo] = newLToo

    
    
# OUTPUT ------------------------------------------------------------------------------------------


        print('Final max population deviation: ',popDev)
        print('Final total perimeter: ',totalPerim)

        if (objective == 'eg') or ('eg' in constraints):
            print('Final EG: ', EGNominal)

        if (objective == 'eg_shift') or ('eg_shift' in constraints):
            print('Final Max EG Shift: ',EGShift)

        if (objective == 'mm') or ('mm' in constraints):
            print('Final MM: ',MM)
        
        if (objective == 'pa') or ('pa' in constraints):
            print('Final PA: ',PA)

        if (objective == 'cmpttv') or ('cmpttv' in constraints):
        
            print('Final avg margin (Cmpttv): ',Cmpttv)
            print('Final max margin (Cmpttv): ',max(compFrac))
            countCmpttv = 0
            for cp in compFrac[1:]:
                if cp <= 0.1:
                    countCmpttv += 1
                    #print(cp)

            print('Margins <= 10%: ',countCmpttv)


        if 'demo' in constraints:

            numPlurB = 0
            numPlurL = 0
            numPlurW = 0

            for i in range(1,numDistricts):
                if fracW[i] < 0.5:
                    if fracB[i] > fracW[i] and fracB[i] > fracL[i]:
                        numPlurB += 1
                    elif fracL[i] > fracW[i] and fracL[i] > fracB[i]:
                        numPlurL += 1
                    else:
                        numPlurW += 1

            print('Final # plurality-Black MM districts: ',numPlurB)
            print('Final # plurality-Lat/Hisp MM districts: ',numPlurL)
            print('Final # plurality-white MM districts: ',numPlurW)

    
    

        # Record map runtime
        mapTime = round((time.time()-startMap)/60,2)
        print('Runtime for this map: ',mapTime,' minutes')


        if StateFolder[0:5] == '2020/':
            StateFolder = StateFolder[5:]
            
        if StateFolder[0:9] == '2020_NEW/':
            StateFolder = StateFolder[9:]


        # Used for output file name
        objectiveString = ''
        
        if objective == 'none':
            
            objectiveString = objective + 'Obj'

        elif objective == 'pop':
            
            objectiveString = objective + 'Obj' + str(round(popDev,4))

            plt.plot(valuesPop)
            plt.xlabel('Iterations')
            plt.ylabel('Max pop dev')
            plt.title('Change in max pop dev over time')
            plt.show()

        elif objective == 'perim':
            
            objectiveString = objective + 'Obj' + str(int(totalPerim))

            plt.plot(valuesPerim)
            plt.xlabel('Iterations')
            plt.ylabel('Total perimeter')
            plt.title('Change in total perimeter over time')
            plt.show()
    
        elif objective == 'eg':
            
            objectiveString = objective + 'Obj' + str(round(EGNominal,5))

            plt.plot(valuesEGNominal)
            plt.xlabel('Iterations')
            plt.ylabel('EG')
            plt.title('Change in EG over time')
            plt.show()

        elif objective == 'eg_shift':
            
            objectiveString = objective + 'Obj' + str(round(EGShift,5))

            plt.plot(valuesEGShift)
            plt.xlabel('Iterations')
            plt.ylabel('Max EG Shift')
            plt.title('Change in Max EG Shift over time')
            plt.show()

        elif objective == 'mm':

            objectiveString = objective + 'Obj' + str(round(MM,5))
            
            plt.plot(valuesMM)
            plt.xlabel('Iterations')
            plt.ylabel('MM')
            plt.title('Change in MM over time')
            plt.show()
        
        elif objective == 'pa':

            objectiveString = objective + 'Obj' + str(round(PA,5))
            
            plt.plot(valuesPA)
            plt.xlabel('Iterations')
            plt.ylabel('PA')
            plt.title('Change in PA over time')
            plt.show()

        elif objective == 'cmpttv':
            
            objectiveString = objective + 'Obj' + str(countCmpttv) + '_' + str(round(Cmpttv,5))

            plt.plot(valuesCmpttv)
            plt.xlabel('Iterations')
            plt.ylabel('Avg Cmpttv')
            plt.title('Change in Avg Cmpttv over time')
            plt.show()



            
        constraintString = ''
#         for c in constraints:
#             constraintString += '_'+c
            
        if 'eg' in constraints:
            constraintString += '_EG'+str(round(EGNominal,5))
            
        if 'eg_shift' in constraints:
            constraintString += '_EGShift'+str(round(EGShift,5))
            
        if 'mm' in constraints:
            constraintString += '_MM'+str(round(MM,5))
            
        if 'pa' in constraints:
            constraintString += '_PA'+str(round(PA,5))
            
        if 'cmpttv' in constraints:
            constraintString += '_Cmpttv'+str(countCmpttv) + '_' + str(round(Cmpttv,5))
            
        if 'demo' in constraints:
            constraintString += '_demo'
            
        if 'whole' in constraints:
            constraintString += '_whole'
            
        if objective != 'perim':
            constraintString += '_Perim'+str(int(totalPerim))
        
        
        # Open and write to output file
        outFile = open(OutputFolder + '/' + StateFolder + '_It' + str(k)+'_'+str(mapTime)+'minutes'+'_B'+str(edgeBonus)+'_P'+str(edgePenalty)+'_'+objectiveString+constraintString+'.csv','w')
            
        writer = csv.writer(outFile,delimiter=',')
        writer.writerow(['GEOID','district'])
        for d in data:
            if d != '0':
                writer.writerow([d,data[d]])

        outFile.close()


    print('\nRuntime = ',time.time()-start,' seconds')


# In[ ]:




