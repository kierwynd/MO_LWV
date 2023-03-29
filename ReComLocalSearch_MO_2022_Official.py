from gerrychain.random import random

SEED = 0
while (SEED % 2 == 0):
    SEED = random.randrange(1000000001,9999999999)
#SEED = 7473378879
random.seed(SEED)
print(SEED)


import matplotlib.pyplot as plt
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election, tree)
#from gerrychain.proposals import recom
from gerrychain.tree import (recursive_tree_part, bipartition_tree)
from functools import partial
import pandas
import geopandas as gp
import time
import csv
import os
import numpy as np



# FUNCTIONS --------------------------------------------------------------------------------------------

# Input: partition object
# Output: number of majority-minority districts
def calculate_MM_districts_basic(my_partition):
    
    plur_W = 0
    
    for district in my_partition['nhwhitepop']:
    
        fraction_W = my_partition['nhwhitepop'][district]/my_partition['population'][district]

        if fraction_W < 0.5:
            plur_W += 1
            
    return plur_W


# Input: partition object
# Output: number of majority-minority districts that are plurality Black, Lat/Hisp, and nh-white
def calculate_MM_districts(my_partition):
    
    plur_B = 0
    plur_L = 0
    plur_W = 0
    
    for district in my_partition['nhwhitepop']:
    
        fraction_W = my_partition['nhwhitepop'][district]/my_partition['population'][district]
        fraction_B = my_partition['blackpop'][district]/my_partition['population'][district]
        fraction_L = my_partition['latpop'][district]/my_partition['population'][district]

        if fraction_W < 0.5:
            if fraction_B > max(fraction_W,fraction_L):
                plur_B += 1
            elif fraction_L > max(fraction_W,fraction_B):
                plur_L += 1
            else:
                plur_W += 1
            
    return plur_B,plur_L,plur_W


# Input: partition object
# Output: number of competitive districts (within 'margin' of victory) in that partition
def calculate_Cmpttv_districts(my_partition, margin):
    
    count_cmpttv = 0
    
    for dem_percent in my_partition['AVG'].percents('Dem'):
        if abs(dem_percent-(1-dem_percent)) <= margin:
            count_cmpttv += 1
            
    return count_cmpttv


# Input: partition object and two-district tuple
# Output: number of districts (out of the two) that are wholly contained within one county
def num_wholly_contained(my_partition, two_dist):
    
    count_wholly_contained = 0
    
    # Count number of counties for first district
    counties = []
    for n in list(my_partition.assignment.parts[two_dist[0]]):
        counties.append(graph.nodes[n]['GEOID20'][0:5])
    if len(set(counties)) == 1:
        count_wholly_contained += 1
        
    # Count number of counties for second district
    counties = []
    for n in list(my_partition.assignment.parts[two_dist[1]]):
        counties.append(graph.nodes[n]['GEOID20'][0:5])
    if len(set(counties)) == 1:
        count_wholly_contained += 1
        
    
    return count_wholly_contained


# Calculate Shifted Efficiency Gap
# Input: for a given plan, the fraction of dem voters in each district
#        and total number of dem/rep voters in each district
# Output: Shifted Efficiency Gap value (max absolute EG value when vote-shares
#         are shifted 0-5% uniformly in either party's favor)
def calculate_SEG(FD,TV_dem,TV_rep):
    
    FD_list = [fd for fd in FD]
    TV = [TV_dem[i]+TV_rep[i] for i in range(0,len(TV_dem))]
    
    # Shifted Efficiency Gap objective function (MO)
    percentShiftsNeg = [-0.05,-0.04,-0.03,-0.02,-0.01,0.0]
    percentShiftsPos = [0.01,0.02,0.03,0.04,0.05]
    allVotes = np.sum(TV)
    EG_shift = []

    for s in percentShiftsNeg:
        fracDem_shift = [max(fd+s,0.0) for fd in FD_list]
        numDem_shift = []
        numRep_shift = []
        for i in range(0,len(TV)):
            numDem_shift.append(fracDem_shift[i]*TV[i])
            numRep_shift.append((1-fracDem_shift[i])*TV[i])
        
        wasted_shift_dem = []
        wasted_shift_rep = []
        for i in range(0,len(numDem_shift)):
            if numDem_shift[i] >= numRep_shift[i]:
                wasted_shift_dem.append(numDem_shift[i] - 0.5*(numDem_shift[i]+numRep_shift[i]))
                wasted_shift_rep.append(numRep_shift[i])
                #wasted_shift.append(.5*(numDem_shift[i]-3*numRep_shift[i]))
            else:
                wasted_shift_rep.append(numRep_shift[i] - 0.5*(numDem_shift[i]+numRep_shift[i]))
                wasted_shift_dem.append(numDem_shift[i])
                #wasted_shift.append(.5*(3*numDem_shift[i]-numRep_shift[i]))
                
        #EG_shift.append(round(abs(sum(wasted_shift))/allVotes,10))
        wasted_shift = np.sum(wasted_shift_rep) - np.sum(wasted_shift_dem)
        EG_shift.append(np.round(wasted_shift/allVotes,10))
        
        
    for s in percentShiftsPos:
        fracDem_shift = [min(fd+s,1.0) for fd in FD_list]
        numDem_shift = []
        numRep_shift = []
        for i in range(0,len(TV)):
            numDem_shift.append(fracDem_shift[i]*TV[i])
            numRep_shift.append((1-fracDem_shift[i])*TV[i])
            
        wasted_shift_dem = []
        wasted_shift_rep = []
        for i in range(0,len(numDem_shift)):
            if numDem_shift[i] >= numRep_shift[i]:
                wasted_shift_dem.append(numDem_shift[i] - 0.5*(numDem_shift[i]+numRep_shift[i]))
                wasted_shift_rep.append(numRep_shift[i])
                #wasted_shift.append(.5*(numDem_shift[i]-3*numRep_shift[i]))
            else:
                wasted_shift_rep.append(numRep_shift[i] - 0.5*(numDem_shift[i]+numRep_shift[i]))
                wasted_shift_dem.append(numDem_shift[i])
                #wasted_shift.append(.5*(3*numDem_shift[i]-numRep_shift[i]))
                
        #EG_shift.append(round(abs(sum(wasted_shift))/allVotes,10))
        wasted_shift = np.sum(wasted_shift_rep) - np.sum(wasted_shift_dem)
        EG_shift.append(np.round(wasted_shift/allVotes,10))
        

#     print('-----')
#     for val in EG_shift:
#         print(val)

    #return max(EG_shift)
    #return EG_shift

    abs_seg = [abs(eg_val) for eg_val in EG_shift]
    max_abs_val = max(abs_seg)
    
    return max_abs_val
    

# Modified from GerryChain package:
# https://github.com/mggg/GerryChain/blob/main/gerrychain/proposals/tree_proposals.py
def recom(my_partition, pop_col, pop_target, epsilon, node_repeats=1, method=bipartition_tree):
    
    edge = random.choice(tuple(my_partition['cut_edges']))
    parts_to_merge = (my_partition.assignment[edge[0]], my_partition.assignment[edge[1]])
    #print('parts_to_merge: ',parts_to_merge)

    subgraph = my_partition.graph.subgraph(
        my_partition.parts[parts_to_merge[0]] | my_partition.parts[parts_to_merge[1]]
    )
    
    # This is the modified part -----
    # Need to modify target population and epsilon based on the two districts chosen
    
    new_pop_target = (my_partition['population'][parts_to_merge[0]] + my_partition['population'][parts_to_merge[1]])/2
    new_epsilon = min(abs(1-(((1+epsilon)*pop_target)/new_pop_target)),abs(1-(((1-epsilon)*pop_target)/new_pop_target)))
    
    #-----

    flips = recursive_tree_part(
        subgraph,
        parts_to_merge,
        pop_col=pop_col,
        pop_target=new_pop_target,
        epsilon=new_epsilon,
        node_repeats=node_repeats,
        method=method,
    )

    # Also modified return statement - here we additionally return the parts to merge
    return my_partition.flip(flips),parts_to_merge


# Input: partition object, constraint/threshold dictionary, objective string
# Output: partition object
# Proposal function returns new population-balanced partition
# New partition checked for constraints and objective improvement
# If new partition passes checks, return new partition. Else, return original partition.
def ReComMove(my_partition,constraint_dict,obj):
    
    proposed_next_state,two_districts = proposal(my_partition)
    #print(two_districts)
    
    checks = True
    why = ''
    
    # Check population balance - UNNECESSARY because built into recom proposal with our modification
#     popDev = max([abs(1-(p/ideal_population))for p in proposed_next_state['population'].values()])
#     if popDev > max_population_dev:
#         checks = False
    
    # Check constraints
    if checks and ('demo_basic' in constraint_dict):
        plur_W = calculate_MM_districts_basic(proposed_next_state)
        if plur_W != constraint_dict['demo_basic']:
            checks = False
            why = 'demo_basic'
            
    if checks and ('demo' in constraint_dict):
        plur_B,plur_L,plur_W = calculate_MM_districts(proposed_next_state)
        if plur_B != constraint_dict['demo']['plurB'] or plur_L != constraint_dict['demo']['plurL'] or plur_W != constraint_dict['demo']['plurW']:
            checks = False
            why = 'demo'
                
    if checks and ('perim' in constraint_dict):
        if np.sum(list(proposed_next_state['perimeter'].values())) > constraint_dict['perim']:
            checks = False
            why = 'perim'
            
    if checks and ('cut_edges' in constraint_dict):
        if len(proposed_next_state['cut_edges']) > constraint_dict['cut_edges']:
            checks = False
            why = 'cut_edges'
            
    if checks and ('cmpttv' in constraint_dict) and obj != 'cmpttv':
        if calculate_Cmpttv_districts(proposed_next_state,constraint_dict['cmpttv']) != calculate_Cmpttv_districts(my_partition,constraint_dict['cmpttv']):
            checks = False
            why = 'cmpttv'
            
    if checks and ('eg' in constraint_dict):
        if abs(proposed_next_state['AVG'].efficiency_gap()) > constraint_dict['eg']:
            checks = False
            why = 'eg'
            
    if checks and ('eg_shift' in constraint_dict):
        if calculate_SEG(proposed_next_state['AVG'].percents('Dem'),proposed_next_state['AVG'].counts('Dem'),proposed_next_state['AVG'].counts('Rep')) > constraint_dict['eg_shift']:
            checks = False
            why = 'eg_shift'
            
    if checks and ('mm' in constraint_dict):
        if abs(proposed_next_state['AVG'].mean_median()) > constraint_dict['mm']:
            checks = False
            why = 'mm'
            
    if checks and ('pa' in constraint_dict):
        if proposed_next_state['AVG'].partisan_gini() > constraint_dict['pa']:
            checks = False
            why = 'pa'
            
    if checks and ('whole' in constraint_dict):
        if num_wholly_contained(proposed_next_state, two_districts) != num_wholly_contained(my_partition, two_districts):
            checks = False
            why = 'whole'
            
            
    # Check objective
    if checks:
        if obj == 'pop':
            if max([abs(1-((proposed_next_state['population'][dist])/ideal_population)) for dist in two_districts]) > max([abs(1-((my_partition['population'][dist])/ideal_population)) for dist in two_districts]):
                checks = False
                why = 'pop'
        
        elif obj == 'perim':
            if np.sum(list(proposed_next_state['perimeter'].values())) > np.sum(list(my_partition['perimeter'].values())):
                checks = False
                why = 'perim'
                
        elif obj == 'cut_edges':
            if len(proposed_next_state['cut_edges']) > len(my_partition['cut_edges']):
                checks = False
                why = 'cut_edges'
                
        elif obj == 'eg':
            if abs(proposed_next_state['AVG'].efficiency_gap()) > abs(my_partition['AVG'].efficiency_gap()):
                checks = False
                why = 'eg'
                
        elif obj == 'eg_shift':
            if calculate_SEG(proposed_next_state['AVG'].percents('Dem'),proposed_next_state['AVG'].counts('Dem'),proposed_next_state['AVG'].counts('Rep')) > calculate_SEG(my_partition['AVG'].percents('Dem'),my_partition['AVG'].counts('Dem'),my_partition['AVG'].counts('Rep')):
                checks = False
                why = 'eg_shift'
                
        elif obj == 'mm':
            if abs(proposed_next_state['AVG'].mean_median()) > abs(my_partition['AVG'].mean_median()):
                checks = False
                why = 'mm'
                
        elif obj == 'pa':
            if proposed_next_state['AVG'].partisan_gini() > my_partition['AVG'].partisan_gini():
                checks = False
                why = 'pa'
                
        elif obj == 'cmpttv':
            if calculate_Cmpttv_districts(proposed_next_state,constraint_dict['cmpttv_obj']) < calculate_Cmpttv_districts(my_partition,constraint_dict['cmpttv_obj']):
                checks = False
                why = 'cmpttv'

    
    if checks:
        #print('Success')
        return proposed_next_state#,why
    else:
        #print('Fail')
        return my_partition#,why





# READ IN ALL PARAMETERS --------------------------------------------------------------------------------

# Create boolean variable to report whether user input is valid
VALID = True

# Read in parameters
parameterFile = open('ReCom_PARAMETERS_MO_2022.csv','r')
readerParam = csv.reader(parameterFile,delimiter=',')
parameters = [line for line in readerParam]

# Folder of district plan files
PlanFolder = parameters[0][1]
print('Folder with maps: ',PlanFolder)

if os.path.isdir(PlanFolder):
    maps = [p for p in os.listdir(PlanFolder+'/') if not p.startswith('.')]
    maps.sort()
else:
    VALID = False
    print('\n-----Folder does not exist-----\n')

if len(parameters[0]) > 2 and os.path.isdir(PlanFolder):
    if parameters[0][2] != '':
        maps = []
        for val in parameters[0][2:]:
            if val == '':
                continue
            elif os.path.isfile(PlanFolder+'/'+val):
                maps.append(val)
            else:
                VALID = False
                print('\n-----File does not exist-----\n')
                break

        print('Maps:\n',maps)
        
print('')
for file in maps:
    if not os.path.isfile(PlanFolder+'/'+file):
        print(file)
        VALID = False
        print('\n-----File does not exist-----\n')
        break
    

# Folder with state data
StateFolder = parameters[1][1]
print('Folder with state data: ',StateFolder)

if not os.path.isdir(StateFolder):
    VALID = False
    print('\n-----Folder does not exist-----\n')
else:
    files = [t for t in os.listdir(StateFolder+'/') if (not t.startswith('.') and t[-4:] == '.shp')]
    #files.sort()
    if len(files) != 1:
        VALID = False
        print('\n-----More than 1 or less than 1 .shp file in this folder-----\n')


# Folder for output
OutputFolder = parameters[2][1]
if len(parameters[2]) > 2:
    output_name = parameters[2][2]
else:
    output_name = ''
    
print('Folder for output: ',OutputFolder)

if not os.path.isdir(OutputFolder):
    VALID = False
    print('\n-----Folder does not exist-----\n')


# Assign number of iterations
num_iterations = int(parameters[3][1])
print('Number of iterations: ',num_iterations)


# Determine objective function
objective = parameters[4][1]

ValidObjectives = ['none','pop','perim','cut_edges','eg','eg_shift','mm','pa','cmpttv']

if objective in ValidObjectives:
    print('Objective: ',objective)
else:
    VALID = False
    print('\n-----',objective,' is not a valid objective-----\n')
    

# Determine constraints
constraints = parameters[5][1:]
constraints = [c for c in constraints if (c != '')]
print('Constraints:\n',constraints)

ValidConstraints = ['pop','perim','cut_edges','eg','eg_shift','mm','pa','cmpttv','demo_basic','demo','whole']

for c in constraints:
    if c not in ValidConstraints:
        VALID = False
        print('\n-----',c,' is not a valid constraint-----\n')


# Assign thresholds
max_pop_dev = float(parameters[6][1])
perim_add = float(parameters[7][1])
cut_edges_mult = float(parameters[8][1])
eg_threshold = float(parameters[9][1])
seg_threshold = float(parameters[10][1])
mm_threshold = float(parameters[11][1])
pa_threshold = float(parameters[12][1])
cmpttv_margin = float(parameters[13][1])

if 'pop' in constraints:
    print('Maximum allowed population deviation: ',max_pop_dev)
    
if 'perim' in constraints:
    print('Value added to initial perimeter for perimeter threshold: ',perim_add)
    
if 'cut_edges' in constraints:
    print('Value multiplied by initial # cut_edges for cut_edges threshold: ',cut_edges_mult)
    
if 'eg' in constraints:
    print('EG threshold: ',eg_threshold)
    
if 'eg_shift' in constraints:
    print('Shifted EG threshold: ',seg_threshold)
    
if 'mm' in constraints:
    print('MM threshold: ',mm_threshold)
    
if 'pa' in constraints:
    print('PA threshold: ',pa_threshold)
    
if objective == 'cmpttv' or 'cmpttv' in constraints:
    print('Margin of competitiveness: ',cmpttv_margin)
    

# Assign edge weights
edgeBonus = float(parameters[14][1])
edgePenalty = float(parameters[15][1])

print('Inter-county perimeter segment bonus: ',edgeBonus)
print('Intra-county perimeter segment penalty: ',edgePenalty)


# Determine if user wants objective convergence
converge = parameters[16][1]

if converge == 'yes' and (objective == 'none'):
    converge = False
    print('Convergence is only an option for non-none objectives, convergence choice ignored.')
elif converge == 'yes':
    converge = True
    print('Run until objective values converge')
    if len(parameters[16]) > 2:
        if parameters[16][2] != '':
            convergence_threshold = float(parameters[16][2])
            print('Convergence threshold: ',convergence_threshold)
        else:
            VALID = False
            print('\n----- Did not provide convergence threshold -----\n')
    else:
        VALID = False
        print('\n----- Did not provide convergence threshold -----\n')
elif converge == 'no':
    converge = False
    print('Run for a fixed number of iterations')
else:
    VALID = False
    print('\n-----',converge,' is not a valid convergence option-----\n')
    
    
# Determine number of repeated runs
repeated_runs = int(parameters[17][1])

if repeated_runs <= 0:
    VALID = False
    print('\n-----',repeated_runs,' is not a valid number of replications-----\n')
else:
    print('Number of replications per map: ',repeated_runs)
    
    
# Determine number of initial None iterations (i.e., if there should be a random walk before optimization)
num_none = int(parameters[18][1])

if num_none < 0:
    VALID = False
    print('\n-----',num_none,' is not a valid number of random walk steps-----\n')
else:
    print('Number of steps in random walk before optimization: ',num_none)
    
    
# Determine if certain county boundaries are "uncrossable" (i.e., county adjacencies to remove)
forbidden = []
if len(parameters[19]) >= 3:
    if (parameters[19][1] != '') and (parameters[19][2] != ''):
        for entry in parameters[19:]:
            forbidden.append(entry[1:3])
            
print('Adjacency to remove:\n',forbidden)
    
# Number of roots to choose for ReCom iteration
num_node_repeats = 1


# If user input is invalid, stop program
if not VALID:
    print('\nINVALID USER INPUT -- continuing to run will give an error')
else:
    print('\nUSER INPUT IS VALID -- good to go!')
    print('\nLoading state data now (shapefile may take a minute or two to load, depending on size)')
    print('...')

    
# SET-UP ----------------------------------------------------------------------------------------------------

    # Read in shapefile with geopandas
    shapefile_path = StateFolder + '/' + files[0]
    df = gp.read_file(shapefile_path)

    # Create Graph object from gerrychain
    graph = Graph.from_geodataframe(df)
    
    # Make sure all GEOIDs are strings, just in case
    for node in graph.nodes:
        temp = str(int(graph.nodes[node]['GEOID20']))
        graph.nodes[node]['GEOID20'] = temp
    
    # Create Election object from gerrychain
    election = Election('AVG', {'Dem': 'VOTES_DEM', 'Rep': 'VOTES_REP'})

    # Apply edge weights
    for edge in graph.edges:
        
        if graph.nodes[edge[0]]['GEOID20'][0:5] == graph.nodes[edge[1]]['GEOID20'][0:5]:
            temp = graph.edges[edge]['shared_perim']
            graph.edges[edge]['shared_perim'] = edgePenalty*temp
            
        else:
            temp = graph.edges[edge]['shared_perim']
            graph.edges[edge]['shared_perim'] = edgeBonus*temp
        
        
    # Remove any forbidden adjacencies
    for edge in graph.edges:
        if ([graph.nodes[edge[0]]['GEOID20'][0:5],graph.nodes[edge[1]]['GEOID20'][0:5]] in forbidden) or ([graph.nodes[edge[1]]['GEOID20'][0:5],graph.nodes[edge[0]]['GEOID20'][0:5]] in forbidden):
            graph.remove_edge(edge[0],edge[1])
        

        
    print('Done!')
        

        
#     print(graph.nodes.data())
#     for node in graph.nodes:
#         print(graph.nodes[node]['GEOID20'])

#     print(graph.edges.data())


#     # Can output unit adjacency list
#     outAdj = open('Adjacency.csv','w')
#     writerAdj = csv.writer(outAdj,delimiter=',')

#     writerAdj.writerow(['Unit 1','Unit 2','Shared perim (meters)'])
#     for edge in graph.edges:
#         writerAdj.writerow([str(graph.nodes[edge[0]]['GEOID20']),str(graph.nodes[edge[1]]['GEOID20']),str(graph.edges[edge]['shared_perim'])])

#     for node in graph.nodes:
#         if graph.nodes[node]['boundary_node']:
#             writerAdj.writerow([str(graph.nodes[node]['GEOID20']),'outside',graph.nodes[node]['boundary_perim']])


#     outAdj.close()





# RUN THROUGH MAPS --------------------------------------------------------------------------------------


for i in range(0,repeated_runs):
    
    for file in maps:
        
        # Read in district plan
        planFile = open(PlanFolder + '/' + file,'r')
        reader = csv.reader(planFile,delimiter=',')

        labels = next(reader)
        plan = {}
        for line in reader:
            if line != [] and line != ['','']:
                plan[line[0]] = line[1]

        planFile.close()
        #print(plan)


        # Make plan dictionary for nodes in graph object
        plan_nodes = {}

        for node in graph.nodes:
            plan_nodes[node] = plan[graph.nodes[node]['GEOID20']]

        #print(plan_nodes)


        # Create a GeographicPartition object from gerrychain (initial district plan)
        # GeographicPartition automatically has updaters for perim, area, cut_edges, and more!
        
        initial_partition = GeographicPartition(
            graph,
            assignment = plan_nodes,
            updaters = {
                'population': updaters.Tally('POP20', alias='population'),
                'nhwhitepop': updaters.Tally('NHWHITEPOP', alias='nhwhitepop'),
                'blackpop': updaters.Tally('BLACKPOP', alias='blackpop'),
                'latpop': updaters.Tally('HISPLATPOP', alias='latpop'),
                'AVG': election
            }
        )


        # Set the ideal population
        ideal_population = np.sum(list(initial_partition['population'].values()))/len(initial_partition)
        
        # Create dictionary of constraint parameters
        constraint_parameters = {}
        
        if 'whole' in constraints:
            constraint_parameters['whole'] = 1
        
        if 'eg' in constraints:
            constraint_parameters['eg'] = eg_threshold
    
        if 'eg_shift' in constraints:
            constraint_parameters['eg_shift'] = seg_threshold
            #print('Initial SEG: ',calculate_SEG(initial_partition['AVG'].percents('Dem'),initial_partition['AVG'].counts('Dem'),initial_partition['AVG'].counts('Rep')))

        if 'mm' in constraints:
            constraint_parameters['mm'] = mm_threshold

        if 'pa' in constraints:
            constraint_parameters['pa'] = pa_threshold

        if objective == 'cmpttv':
            constraint_parameters['cmpttv_obj'] = cmpttv_margin
            
        if 'cmpttv' in constraints:
            constraint_parameters['cmpttv'] = cmpttv_margin
        
        # Calculate number of majority-minority districts
        if 'demo_basic' in constraints:
            num_plurality_W = calculate_MM_districts_basic(initial_partition)
            constraint_parameters['demo_basic'] = num_plurality_W
            #print(num_plurality_W)

        # Calculate number of majority-minority districts that are plurality Black, Lat/Hisp, or nh-white
        if 'demo' in constraints:
            num_plurality_B, num_plurality_L, num_plurality_W = calculate_MM_districts(initial_partition)
            constraint_parameters['demo'] = {'plurB':num_plurality_B, 'plurL':num_plurality_L, 'plurW':num_plurality_W}
            #print(num_plurality_B, num_plurality_L, num_plurality_W)

        # Calculate perimeter threshold
        if 'perim' in constraints:
            perim_threshold = np.sum(list(initial_partition['perimeter'].values())) + (perim_add*1000)
            constraint_parameters['perim'] = perim_threshold
            
        # Calculate cut_edges threshold
        if 'cut_edges' in constraints:
            cut_edges_threshold = (len(initial_partition['cut_edges'])) * cut_edges_mult
            constraint_parameters['cut_edges'] = cut_edges_threshold


        # Record objective values
        if objective == 'pop':
            constraint_parameters['pop'] = ideal_population
            objective_values = [max([abs(i-ideal_population)/ideal_population for i in initial_partition['population'].values()])]
        
        elif objective == 'perim':
            objective_values = [np.sum(list(initial_partition['perimeter'].values()))]
            
        elif objective == 'cut_edges':
            objective_values = [len(initial_partition['cut_edges'])]

        elif objective == 'eg':
            objective_values = [initial_partition['AVG'].efficiency_gap()]
            
        elif objective == 'eg_shift':
            objective_values = [calculate_SEG(initial_partition['AVG'].percents('Dem'),initial_partition['AVG'].counts('Dem'),initial_partition['AVG'].counts('Rep'))]

        elif objective == 'mm':
            objective_values = [initial_partition['AVG'].mean_median()]
            
        elif objective == 'pa':
            objective_values = [initial_partition['AVG'].partisan_gini()]

        elif objective == 'cmpttv':
            objective_values = [calculate_Cmpttv_districts(initial_partition,constraint_parameters['cmpttv_obj'])]


        # Build proposal function
        proposal = partial(recom,
                           pop_col='POP20',
                           pop_target=ideal_population,
                           epsilon=max_pop_dev,
                           node_repeats=num_node_repeats
                          )


        # Print out some basic initial info
        print('Initial max pop dev: ',max([abs(i-ideal_population)/ideal_population for i in initial_partition['population'].values()]))

        if objective != 'perim':
            print('Initial perim: ',np.sum(list(initial_partition['perimeter'].values()))/1000)
        if objective != 'cut_edges':
            print('Initial cut_edges: ',len(initial_partition['cut_edges']))

        if objective != 'none':
            if objective == 'perim':
                print('Initial ' + objective + ': ',objective_values[-1]/1000)
            else:
                print('Initial ' + objective + ': ',objective_values[-1])



# ITERATIONS ------------------------------------------------------------------------------------------------

        start = time.time()

        # Perform iterations
        partition_OLD = initial_partition
        k = 0
        finished = False
        numIt = 0
        
#         countRejects = {}
#         for re in reasons:
#             countRejects[re] = 0


        # Perform any random walk steps before optimization iterations
        if num_none > 0:
        
            print('Random walk steps:')
            
            for k_none in range(0,num_none):
                
                if k_none % 25 == 0:
                    print(k_none)

                    # Get new partition
                    partition_NEW = ReComMove(partition_OLD, constraint_parameters, 'none')
                    
                    # New partition becomes old partition
                    partition_OLD = partition_NEW
                    
            
            # Record objective values
            if objective == 'pop':
                objective_values = [max([abs(i-ideal_population)/ideal_population for i in initial_partition['population'].values()])]

            elif objective == 'perim':
                objective_values = [np.sum(list(initial_partition['perimeter'].values()))]

            elif objective == 'cut_edges':
                objective_values = [len(initial_partition['cut_edges'])]

            elif objective == 'eg':
                objective_values = [initial_partition['AVG'].efficiency_gap()]

            elif objective == 'eg_shift':
                objective_values = [calculate_SEG(initial_partition['AVG'].percents('Dem'),initial_partition['AVG'].counts('Dem'),initial_partition['AVG'].counts('Rep'))]

            elif objective == 'mm':
                objective_values = [initial_partition['AVG'].mean_median()]

            elif objective == 'pa':
                objective_values = [initial_partition['AVG'].partisan_gini()]

            elif objective == 'cmpttv':
                objective_values = [calculate_Cmpttv_districts(initial_partition,constraint_parameters['cmpttv_obj'])]

            
            print(str(k_none+1),' random walk steps completed')


        # Perform optimization iterations
        while not finished:

            k += 1
            if converge and k >= 2:
                if (abs(objective_values[-2] - objective_values[-1]) < convergence_threshold):
                    numIt += 1
                else:
                    numIt = 0

                if numIt >= num_iterations:
                    finished = True
                    print(k)
                    continue

            elif k >= num_iterations:
                finished = True


            # Don't get rid of this perimeter calculation!
            currentPerim = np.round(np.sum(list(partition_OLD['perimeter'].values()))/1000)
            if k % 100 == 0:
                print(k)
                print(currentPerim)
                
            # Constrict pop threshold for recom proposal if optimizing for pop
            if k % 100 == 0 and objective == 'pop':
                max_pop_dev = max(0.03, max([abs(i-ideal_population)/ideal_population for i in partition_OLD['population'].values()]) + 0.001)
                print(max_pop_dev)
                # Build new proposal function
                proposal = partial(recom,
                           pop_col='POP20',
                           pop_target=ideal_population,
                           epsilon=max_pop_dev,
                           node_repeats=num_node_repeats
                )
                

            # Get new partition
            partition_NEW = ReComMove(partition_OLD, constraint_parameters, objective)
            
#             partition_NEW,whyRejected = ReComMove(partition_OLD, constraints, objective, cmpttv_margin)

#             if whyRejected != '':
#                 countRejects[whyRejected] += 1

            # Record new objective value
            if objective == 'pop':
                objective_values.append(max([abs(i-ideal_population)/ideal_population for i in partition_NEW['population'].values()]))
        
            elif objective == 'perim':
                objective_values.append(np.sum(list(partition_NEW['perimeter'].values())))

            elif objective == 'cut_edges':
                objective_values.append(len(partition_NEW['cut_edges']))

            elif objective == 'eg':
                objective_values.append(partition_NEW['AVG'].efficiency_gap())

            elif objective == 'eg_shift':
                objective_values.append(calculate_SEG(partition_NEW['AVG'].percents('Dem'),partition_NEW['AVG'].counts('Dem'),partition_NEW['AVG'].counts('Rep')))
                
            elif objective == 'mm':
                objective_values.append(partition_NEW['AVG'].mean_median())

            elif objective == 'pa':
                objective_values.append(partition_NEW['AVG'].partisan_gini())

            elif objective == 'cmpttv':
                objective_values.append(calculate_Cmpttv_districts(partition_NEW,constraint_parameters['cmpttv_obj']))


            # New partition becomes old partition
            partition_OLD = partition_NEW



# OUTPUT -----------------------------------------------------------------------------------------------------

        
        # Want perimeter output in kms not meters
        if objective == 'perim':    
            objective_values = [p/1000 for p in objective_values]


        # Plot objective values
        if objective != 'none':
            plt.plot(objective_values)
            plt.show()
            print('Final ',objective,' = ',objective_values[-1])


        # Print out some basic final info
        maxDev = max([abs(i-ideal_population)/ideal_population for i in partition_OLD['population'].values()])
        print('Final max pop dev = ',maxDev)

        finalPerim = np.round(np.sum(list(partition_OLD['perimeter'].values()))/1000)
        if objective != 'perim':
            print('Final perim = ',finalPerim)
            
        if objective != 'cut_edges':
            finalCutEdges = len(partition_OLD['cut_edges'])
            print('Final cut_edges = ',finalCutEdges)
            
            
        #print('Final SEG: ',calculate_SEG(initial_partition['AVG'].percents('Dem'),initial_partition['AVG'].counts('Dem'),initial_partition['AVG'].counts('Rep')))


        # Record time to complete run
        mapTime = np.round(time.time()-start,3)
        print('Runtime = ',mapTime,' seconds')


        # Output optimized map

        constraint_str = ''
        for c in constraints:
            constraint_str += c

        if objective != 'none':
            outFile_str = output_name+'_'+str(SEED)+'Seed_'+str(k)+'It_'+str(num_none)+'NoneIt_'+str(mapTime)+'seconds_'+str(edgeBonus)+'B_'+str(edgePenalty)+'P_'+str(objective)+'Obj'+str(np.round(objective_values[-1],5))+'_'+constraint_str+'.csv'
            outFile = open(OutputFolder + '/' + outFile_str,'w')
            
        else:
            outFile_str = output_name+'_'+str(SEED)+'Seed_'+str(k)+'It_'+str(mapTime)+'seconds_'+str(objective)+'Obj_'+constraint_str+'.csv'
            outFile = open(OutputFolder + '/' + outFile_str,'w')
            
            
        writer = csv.writer(outFile,delimiter=',')

        writer.writerow(['GEOID20','district'])
        for node in list(partition_OLD.graph.nodes):
            writer.writerow([graph.nodes[node]['GEOID20'],partition_OLD.assignment[node]])

        outFile.close()
        




