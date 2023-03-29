import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import time
import geopandas as gp
#import pandas
from gerrychain import (GeographicPartition, Partition, Graph, Election, updaters)


# Calculate Shifted Efficiency Gap
def SEG(FD,TV_dem,TV_rep):
    
    FD = list(FD)
    TV = [TV_dem[i]+TV_rep[i] for i in range(0,len(TV_dem))]
    
    # Shifted Efficiency Gap objective function (MO)
    percentShiftsNeg = [-0.05,-0.04,-0.03,-0.02,-0.01,0.0]
    percentShiftsPos = [0.01,0.02,0.03,0.04,0.05]
    allVotes = sum(TV)
    EG_shift = []

    for s in percentShiftsNeg:
        fracDem_shift = [max(fd+s,0.0) for fd in FD]
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
        wasted_shift = sum(wasted_shift_rep) - sum(wasted_shift_dem)
        EG_shift.append(round(wasted_shift/allVotes,10))
        
        
    for s in percentShiftsPos:
        fracDem_shift = [min(fd+s,1.0) for fd in FD]
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
        wasted_shift = sum(wasted_shift_rep) - sum(wasted_shift_dem)
        EG_shift.append(round(wasted_shift/allVotes,10))
        

#     print('-----')
#     for val in EG_shift:
#         print(val)

    #return max(EG_shift)
    return EG_shift





# Determine file names
shapefile_path = 'MO_Input_Senate/MO_Senate_HybridCountyTract_7+1SplitCounties_projected32615_Data/MO_Senate_HybridCountyTract_7+1SplitCounties_projected32615_Data.shp'

# Read in shapefile with geopandas
df = gp.read_file(shapefile_path)

# Create Graph object from gerrychain
graph = Graph.from_geodataframe(df)

# Make sure all GEOIDs are strings, just in case
for node in graph.nodes:
    temp = str(int(graph.nodes[node]['GEOID20']))
    graph.nodes[node]['GEOID20'] = temp

# Create Election object from gerrychain
election = Election('AVG', {'Dem': 'VOTES_DEM', 'Rep': 'VOTES_REP'})


#print(graph.nodes.data())





# FOR A FOLDER OF .csv PLANS FOR SAME SHAPEFILE DATA ---------------------------------------------------------

PlanFolderPath = 'Test'

JustFolderName = PlanFolderPath.split('/')

maps = [p for p in os.listdir(PlanFolderPath+'/') if not p.startswith('.')]
maps.sort()

#maps = ['']
    
    
# Open file for grouped metric values
#outAll = open(PlanFolderPath+'_Evals/Evaluation_'+JustFolderName[-1]+'.csv','w')
outAll = open('Evaluation_'+JustFolderName[-1]+'.csv','w')
writerAll = csv.writer(outAll,delimiter=',')

writerAll.writerow(['Plan','Pop Dev','% 1%','Cut-Edges','Perimeter','Efficiency Gap','Shifted EG',
                    'Mean-Median','Partisan Asymmetry','# Cmpttv 10%','# Cmpttv D','# Cmpttv R',
                    '# Cmpttv 7%','# Cmpttv D','# Cmpttv R','# Dem','# Rep','# MM Black',
                    '# MM Lat','# MM W','# Spanned','# Whole'])


for file in maps:
    
    # Record evaluation time for each map
    start = time.time()
    
    print('\nGiven district map: ',file)
    
    # Read in district plan
    planFile = open(PlanFolderPath + '/' + file,'r')
    reader = csv.reader(planFile,delimiter=',')

    labels = next(reader)
    plan = {}
    numDistricts = 0
    for line in reader:
        if line != [] and line != ['','']:
            plan[line[0]] = int(line[1])
            if int(line[1]) > numDistricts:
                numDistricts = int(line[1])

    planFile.close()
    #print(plan)
    
    
    # Group units by part
    partition = [[] for p in range(0,numDistricts+1)]

    for i in plan:
        partition[plan[i]].append(i)

    # Count number of counties spanned by each district
    numCounties = []
    for p in partition[1:]:
        temp = []
        for i in p:
            temp.append(i[0:5])

        temp = list(set(temp))
        numCounties.append(len(temp))
            

    print('\nNumber of counties spanned by districts: ',sum(numCounties))
    
    
    # Calculate number of districts with area in each county
    counties = {}
    for i in plan:
        counties[i[0:5]] = []

    for i in plan:
        if i != '0':
            counties[i[0:5]].append(plan[i])
        
    countWhole = 0
    for c in counties:
        if len(list(set(counties[c]))) == 1:
            countWhole += 1


    print('Number of whole counties: ',countWhole)
    


    # Make plan dictionary for nodes in graph object
    plan_nodes = {}

    for node in graph.nodes:
        plan_nodes[node] = plan[graph.nodes[node]['GEOID20']]

    #print(plan_nodes)


    # Create a GeographicPartition object from gerrychain (initial district plan)
    # GeographicPartition automatically has updaters for perim, area, cut-edges, and more!

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

    

    # Open file for individual metric values
    #outFile = open(PlanFolderPath+'_Evals/Evaluation_'+file[:-4]+'.txt','w')

    # Record map name
    #outFile.write('Given district map: '+str(file))
    #print('\nGiven district map: ',file)
    
    # Record pop deviation
    ideal_population = sum(initial_partition['population'].values())/len(initial_partition)
    pop_percent_dev = [abs(1-(p/ideal_population))for p in initial_partition['population'].values()]
    max_pop_percent_dev = max(pop_percent_dev)
    
    # Just for MO state senate/house
    count_1per = 0
    for per in pop_percent_dev:
        if per < 0.01:
            count_1per += 1
            
    print('\nPercent of districts within a 1% population percent deviation: '+str(count_1per/(len(pop_percent_dev))))

    #outFile.write('\n\nMax population percent deviation: '+str(max_pop_percent_dev))
    print('\nMax population percent deviation: '+str(max_pop_percent_dev))

    # Record # cut-edges
    #outFile.write('\n\nNumber of cut-edges (border edges): '+str(len(initial_partition['cut_edges'])))
    print('\nNumber of cut-edges (border edges): '+str(len(initial_partition['cut_edges'])))
    
    # Record compactness
    total_perim = sum(initial_partition['perimeter'].values())/1000
    
    #outFile.write('\n\nTotal perimeter: '+str(total_perim))
    print('\nTotal perimeter: '+str(total_perim))
          
    # Record voting metrics
    #outFile.write('\n\nEfficiency Gap (+: dem advantage, -: rep advantage): '+str(initial_partition['AVG'].efficiency_gap()))
    print('\nEfficiency Gap (+: dem advantage, -: rep advantage): '+str(initial_partition['AVG'].efficiency_gap()))
    
    #outFile.write('\n\nMean-Median (+: dem advantage, -:rep advantage): '+str(initial_partition['AVG'].mean_median()))
    print('\nMean-Median (+: dem advantage, -:rep advantage): '+str(initial_partition['AVG'].mean_median()))      
        
    #outFile.write('\n\nPartisan Asymmetry: '+str(initial_partition['AVG'].partisan_gini()))
    print('\nPartisan Asymmetry: '+str(initial_partition['AVG'].partisan_gini()))      
          
    seg = SEG(initial_partition['AVG'].percents('Dem'),initial_partition['AVG'].counts('Dem'),initial_partition['AVG'].counts('Rep'))
    abs_seg = [abs(val) for val in seg]
    #outFile.write('\n\nShifted Efficiency Gap: '+str(max(abs_seg))+'\n')
    print('\nShifted Efficiency Gap: '+str(max(abs_seg)))
    for val in seg:
        #outFile.write('\n'+str(val))
        print(val)

    count_cmpttv_10 = 0
    count_cmpttv_7 = 0
    dem_lean_10 = 0
    dem_lean_7 = 0
    for dem_percent in initial_partition['AVG'].percents('Dem'):
        if abs(dem_percent-(1-dem_percent)) < 0.07:
            count_cmpttv_7 += 1
            if dem_percent >= 0.5:
                dem_lean_7 += 1

        if abs(dem_percent-(1-dem_percent)) < 0.10:
            count_cmpttv_10 += 1
            if dem_percent >= 0.5:
                dem_lean_10 += 1


    #outFile.write('\n\n# Margins <= 7%: ' + str(count_cmpttv_7))
    #outFile.write('\n\t\t# Dem: ' + str(dem_lean_7))
    #outFile.write('\n\t\t# Rep: ' + str(count_cmpttv_7-dem_lean_7))
    print('\n# Margins <= 7%: ' + str(count_cmpttv_7))
    print('\n\t\t# Dem: ' + str(dem_lean_7))
    print('\n\t\t# Rep: ' + str(count_cmpttv_7-dem_lean_7))

    #outFile.write('\n\n# Margins <= 10%: ' + str(count_cmpttv_10))
    #outFile.write('\n\t\t# Dem: ' + str(dem_lean_10))
    #outFile.write('\n\t\t# Rep: ' + str(count_cmpttv_10-dem_lean_10))
    print('\n# Margins <= 10%: ' + str(count_cmpttv_10))
    print('\n\t\t# Dem: ' + str(dem_lean_10))
    print('\n\t\t# Rep: ' + str(count_cmpttv_10-dem_lean_10))

    
    print('\n# Dem Seats: '+str(initial_partition['AVG'].wins('Dem')))
    print('# Rep Seats: '+str(initial_partition['AVG'].wins('Rep')))
    #outFile.write('\n\n# Dem Seats: '+str(initial_partition['AVG'].wins('Dem')))
    #outFile.write('\n# Rep Seats: '+str(initial_partition['AVG'].wins('Rep')))
          
    percent_dem = int(1000*round(initial_partition['AVG'].count('Dem')/initial_partition['AVG'].total_votes(),3))/10
    percent_rep = int(1000*round(initial_partition['AVG'].count('Rep')/initial_partition['AVG'].total_votes(),3))/10
        
    print('\nPercent Dem: '+str(percent_dem)+'%')
    print('Percent Rep: '+str(percent_rep)+'%')
    #outFile.write('\n\nPercent Dem: '+str(percent_dem)+'%')
    #outFile.write('\nPercent Rep: '+str(percent_rep)+'%')
          
          
    # Record racial/ethnic demographics
    plur_B = 0
    plur_L = 0
    plur_W = 0
    
    print('\nRacial/Ethnic demographics:\n')
    #outFile.write('\n\nRacial/Ethnic demographics:\n')
    
    for district in initial_partition['nhwhitepop']:
    
        fraction_W = initial_partition['nhwhitepop'][district]/initial_partition['population'][district]
        fraction_B = initial_partition['blackpop'][district]/initial_partition['population'][district]
        fraction_L = initial_partition['latpop'][district]/initial_partition['population'][district]

        if fraction_W < 0.5:
            if fraction_B > max(fraction_W,fraction_L):
                plur_B += 1
            elif fraction_L > max(fraction_W,fraction_B):
                plur_L += 1
            else:
                plur_W += 1
          
            print('-----' + 'District '+str(district)+': % NH-white = '+str(round(100*fraction_W,2))+'%, % Black = '+str(round(100*fraction_B,2))+'%, % Lat/Hisp = '+str(round(100*fraction_L,2))+'%' + '-----')
            #outFile.write('\n-----' + 'District '+str(district)+': % NH-white = '+str(round(100*fraction_W,2))+'%, % Black = '+str(round(100*fraction_B,2))+'%, % Lat/Hisp = '+str(round(100*fraction_L,2))+'%' + '-----')

        else:
            print('District '+str(district)+': % NH-white = '+str(round(100*fraction_W,2))+'%, % Black = '+str(round(100*fraction_B,2))+'%, % Lat/Hisp = '+str(round(100*fraction_L,2))+'%' + '-----')
            #outFile.write('\nDistrict '+str(district)+': % NH-white = '+str(round(100*fraction_W,2))+'%, % Black = '+str(round(100*fraction_B,2))+'%, % Lat/Hisp = '+str(round(100*fraction_L,2))+'%' + '-----')


    print('\n# plurality-Black MM districts: '+str(plur_B))
    print('# plurality-Lat/Hisp MM districts: '+str(plur_L))
    print('# plurality-NH-white MM districts: '+str(plur_W))
    #outFile.write('\n\n# plurality-Black MM districts: ' +str(plur_B))
    #outFile.write('\n# plurality-Lat/Hisp MM districts: '+str(plur_L))
    #outFile.write('\n# plurality-NH-white MM districts: '+str(plur_W))
    
    
    
    # Plot vote-share margins
    test = [dem_percent for dem_percent in initial_partition['AVG'].percents('Dem')]
    test.sort()
    fracRep = [1-d for d in test]
    
    #print(test)
    
#     forLabels = [[fracDem[i],i] for i in range(1,len(fracDem[1:])+1)]
#     forLabels.sort()
#     labels = [FL[1] for FL in forLabels]
#     labels_str = [str(lab) for lab in labels]
#     #print(labels_str)
    
    x = np.arange(len(initial_partition['AVG'].percents('Dem')))
    x = [n+1 for n in x]
    y = [.5 for n in x]

#     # For MO House
#     plt.bar(x, test, width=1, color='blue', edgecolor='blue',label='Democrats')
#     plt.bar(x, fracRep, width=1, bottom=test, color='red', edgecolor='red',label='Republicans')
    
    # For not MO House
    plt.bar(x, test, color='blue', edgecolor='k',linewidth='.5',label='Democrats')
    plt.bar(x, fracRep, bottom=test, color='red', edgecolor='k',linewidth='.5',label='Republicans')
    
    #plt.xticks([i for i in range(1,len(initial_partition['AVG'].percents('Dem'))+1)])
    #plt.xticks([i for i in range(1,numDistricts)],labels_str)
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.plot(x,y,'k-',linewidth='1')
    plt.legend(loc='lower right')
    plt.xlabel('Districts')
    plt.ylabel('Vote-share')
    #plt.savefig('Margins_'+file[:-4]+'.pdf',dpi=600)
    plt.show()
    
    
    # Write to group file
    writerAll.writerow([str(file),str(max_pop_percent_dev),str(count_1per/(len(pop_percent_dev))),
                        str(len(initial_partition['cut_edges'])),str(total_perim),
                        str(initial_partition['AVG'].efficiency_gap()),
                        str(max(abs_seg)),str(initial_partition['AVG'].mean_median()),
                        str(initial_partition['AVG'].partisan_gini()),str(count_cmpttv_10),str(dem_lean_10),
                        str(count_cmpttv_10-dem_lean_10),str(count_cmpttv_7),str(dem_lean_7),
                        str(count_cmpttv_7-dem_lean_7),str(initial_partition['AVG'].wins('Dem')),
                        str(initial_partition['AVG'].wins('Rep')),str(plur_B),str(plur_L),str(plur_W),
                        str(sum(numCounties)),str(countWhole)])
    
          
    print('\nRuntime to evaluate this map = ',time.time()-start,' seconds\n\n')
    #outFile.write('\n\n')
    #outFile.close()
          
outAll.close()





# FOR A SINGLE PLAN THAT IS PART OF SHAPEFILE DATA -------------------------------------------------


# plan_name = 'DISTRICT'
# outFile_name = 'MO_Enacted_House_2022'


# # Create a GeographicPartition object from gerrychain (initial district plan)
# # GeographicPartition automatically has updaters for perim, area, cut-edges, and more!

# initial_partition = GeographicPartition(
#     graph,
#     assignment = plan_name,
#     updaters = {
#         'population': updaters.Tally('POP20', alias='population'),
#         'nhwhitepop': updaters.Tally('NHWHITEPOP', alias='nhwhitepop'),
#         'blackpop': updaters.Tally('BLACKPOP', alias='blackpop'),
#         'latpop': updaters.Tally('HISPLATPOP', alias='latpop'),
#         'AVG': election
#     }
# )

    

# # Open file for individual metric values
# outFile = open('Evaluation_'+outFile_name+'.txt','w')

# # Record map name
# outFile.write('Given district map: '+str(outFile_name))
# print('\nGiven district map: ',outFile_name)

# # Record pop deviation
# ideal_population = sum(initial_partition['population'].values())/len(initial_partition)
# max_pop_percent_dev = max([abs(1-(p/ideal_population))for p in initial_partition['population'].values()])

# outFile.write('\n\nMax population percent deviation: '+str(max_pop_percent_dev))
# print('\nMax population percent deviation: '+str(max_pop_percent_dev))

# # Record # cut-edges
# outFile.write('\n\nNumber of cut-edges (border edges): '+str(len(initial_partition['cut_edges'])))
# print('\nNumber of cut-edges (border edges): '+str(len(initial_partition['cut_edges'])))

# # Record compactness
# total_perim = sum(initial_partition['perimeter'].values())/1000

# outFile.write('\n\nTotal perimeter: '+str(total_perim))
# print('\nTotal perimeter: '+str(total_perim))

# # Record voting metrics
# outFile.write('\n\nEfficiency Gap (+: dem advantage, -: rep advantage): '+str(initial_partition['AVG'].efficiency_gap()))
# print('\nEfficiency Gap (+: dem advantage, -: rep advantage): '+str(initial_partition['AVG'].efficiency_gap()))

# outFile.write('\n\nMean-Median (+: dem advantage, -:rep advantage): '+str(initial_partition['AVG'].mean_median()))
# print('\nMean-Median (+: dem advantage, -:rep advantage): '+str(initial_partition['AVG'].mean_median()))      

# outFile.write('\n\nPartisan Asymmetry: '+str(initial_partition['AVG'].partisan_gini()))
# print('\nPartisan Asymmetry: '+str(initial_partition['AVG'].partisan_gini()))      


# seg = SEG(initial_partition['AVG'].percents('Dem'),initial_partition['AVG'].counts('Dem'),initial_partition['AVG'].counts('Rep'))
# abs_seg = [abs(val) for val in seg]
# outFile.write('\n\nShifted Efficiency Gap: '+str(max(abs_seg))+'\n')
# print('\nShifted Efficiency Gap: '+str(max(abs_seg)))
# for val in seg:
#     outFile.write('\n'+str(val))
#     print(val)

# count_cmpttv_10 = 0
# count_cmpttv_7 = 0
# dem_lean_10 = 0
# dem_lean_7 = 0
# for dem_percent in initial_partition['AVG'].percents('Dem'):
#     if abs(dem_percent-(1-dem_percent)) < 0.07:
#         count_cmpttv_7 += 1
#         if dem_percent >= 0.5:
#             dem_lean_7 += 1
            
#     if abs(dem_percent-(1-dem_percent)) < 0.10:
#         count_cmpttv_10 += 1
#         if dem_percent >= 0.5:
#             dem_lean_10 += 1


# outFile.write('\n\n# Margins <= 7%: ' + str(count_cmpttv_7))
# outFile.write('\n\t\t# Dem: ' + str(dem_lean_7))
# outFile.write('\n\t\t# Rep: ' + str(count_cmpttv_7-dem_lean_7))
# print('\n# Margins <= 7%: ' + str(count_cmpttv_7))
# print('\n\t\t# Dem: ' + str(dem_lean_7))
# print('\n\t\t# Rep: ' + str(count_cmpttv_7-dem_lean_7))

# outFile.write('\n\n# Margins <= 10%: ' + str(count_cmpttv_10))
# outFile.write('\n\t\t# Dem: ' + str(dem_lean_10))
# outFile.write('\n\t\t# Rep: ' + str(count_cmpttv_10-dem_lean_10))
# print('\n# Margins <= 10%: ' + str(count_cmpttv_10))
# print('\n\t\t# Dem: ' + str(dem_lean_10))
# print('\n\t\t# Rep: ' + str(count_cmpttv_10-dem_lean_10))


# print('\n# Dem Seats: '+str(initial_partition['AVG'].wins('Dem')))
# print('# Rep Seats: '+str(initial_partition['AVG'].wins('Rep')))
# outFile.write('\n\n# Dem Seats: '+str(initial_partition['AVG'].wins('Dem')))
# outFile.write('\n# Rep Seats: '+str(initial_partition['AVG'].wins('Rep')))

# percent_dem = int(1000*round(initial_partition['AVG'].count('Dem')/initial_partition['AVG'].total_votes(),3))/10
# percent_rep = int(1000*round(initial_partition['AVG'].count('Rep')/initial_partition['AVG'].total_votes(),3))/10

# print('\nPercent Dem: '+str(percent_dem)+'%')
# print('Percent Rep: '+str(percent_rep)+'%')
# outFile.write('\n\nPercent Dem: '+str(percent_dem)+'%')
# outFile.write('\nPercent Rep: '+str(percent_rep)+'%')


# # Record racial/ethnic demographics
# plur_B = 0
# plur_L = 0
# plur_W = 0

# print('\nRacial/Ethnic demographics:\n')
# outFile.write('\n\nRacial/Ethnic demographics:\n')

# for district in initial_partition['nhwhitepop']:

#     fraction_W = initial_partition['nhwhitepop'][district]/initial_partition['population'][district]
#     fraction_B = initial_partition['blackpop'][district]/initial_partition['population'][district]
#     fraction_L = initial_partition['latpop'][district]/initial_partition['population'][district]

#     if fraction_W < 0.5:
#         if fraction_B > max(fraction_W,fraction_L):
#             plur_B += 1
#         elif fraction_L > max(fraction_W,fraction_B):
#             plur_L += 1
#         else:
#             plur_W += 1

#         print('-----' + 'District '+str(district)+': % NH-white = '+str(round(100*fraction_W,2))+'%, % Black = '+str(round(100*fraction_B,2))+'%, % Lat/Hisp = '+str(round(100*fraction_L,2))+'%' + '-----')
#         outFile.write('\n-----' + 'District '+str(district)+': % NH-white = '+str(round(100*fraction_W,2))+'%, % Black = '+str(round(100*fraction_B,2))+'%, % Lat/Hisp = '+str(round(100*fraction_L,2))+'%' + '-----')

#     else:
#         print('District '+str(district)+': % NH-white = '+str(round(100*fraction_W,2))+'%, % Black = '+str(round(100*fraction_B,2))+'%, % Lat/Hisp = '+str(round(100*fraction_L,2))+'%' + '-----')
#         outFile.write('\nDistrict '+str(district)+': % NH-white = '+str(round(100*fraction_W,2))+'%, % Black = '+str(round(100*fraction_B,2))+'%, % Lat/Hisp = '+str(round(100*fraction_L,2))+'%' + '-----')


# print('\n# plurality-Black MM districts: '+str(plur_B))
# print('# plurality-Lat/Hisp MM districts: '+str(plur_L))
# print('# plurality-NH-white MM districts: '+str(plur_W))
# outFile.write('\n\n# plurality-Black MM districts: ' +str(plur_B))
# outFile.write('\n# plurality-Lat/Hisp MM districts: '+str(plur_L))
# outFile.write('\n# plurality-NH-white MM districts: '+str(plur_W))



# # Plot vote-share margins
# test = [dem_percent for dem_percent in initial_partition['AVG'].percents('Dem')]
# test.sort()
# fracRep = [1-d for d in test]

# # forLabels = [[fracDem[i],i] for i in range(1,len(fracDem[1:])+1)]
# # forLabels.sort()
# # labels = [FL[1] for FL in forLabels]
# # labels_str = [str(lab) for lab in labels]
# # #print(labels_str)

# x = np.arange(len(initial_partition['AVG'].percents('Dem')))
# x = [n+1 for n in x]
# y = [.5 for n in x]

# # # For MO House
# # plt.bar(x, test, width=1, color='blue', edgecolor='blue',label='Democrats')
# # plt.bar(x, fracRep, width=1, bottom=test, color='red', edgecolor='red',label='Republicans')
    
# # For not MO House
# plt.bar(x, test, color='blue', edgecolor='k',linewidth='.5',label='Democrats')
# plt.bar(x, fracRep, bottom=test, color='red', edgecolor='k',linewidth='.5',label='Republicans')

# #plt.xticks([i for i in range(1,len(initial_partition['AVG'].percents('Dem'))+1)])
# #plt.xticks([i for i in range(1,numDistricts)],labels_str)
# plt.yticks(np.arange(0, 1.1, step=0.1))
# plt.plot(x,y,'k-',linewidth='1')
# plt.legend(loc='lower right')
# plt.xlabel('Districts')
# plt.ylabel('Vote-share')
# #plt.savefig('Margins_'+outFile_name+'.pdf',dpi=600)
# plt.show()


# outFile.write('\n\n')
# outFile.close()
          

