import numpy as np
import pylab as plt
import matplotlib
from statistics import mean
from scipy import stats
from matplotlib.ticker import MaxNLocator
from itertools import groupby
import math
from operator import add
from matplotlib.pyplot import cm
from matplotlib.colors import ListedColormap

#----------------------------------------------------------------------------------
# General functions
#----------------------------------------------------------------------------------

def get_mean_travel(session):
    'Get the average travel times for each patch in a session'
    ses_travels = [patch['travel_time'] for patch in session.patch_data] # get actual travel times for each patch
    mean_travel = []
    for i,travel in enumerate(ses_travels): # travels jitter between 87.5% and 112.5% of average travel time 
        if 875 <= travel <= 1125:
            mean_travel.append(1000)
        elif 1750 <= travel <= 2250:
            mean_travel.append(2000)
        elif 3500 <= travel <= 4500:
            mean_travel.append(4000)
    return(mean_travel)
    
    
def get_column(matrix,i):
    'Return indexed column i'
    return [row[i] for row in matrix]


def no_patches(session):
    'Extract the number of patches an animal engaged in in a session'
    patch_no = [patch['number'] for patch in session.patch_data][-1]
    return(patch_no)
    
    
def no_patches_in_block(session):
    'Extract the number of patches an animal engaged in for each block in a session'
    block_nos = [patch['block_number'] for patch in session.patch_data]
    no_patches_per_block = [len(list(group)) for key, group in groupby(block_nos)]
    return(no_patches_per_block)
    

def no_rewards_in_patch(session):
    rewards_in_patch = [(len(patch['ave_time'])) for patch in session.patch_data if session.patch_data]
    return(rewards_in_patch)
    
def get_patches(session):
    'Extract patches - 0 left or 1 right'
    patches = [patch['patch'] for patch in session.patch_data]
    return(patches)
    
def get_patch_nos_in_block(session):
    patch_in_block = session.patch_in_block
    return(patch_in_block)
    
#----------------------------------------------------------------------------------
# Reward - SESSIONS
#----------------------------------------------------------------------------------
    
def rew_rate(session):
    'Extract normalised reward delivery counts per minute across a session'
    reward_times     = sorted(np.concatenate((session.times['reward_right'],session.times['reward_left']))) # times reward was collected
    reward_times_min = [time/60000 for time in reward_times] # convert ms to minutes
    
    bins = list(range(0,121,1)) # up to a max of 120 minutes (2 hours)
    
    reward_time_counts = np.histogram(reward_times_min,bins)[0] # in betweens
    sum_rew_counts     = np.sum(reward_time_counts) # sum counts to give total number of observations for normalisation
    norm_rew_counts    = np.asarray([time/sum_rew_counts for time in reward_time_counts]) 
    
    return(norm_rew_counts,bins)
    

def rew_to_rew_lat(session,rew_ind1,rew_ind2):
    '''Extract latencies from one reward cue to the next by reward in patch.
    rew_ind1 is the first reward index (e.g. 1 for first reward in patch, 2 
    for second reward, etc). rew_ind2 is the same but for the second reward index. 
    The latency between these will be returned.'''
    rew_cue_times = sorted(np.concatenate((session.times['reward_right_available'],session.times['reward_left_available']))) # times of reward cues
    rewards_in_patch = [(len(patch['ave_time'])) for patch in session.patch_data if session.patch_data]
    
    positions = [list(range(1,rewards_in_patch[i]+1)) for i,reward_no in enumerate(rewards_in_patch)] # numbered rewards in patch
    positions = [reward for sublist in positions for reward in sublist] # flatten
    
    rew_cue_times_by_rew_in_patch = list(zip(positions,rew_cue_times))
    
    # sometimes the animal only harvests one reward in a patch before travelling. Remove these instances
    rew_cue_times_by_rew_in_patch = [rew_cue_times_by_rew_in_patch[j] for j in range(len(
            rew_cue_times_by_rew_in_patch[:-1])) if rew_cue_times_by_rew_in_patch[j][0] != rew_cue_times_by_rew_in_patch[j+1][0]]
    
    # might be that the last patch the animal does, they only get one reward. Remove if so 
    if rew_cue_times_by_rew_in_patch[-1][0] == 1:
        rew_cue_times_by_rew_in_patch = rew_cue_times_by_rew_in_patch[:-1]
        
    lat_x_to_y = []
    for i, time in enumerate(rew_cue_times_by_rew_in_patch[:-1]):
        if rew_cue_times_by_rew_in_patch[i+1][0] == rew_ind2 and rew_cue_times_by_rew_in_patch[i][0] == rew_ind1:
            lat_first_to_second = (rew_cue_times_by_rew_in_patch[i+1][1] - rew_cue_times_by_rew_in_patch[i][1])/1000
            lat_x_to_y.append(lat_first_to_second)

    bins = list(np.linspace(0,10,50))
    
    latency_counts  = np.histogram(lat_x_to_y,bins)[0]
    sum_counts      = np.sum(latency_counts)
    norm_lat_counts = np.asarray([x/sum_counts for x in latency_counts])
    
    return(norm_lat_counts,bins)
    
    
def rew_to_harvest_lat(session,rew_ind1):
    'Extract latencies from reward collection to recommencing harvesting' 
    rew_and_harvest = [event for event in session.events if event.name in ['reward_left','reward_right','in_left','in_right']]
    # as wanting latencies of reward --> harvest, remove any initial harvest events 
    if rew_and_harvest[0].name == 'in_left' or rew_and_harvest[0].name =='in_right':
        first_rew_idx = [i for i,event in enumerate(rew_and_harvest) if rew_and_harvest[i].name =='reward_left' or rew_and_harvest[i].name =='reward_right'][0]
        rew_and_harvest = rew_and_harvest[first_rew_idx:-1]
    
    rewards_in_patch = [(len(patch['ave_time'])) for patch in session.patch_data if session.patch_data]
    
    positions = [list(range(1,rewards_in_patch[i]+1)) for i,reward_no in enumerate(rewards_in_patch)] # numbered rewards in patch
    positions = [reward for sublist in positions for reward in sublist] # flatten

    rews = sorted(np.concatenate((session.times['reward_right'],session.times['reward_left'])))
    first_harvests = [rew_and_harvest[j+1].time for j in range(len(rew_and_harvest[:-1])) if (rew_and_harvest[j].name == 'reward_left' or rew_and_harvest[j].name == 'reward_right')]
    
    # sometimes might get reward but then not harvest again
    if len(rews)>len(first_harvests):
        rews=rews[:-1]
        positions=positions[:-1]
        
    rews_by_rew_in_patch = list(zip(positions,rews))
        
    lat_x_to_y = []
    for i, time in enumerate(rews_by_rew_in_patch):
        if rews_by_rew_in_patch[i][0] == rew_ind1:
            lat_rew_to_harvest = (first_harvests[i] - rews_by_rew_in_patch[i][1])/1000
            lat_x_to_y.append(lat_rew_to_harvest)  
            
    bins = list(np.linspace(0,5,30))
            
    latency_counts  = np.histogram(lat_x_to_y,bins)[0]
    sum_counts      = np.sum(latency_counts)
    norm_reharvest_counts = np.asarray([x/sum_counts for x in latency_counts])
        
    
    #lat_to_reharvest = [(second_time - first_time)/1000 for (second_time,first_time) in zip(first_harvests,rews)] # latencies converted to seconds from ms on a log scale
    #lat_to_reharvest = np.log10([(second_time - first_time)/1000 for (second_time,first_time) in zip(first_harvests,rews)]) # latencies converted to seconds from ms on a log scale
    #np.isnan(lat_to_reharvest)
    
    
    return(norm_reharvest_counts,bins)

#----------------------------------------------------------------------------------
# Reward - EXPERMENTS
#----------------------------------------------------------------------------------

def patch_in_block_exp(experiment,subject_IDs='all',when='all'):
     if subject_IDs == 'all': subject_IDs = experiment.subject_IDs   
    
     subject_sessions = [experiment.get_sessions(subject_ID,when) for i, subject_ID in enumerate(subject_IDs)]
    
     patch_in_blocks = [[get_patch_nos_in_block(session) for session in subject] for subject in subject_sessions]
        
     return(patch_in_blocks)

def av_rew_rate(experiment,subject_IDs='all',when='all',plot='hist'):
    if subject_IDs == 'all': subject_IDs = experiment.subject_IDs   
    
    subject_sessions = [experiment.get_sessions(subject_ID,when) for i, subject_ID in enumerate(subject_IDs)]
    
    bin_edges = [[rew_rate(session)[1] for session in subject] for subject in subject_sessions][0][0]#.tolist()
    bin_centres = list(map(add, bin_edges[:-1], bin_edges[1:]))
    bin_centres = [x / 2 for x in bin_centres]
    
    norm_rew_counts  = [[rew_rate(session)[0] for session in subject] for subject in subject_sessions]
    sub_av_rew_times = [np.mean(norm_rew_counts[i],axis = 0).tolist() for i, subject in enumerate(norm_rew_counts)] # averaged across sessions for each subject
    
    # SEM
    sem_rew_times = stats.sem(sub_av_rew_times,axis=0)
    sem_rew_times = sem_rew_times.data.tolist()
    
    av_rew_times     = np.nanmean(sub_av_rew_times,axis=0).tolist() # averaged across all subjects
    cum_av_rew_times = np.cumsum(av_rew_times)
    
    if plot == 'hist':
        plt.figure()
        plt.bar(bin_centres,av_rew_times,yerr=sem_rew_times,color='g')
        plt.xlabel("Time (mins)")
        plt.ylabel("Normalised count")
        plt.title("Distribution of rewards across the session")
        
    elif plot == 'cumulative':
        plt.figure()
        plt.plot(bin_centres,cum_av_rew_times,color='g')
        plt.xlabel("Time (mins)")
        plt.ylabel("Cumulative normalised count")
        plt.title("Distribution of rewards across the session")
 
def av_rew_lat(experiment,rew_ind1=1,rew_ind2=2,latency='rew_to_rew',subject_IDs='all',when='all',plot='hist'):
    'Change latency to rew_to_rew for reward cue to reward cue latencies, or to rew_to_harvest for reward collection to re-harvesting latencies'
    if subject_IDs == 'all': subject_IDs = experiment.subject_IDs          
    
    subject_sessions = [experiment.get_sessions(subject_ID,when) for i, subject_ID in enumerate(subject_IDs)]
    
    if latency == 'rew_to_rew':
        bin_edges = [[rew_to_rew_lat(session,rew_ind1,rew_ind2)[1] for session in subject] for subject in subject_sessions][0][0]#.tolist()
    else:
        bin_edges = [[rew_to_harvest_lat(session,rew_ind1)[1] for session in subject] for subject in subject_sessions][0][0]#.tolist()
    
    bin_centres = list(map(add, bin_edges[:-1], bin_edges[1:]))
    bin_centres = [x / 2 for x in bin_centres]    
    
    if latency == 'rew_to_rew':
        latencies = [[rew_to_rew_lat(session,rew_ind1,rew_ind2)[0] for session in subject] for subject in subject_sessions]
    else:
        latencies = [[rew_to_harvest_lat(session,rew_ind1)[0] for session in subject] for subject in subject_sessions]
        
    sub_av_rew_to_rew_lat = [np.mean(latencies[i],axis = 0).tolist() for i, subject in enumerate(latencies)] # averaged across sessions for each subject
    
    # SEM
    sem_rew_to_rew = stats.sem(sub_av_rew_to_rew_lat,axis=0)
    sem_rew_to_rew = sem_rew_to_rew.data.tolist()        

    av_rew_to_rew  = np.nanmean(sub_av_rew_to_rew_lat,axis=0).tolist() # averaged across all subjects
    cum_rew_to_rew = np.cumsum(av_rew_to_rew)
    
   # x_tick = [0.1,1,10,100,1000]

    if plot == 'hist':
        plt.figure()
        plt.bar(bin_centres,av_rew_to_rew,width=0.1,yerr=sem_rew_to_rew,color='g')   
       # plt.xticks([-1,0,1,2,3],x_tick)
        plt.xlabel("Time (s)")
        plt.ylabel("Normalised count")
        plt.title("3rd reward in patch to harvesting")
        
    elif plot == 'cumulative':
        plt.figure()
        plt.plot(bin_centres,cum_rew_to_rew,color='g')   
        plt.xticks([-1,0,1,2],x_tick)
        plt.xlabel("Log time (s)")
        plt.ylabel("Likelihood")
        plt.title("Latency from reward collection to harvesting")        
                
#----------------------------------------------------------------------------------
# Time to complete travel
#----------------------------------------------------------------------------------

def time_to_complete_travel(session):
    'Plot how long it takes animals to complete travel each time they travel as a histogram, with x as a log scale'
    travel_events         = [event for event in session.events if event.name in ['travel','travel_complete','reward_left','reward_right']] # get times and names of every instance animal has commenced and completed travel
    #travel_events         = [event for event in travel_events if event.time < 3600000] # change this number to sample from different timepoints of the session. 360000 = one hour
    last_travel_completed = [a for a, completes in enumerate(travel_events) if travel_events[a].name == 'travel_complete'][-1]
    travel_events         = travel_events[0:last_travel_completed+1] # take everything up until the last completed travel
    # check that after each travel complete there is a reward that has been consumed. If not then only include the first one
    # if travel is followed by another travel, remove the repeated travel
    duplicate_indices = [i+1 for i,time in enumerate(travel_events[:-1]) if travel_events[i].name == 'travel' and (travel_events[i].name == travel_events[i+1].name)]
    for index in sorted(duplicate_indices,reverse=True):
        del travel_events[index] # remove all duplicated travels - these occur when animal has disengaged in the task
        
    # remove when animal completes travel, gets reward in patch - but does not CONSUME reward
    indices_travel_again = [i+1 for i, time in enumerate(travel_events[:-1]) if travel_events[i].name == 'travel_complete' and travel_events[i+1].name == 'travel']
    if indices_travel_again:
        for index in sorted(indices_travel_again,reverse=True):
            del travel_events[index] # deletes the extra travel
            del travel_events[index] # deletes the extra travel complete
            
    # remove reward events
    travel_events = [travel for i,travel in enumerate(travel_events) if travel_events[i].name == 'travel' or travel_events[i].name == 'travel_complete']
     
    travels = list(get_mean_travel(session)) # get travel times for each travel    
    time_to_travel = [(travel_events[l+1].time - travel_events[l].time)/1000 for l, event in enumerate(travel_events) if event.name == 'travel'] # calculate duration of travels
  
    #if len(time_to_travel) != len(travels): 
     #   travels = travels[:-1]

    times_to_travel_by_travel = [list(zip(travels,time_to_travel))]
    times_to_travel_by_travel = [item for sublist in times_to_travel_by_travel for item in sublist] # flatten by one

  #  short_times_to_travel = [travel[1] for travel in times_to_travel_by_travel if travel[0] == 1000]
  #  long_times_to_travel  = [travel[1] for travel in times_to_travel_by_travel if travel[0] == 4000]

#    less_than_4 = 0
#    for travel in long_times_to_travel:
#        if travel < 4:
#            less_than_4 += 1
#    
#    return(less_than_4)

    short_times_to_travel = np.log10([travel[1] for travel in times_to_travel_by_travel if travel[0] == 1000])
    long_times_to_travel  = np.log10([travel[1] for travel in times_to_travel_by_travel if travel[0] == 4000])
    long_times_to_travel = [travel for travel in long_times_to_travel if travel > 0.6018]

    bins = list(np.linspace(-1,4,40))
    
    long_time_counts   = np.histogram(long_times_to_travel,bins)[0] # get counts for each bin
    short_time_counts  = np.histogram(short_times_to_travel,bins)[0]
    
    sum_long_counts    = np.sum(long_time_counts) # sum counts to give total number of observations for normalisation
    sum_short_counts   = np.sum(short_time_counts)
    
    norm_long_counts   = np.asarray([x/sum_long_counts for x in long_time_counts]) # divide each count number by the total number of observations, turn into array
    norm_short_counts  = np.asarray([x/sum_short_counts for x in short_time_counts])
    
    return(norm_short_counts,norm_long_counts,bins)
    
def av_time_to_complete_travel(experiment,subject_IDs='all',when='all',plot='hist'):
    'Plot the average amount of time it takes for animals to complete travel'
    # Plot: 'hist' for histogram, 'cumulative' for cumulative frequency histogram
    
    if subject_IDs == 'all': subject_IDs = experiment.subject_IDs
    subject_sessions        = [experiment.get_sessions(subject_ID,when) for i, subject_ID in enumerate(subject_IDs)]
    
    bin_edges = [[time_to_complete_travel(session)[2] for session in subject] for subject in subject_sessions][0][0]#.tolist()
    bin_centres = list(map(add, bin_edges[:-1], bin_edges[1:]))
    bin_centres = [x / 2 for x in bin_centres]
    
    time_to_complete_short_travel = [[time_to_complete_travel(session)[0] for session in subject] for subject in subject_sessions]
    time_to_complete_long_travel  = [[time_to_complete_travel(session)[1] for session in subject] for subject in subject_sessions]
    
    # averaged across sessions for each subject
    sub_av_time_to_complete_short_travel = [np.mean(time_to_complete_short_travel[i],axis = 0).tolist() for i, subject in enumerate(time_to_complete_short_travel)]
    sub_av_time_to_complete_long_travel  = [np.mean(time_to_complete_long_travel[i],axis = 0).tolist() for i, subject in enumerate(time_to_complete_long_travel)]
    
    # SEMs
    sem_short_completes  = stats.sem(sub_av_time_to_complete_short_travel, axis = 0,nan_policy = 'omit')
    sem_short_completes  = sem_short_completes.data.tolist()
    sem_long_completes   = stats.sem(sub_av_time_to_complete_long_travel, axis = 0,nan_policy = 'omit')
    sem_long_completes   = sem_long_completes.data.tolist()
    
    # averaged across all subjects
    av_time_to_complete_short_travel = np.nanmean(sub_av_time_to_complete_short_travel, axis = 0).tolist()  
    av_time_to_complete_long_travel  = np.nanmean(sub_av_time_to_complete_long_travel, axis = 0).tolist()
    
    # cumulative
    cum_av_time_to_complete_short_travel  = np.cumsum(av_time_to_complete_short_travel)
    cum_av_time_to_complete_long_travel   = np.cumsum(av_time_to_complete_long_travel)
    
    av_time_to_complete_travel     = [av_time_to_complete_short_travel,av_time_to_complete_long_travel]# av_time_to_complete_med_travel
    cum_av_time_to_complete_travel = [cum_av_time_to_complete_short_travel,cum_av_time_to_complete_long_travel]#cum_av_time_to_complete_med_travel
    
    sems     = [sem_short_completes,sem_long_completes] #sem_med_completes
    
    x_tick = [0.1,1,10,100,1000,10000]
    travels = [1,4]#2
    
    if plot == 'hist':
        for k,travel in enumerate(av_time_to_complete_travel):
            plt.figure()
            plt.bar(bin_centres,av_time_to_complete_travel[k],width=0.1,yerr=sems[k],color='g')
            plt.xticks([-1,0,1,2,3,4],x_tick)
            plt.xlabel("Log time to complete travel(s)")
            plt.ylabel("Normalised count")
            plt.title("travel time %ss" %travels[k])
            
    elif plot == 'cumulative':
       # for k, travel in enumerate(cum_av_time_to_complete_travel):
       #     plt.figure()
            plt.plot(bin_centres,cum_av_time_to_complete_travel[1],color='g') # change index to 0 for short travel, 1 for long travel
            plt.xticks([-1,0,1,2,3,4],x_tick)
            plt.xlabel("Log time to complete travel(s)")
            plt.ylabel("Likelihood")
            plt.title("4s travel")
     #       plt.title("travel time %ss" %travels[k])
            
#----------------------------------------------------------------------------------
# Number of bouts of poking to complete travel
#----------------------------------------------------------------------------------

def travel_bouts(session):
    travel_in_out = [event for event in session.events if event.name in ['travel','travel_complete','travel_out_of_poke','travel_resumed']]
    travel_idxs   = [i for i,event in enumerate(travel_in_out) if event.name == 'travel']





#----------------------------------------------------------------------------------
# Effects of travel time or patch richness on number of rewards harvested per patch
#----------------------------------------------------------------------------------
    
def rew_harvested_travel(session,limit_block=False,limit_patch=False,limit_both=False):
    # If wanting to look at just the first X blocks
    if limit_block == True:
        session.patch_data_limited = [patch for patch in session.patch_data if patch['block_number'] <= 4]
        
    elif limit_patch == True:
        patch_in_block_no = [session.patch_data[k]['patch_in_block'] for k, patch in enumerate(session.patch_data)] # get number of each patch in block
        last_patch_in_block = [i-1 for i,idx in enumerate(patch_in_block_no) if idx ==1][1:] # get index of last patch in each block. This ignores the last (part) block that animals have done in a session
        patches_of_interest = [[patch-2,patch-1,patch] for patch in last_patch_in_block] # include patch-2 for last three patches
        patches_of_interest = [patch for sublist in patches_of_interest for patch in sublist] # flatten
        session.patch_data_limited = [patch for i,patch in enumerate(session.patch_data) if i in patches_of_interest]
        #session.patch_data_limited = [patch for patch in session.patch_data if patch['patch_in_block'] > 3] # this is for limiting by saying 'any patches beyond x patch be included'
        
    elif limit_both == True:   
        session.patch_data_limited = [patch for patch in session.patch_data if patch['block_number'] <= 4]
        session.patch_data_limited = [patch for patch in session.patch_data_limited if patch['patch_in_block'] > 3]
    
    else:
        session.patch_data_limited = session.patch_data
          
    rewards_in_patch_low  = [(len(patch['ave_time'])) for patch in session.patch_data_limited if 875  <= patch['travel_time'] <= 1125]
    rewards_in_patch_high = [(len(patch['ave_time'])) for patch in session.patch_data_limited if 3500 <= patch['travel_time'] <= 4500]
        
    return(rewards_in_patch_low,rewards_in_patch_high)
    
def rew_harvested_richness(session,limit_block=False,limit_patch=False):
    if limit_block == True:
        session.patch_data_limited = [patch for patch in session.patch_data if patch['block_number'] <= 4]
    
        
    else:
        session.patch_data_limited = session.patch_data

    rewards_in_patch_low  = [(len(patch['ave_time'])) for patch in session.patch_data_limited if patch['patch_richness'] == 1098.5]
    rewards_in_patch_med  = [(len(patch['ave_time'])) for patch in session.patch_data_limited if patch['patch_richness'] == 500]
    rewards_in_patch_high = [(len(patch['ave_time'])) for patch in session.patch_data_limited if 227 <= patch['patch_richness'] <= 228]
        
    return(rewards_in_patch_low,rewards_in_patch_med,rewards_in_patch_high)

def av_rew_harvested(experiment,subject_IDs='all',when='all',manipulation='travel', plot_figure = 'overall_average'):
    'Plot average number of rewards harvested for travel blocks or patch richnesses either for each subject or averaged across all subjects'
    
    # set manipulation to 'travel' for average rewards harvested per block
    # set manipulation to 'richness' for average rewards harvested per type of patch richness 
    
    if subject_IDs == 'all': subject_IDs = experiment.subject_IDs   
    
    subject_sessions   = [experiment.get_sessions(subject_ID,when) for i, subject_ID in enumerate(subject_IDs)]
    
    if manipulation == 'travel':
    
        rewards_in_patch_low =  [[rew_harvested_travel(session)[0] for session in subject] for subject in subject_sessions]
        rewards_in_patch_high = [[rew_harvested_travel(session)[1] for session in subject] for subject in subject_sessions]
        
        rewards_in_patch_low  = [list(filter(None, subject)) for subject in rewards_in_patch_low] # remove empty lists
        rewards_in_patch_high = [list(filter(None, subject)) for subject in rewards_in_patch_high]
        
        # calculating averages  
        av_rewards_in_patch_low_session = [[mean(session) for session in subject] for subject in rewards_in_patch_low]
        av_rewards_in_patch_low_subject = [mean(subject) for subject in av_rewards_in_patch_low_session]
        
        av_rewards_in_patch_high_session  = [[mean(session) for session in subject] for subject in rewards_in_patch_high]
        av_rewards_in_patch_high_subject  = [mean(subject) for subject in av_rewards_in_patch_high_session]
        
        av_rewards_in_patch_low  = mean(av_rewards_in_patch_low_subject)
        av_rewards_in_patch_high = mean(av_rewards_in_patch_high_subject)
        
        sem_low_rewards_in_patch  = stats.sem(av_rewards_in_patch_low_subject)
        sem_high_rewards_in_patch = stats.sem(av_rewards_in_patch_high_subject)
        
        sem_low_rewards_in_patch_subject = [stats.sem(av_rewards_in_patch_low_session[k]) for k,session in enumerate(av_rewards_in_patch_low_session)] # SEMs for each subject
        sem_high_rewards_in_patch_subject  = [stats.sem(av_rewards_in_patch_high_session[m]) for m,session in enumerate(av_rewards_in_patch_high_session)]
            
    elif manipulation == 'richness':
        
        rewards_in_patch_low =  [[rew_harvested_richness(session)[0] for session in subject] for subject in subject_sessions]
        rewards_in_patch_med =  [[rew_harvested_richness(session)[1] for session in subject] for subject in subject_sessions]
        rewards_in_patch_high = [[rew_harvested_richness(session)[2] for session in subject] for subject in subject_sessions]        
        
                
        rewards_in_patch_low  = [list(filter(None, subject)) for subject in rewards_in_patch_low] # remove empty lists
        rewards_in_patch_med = [list(filter(None, subject)) for subject in rewards_in_patch_med]
        rewards_in_patch_high = [list(filter(None, subject)) for subject in rewards_in_patch_high]
        
        # calculating averages  
        av_rewards_in_patch_low_session = [[mean(session) for session in subject] for subject in rewards_in_patch_low]
        av_rewards_in_patch_low_subject = [mean(subject) for subject in av_rewards_in_patch_low_session]
        
        av_rewards_in_patch_med_session = [[mean(session) for session in subject] for subject in rewards_in_patch_med]
        av_rewards_in_patch_med_subject = [mean(subject) for subject in av_rewards_in_patch_med_session]
        
        av_rewards_in_patch_high_session  = [[mean(session) for session in subject] for subject in rewards_in_patch_high]
        av_rewards_in_patch_high_subject  = [mean(subject) for subject in av_rewards_in_patch_high_session]
        
        av_rewards_in_patch_low  = mean(av_rewards_in_patch_low_subject)
        av_rewards_in_patch_med = mean(av_rewards_in_patch_med_subject)
        av_rewards_in_patch_high = mean(av_rewards_in_patch_high_subject)
        
        sem_low_rewards_in_patch  = stats.sem(av_rewards_in_patch_low_subject)
        sem_med_rewards_in_patch = stats.sem(av_rewards_in_patch_med_subject)
        sem_high_rewards_in_patch = stats.sem(av_rewards_in_patch_high_subject)
        
        sem_low_rewards_in_patch_subject = [stats.sem(av_rewards_in_patch_low_session[k]) for k,session in enumerate(av_rewards_in_patch_low_session)] # SEMs for each subject
        sem_med_rewards_in_patch_subject  = [stats.sem(av_rewards_in_patch_med_session[m]) for m,session in enumerate(av_rewards_in_patch_med_session)]
        sem_high_rewards_in_patch_subject  = [stats.sem(av_rewards_in_patch_high_session[m]) for m,session in enumerate(av_rewards_in_patch_high_session)]
        
    summerBig = cm.get_cmap('Oranges', 512)
    newcmp = ListedColormap(summerBig(np.linspace(0.4, 1.0, 256)))

    if plot_figure == 'overall_average':
        if manipulation == 'travel':
            
            fig = plt.figure(figsize=(3,5))
            ax = fig.add_subplot(111)
            
            
            y_axis   = [av_rewards_in_patch_high,av_rewards_in_patch_low] 
            y_error  = [sem_high_rewards_in_patch,sem_low_rewards_in_patch]
            x_axis   = [1,2]
            x_labels = ['4s','1s']#2s

            color=iter(newcmp(np.linspace(0,1,len(y_axis))))
            for i,travel in enumerate(y_axis):
                c = next(color)
                ax.bar(x_axis[i],y_axis[i],yerr=y_error[i],color=c)
                
            ax.set_ylim([4,7])
            ax.set_ylabel("Mean no. rewards",fontsize=14)
            ax.set_xlabel("Travel time")
            ax.set_xticks(x_axis)
            y_labels = [4,4.5,5,5.5,6,6.5,7]
            ax.set_yticklabels(y_labels,fontsize=14)
            ax.set_xticklabels(x_labels,fontsize=12)
         #   plt.title("Travel on rew harvested/patch - last 3 patches")
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            for axis in ['bottom','left']:
                ax.spines[axis].set_linewidth(2)
            [t.set_visible(False) for t in ax.get_xticklines()]
            [t.set_visible(False) for t in ax.get_yticklines()]      
            fig.tight_layout()
            
        elif manipulation == 'richness':
            
            fig = plt.figure(figsize=(4,5))
            ax = fig.add_subplot(111)
            
            y_axis   = [av_rewards_in_patch_low,av_rewards_in_patch_med,av_rewards_in_patch_high]
            y_error  = [sem_low_rewards_in_patch,sem_med_rewards_in_patch,sem_high_rewards_in_patch]
            x_axis   = [1,2,3]
            x_labels = ['Poor','Medium','Rich']
            
            color=iter(newcmp(np.linspace(0,1,len(y_axis))))
            
            for i,patch_type in enumerate(y_axis):
                c=next(color)
                ax.bar(x_axis[i],y_axis[i],yerr = y_error[i],color=c) #tick_label=x_labels[i]
          #  plt.ylim([5,8])
            ax.set_ylabel("Mean no. rewards",fontsize=14)
            ax.set_xlabel("Patch richness",fontsize=14)
        #    plt.title("Effect of patch richness on rewards harvested/patch")
    #        ax.tick_params(labelsize=12)
            ax.set_xticks(x_axis)
            y_labels = [0,2,4,6,8]
            ax.set_yticklabels(y_labels,fontsize=14)
            ax.set_xticklabels(x_labels,fontsize=12)
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            for axis in ['bottom','left']:
                ax.spines[axis].set_linewidth(2)
            [t.set_visible(False) for t in ax.get_xticklines()]
            [t.set_visible(False) for t in ax.get_yticklines()]      
            fig.tight_layout()
            

            
            
            
            
            
            
            
            
            
            
    elif plot_figure == 'subject_average':
        if manipulation == 'richness':
        
            for i,subject in enumerate(subject_IDs):
                plt.figure(figsize=(4.1,4.8))
                y_axis = [av_rewards_in_patch_low_subject[i],av_rewards_in_patch_med_subject[i],av_rewards_in_patch_high_subject[i]]
                y_error = [sem_low_rewards_in_patch_subject[i],sem_med_rewards_in_patch_subject[i],sem_high_rewards_in_patch_subject[i]]
                x_axis = [1,2,3]
                x_labels = ['Poor','Medium','Rich']
                
                plt.bar(x_axis,y_axis,yerr = y_error,tick_label=x_labels)
                plt.ylim([1,10])
                plt.ylabel("Mean rewards harvested per patch")
                plt.xlabel("Mean patch richness")
                plt.title('mouse %s' %subject_IDs[i])
                plt.tight_layout
                
    return(av_rewards_in_patch_low_subject,av_rewards_in_patch_high_subject)
    
#----------------------------------------------------------------------------------
# Effects of patch richness on average harvest time at leaving patch
#----------------------------------------------------------------------------------   
    
    # patch richnesses are: init mean forage time of 227s, 500s, or 1098s

def harvest_time_travel(session,limit_block=False,limit_patch=False):
    # If wanting to look at just the first X blocks
    if limit_block == True:
        session.patch_data_limited = [patch for patch in session.patch_data if patch['block_number'] <= 4]
        
    # If wanting to look at a subset of patches in block  
    if limit_patch == True:
        patch_in_block_no = [session.patch_data[k]['patch_in_block'] for k, patch in enumerate(session.patch_data)] # get number of each patch in block
        last_patch_in_block = [i-1 for i,idx in enumerate(patch_in_block_no) if idx ==1][1:]
        patches_of_interest = [[patch-2,patch-1,patch] for patch in last_patch_in_block] 
        patches_of_interest = [patch for sublist in patches_of_interest for patch in sublist] # flatten
        session.patch_data_limited = [patch for i,patch in enumerate(session.patch_data) if i in patches_of_interest]
        #session.patch_data_limited = [patch for patch in session.patch_data if patch['patch_in_block'] > 3]
    else:
        session.patch_data_limited = session.patch_data
        
    harvest_time_low  = [patch['ave_time'][-1] for patch in session.patch_data_limited if 875  <= patch['travel_time'] <= 1125]
    harvest_time_high = [patch['ave_time'][-1] for patch in session.patch_data_limited if 3500 <= patch['travel_time'] <= 4500]
        
    return(harvest_time_low,harvest_time_high)
    
def harvest_time_richness(session,limit_block=False,limit_patch=False):
    if limit_block == True:
        session.patch_data_limited = [patch for patch in session.patch_data if patch['block_number'] <= 4]

    harvest_time_low  = [patch['ave_time'][-1] for patch in session.patch_data_limited if patch['patch_richness'] == 1098.5]
    harvest_time_med  = [patch['ave_time'][-1] for patch in session.patch_data_limited if patch['patch_richness'] == 500]
    harvest_time_high = [patch['ave_time'][-1] for patch in session.patch_data_limited if 227 <= patch['patch_richness'] <= 228]
        
    return(harvest_time_low,harvest_time_med,harvest_time_high)
  
def av_harvest_time_patch_richness(experiment,subject_IDs='all',when='all',plot_figure ='overall_average',manipulation='richness'):
    'Plot average harvest time in patch for each patch richness either for each subject or averaged across all subjects'
    if subject_IDs == 'all': subject_IDs = experiment.subject_IDs   
    
    subject_sessions  = [experiment.get_sessions(subject_ID,when) for i, subject_ID in enumerate(subject_IDs)]
    
    # IF WANTING TO LOOK AT TRAVEL, CHANGE HARVEST_TIME_RICHNESS TO HARVEST_TIME_TRAVEL
    # in future change the code to make this easier...
    harvest_time_low  = [[harvest_time_travel(session)[0] for session in subject] for subject in subject_sessions]
    harvest_time_med  = [[harvest_time_travel(session)[1] for session in subject] for subject in subject_sessions]
  #  harvest_time_high = [[harvest_time_travel(session)[2] for session in subject] for subject in subject_sessions]

    harvest_time_low  = [list(filter(None, subject)) for subject in harvest_time_low] # remove empty list
    harvest_time_med  = [list(filter(None, subject)) for subject in harvest_time_med] # remove empty list
 #   harvest_time_high = [list(filter(None, subject)) for subject in harvest_time_high]
     
    # calculating averages 
    
    av_harvest_time_low_session  = [[mean(session) for session in subject] for subject in harvest_time_low]
    av_harvest_time_low_subject  = [mean(subject) for subject in av_harvest_time_low_session]
    
    av_harvest_time_med_session  = [[mean(session) for session in subject] for subject in harvest_time_med]
    av_harvest_time_med_subject  = [mean(subject) for subject in av_harvest_time_med_session]
    
  ##  av_harvest_time_high_session  = [[mean(session) for session in subject] for subject in harvest_time_high]
  #  av_harvest_time_high_subject  = [mean(subject) for subject in av_harvest_time_high_session]
    
    av_harvest_time_low = mean(av_harvest_time_low_subject)
    av_harvest_time_med = mean(av_harvest_time_med_subject)
  #  av_harvest_time_high = mean(av_harvest_time_high_subject)

    
    sem_harvest_time_low = stats.sem(av_harvest_time_low_subject)
    sem_harvest_time_med = stats.sem(av_harvest_time_med_subject)
   # sem_harvest_time_high  = stats.sem(av_harvest_time_high_subject)
    
    sem_harvest_time_low_subject = [stats.sem(av_harvest_time_low_session [k]) for k,session in enumerate(av_harvest_time_low_session)] # SEMs for each subject
    sem_harvest_time_med_subject = [stats.sem(av_harvest_time_med_session [k]) for k,session in enumerate(av_harvest_time_med_session)] # SEMs for each subject
  #  sem_harvest_time_high_subject = [stats.sem(av_harvest_time_high_session [k]) for k,session in enumerate(av_harvest_time_high_session)] 

    # plot average harvest time at leaving for each patch richness
    
    summerBig = cm.get_cmap('Oranges', 512)
    newcmp = ListedColormap(summerBig(np.linspace(0.4, 1.0, 256)))
    
    if plot_figure == 'overall_average':
        if manipulation == 'travel':
        
            fig = plt.figure(figsize=(3,5))
            ax = fig.add_subplot(111)
        
            y_axis   = [av_harvest_time_med, av_harvest_time_low]#av_harvest_time_med
            y_error  = [sem_harvest_time_med, sem_harvest_time_low]#sem_harvest_time_med
            x_axis   = [1,2]
            x_labels = ['4s','1s']
            
            color=iter(newcmp(np.linspace(0,1,len(y_axis))))
                
            for i,patch_type in enumerate(y_axis):
                c=next(color)
                ax.bar(x_axis[i],y_axis[i],yerr = y_error[i],color=c) #tick_label=x_labels[i]
            ax.set_ylabel("Mean harvest time (ms)",fontsize=14)
            ax.set_xlabel("Travel time",fontsize=14)        
            ax.set_xticks(x_axis)       
        
            y_labels = [0,500,1000,1500,2000,2500]
            ax.set_yticklabels(y_labels,fontsize=14)
            ax.set_xticklabels(x_labels,fontsize=12)     
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            for axis in ['bottom','left']:
                ax.spines[axis].set_linewidth(2)
            [t.set_visible(False) for t in ax.get_xticklines()]
            [t.set_visible(False) for t in ax.get_yticklines()]      
            fig.tight_layout()
            

            
            
        
        elif manipulation == 'richness':
            
            fig = plt.figure(figsize=(4,5))
            ax = fig.add_subplot(111)
            
            y_axis   = [av_harvest_time_low, av_harvest_time_med,av_harvest_time_high]#av_harvest_time_med
            y_error  = [sem_harvest_time_low, sem_harvest_time_med,sem_harvest_time_high]#sem_harvest_time_med
            x_axis   = [1,2,3]
            x_labels = ['Poor','Medium','Rich']
            
            
            color=iter(newcmp(np.linspace(0,1,len(y_axis))))
            
            for i,patch_type in enumerate(y_axis):
                c=next(color)
                ax.bar(x_axis[i],y_axis[i],yerr = y_error[i],color=c) #tick_label=x_labels[i]
            ax.set_ylabel("Mean harvest time (ms)",fontsize=14)
            ax.set_xlabel("Patch richness",fontsize=14)
            ax.set_xticks(x_axis)
            y_labels = [0,500,1000,1500,2000,2500]
            ax.set_yticklabels(y_labels,fontsize=14)
            ax.set_xticklabels(x_labels,fontsize=12)
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            for axis in ['bottom','left']:
                ax.spines[axis].set_linewidth(2)
            [t.set_visible(False) for t in ax.get_xticklines()]
            [t.set_visible(False) for t in ax.get_yticklines()]      
            fig.tight_layout()
        
    #        plt.bar(x_axis,y_axis,yerr = y_error,tick_label=x_labels)

        #    plt.title("Effect of patch richness on rewards harvested/patch")
    #        ax.tick_params(labelsize=12)


            
            
            
            
            
            
            
        
    elif plot_figure == 'subject_average':
        if manipulation == 'travel':
        
            for i,subject in enumerate(subject_IDs):
                plt.figure(figsize=(4,4.8))
                y_axis = [av_harvest_time_low_subject[i],av_harvest_time_high_subject[i]]#av_harvest_time_med_subject[i]
                y_error = [sem_harvest_time_low_subject[i],sem_harvest_time_high_subject[i]]#sem_harvest_time_med_subject[i]
                x_axis = [1,2]
             #   plt.ylim([1000,5000])
                x_labels = ['1s','4s']
                
                plt.bar(x_axis,y_axis,yerr = y_error,tick_label=x_labels)
                plt.ylabel("Mean harvest time at leaving (ms)")
                plt.xlabel("Mean travel")
                plt.title('Travel on leaving time - mouse %s' %subject_IDs[i])
                
        elif manipulation == 'richness':
            
            for i,subject in enumerate(subject_IDs):
                plt.figure(figsize=(4,4.8))
                y_axis = [av_harvest_time_low_subject[i],av_harvest_time_high_subject[i]]#av_harvest_time_med_subject[i]
                y_error = [sem_harvest_time_low_subject[i],sem_harvest_time_high_subject[i]]#sem_harvest_time_med_subject[i]
                x_axis = [1,2]
             #   plt.ylim([1000,5000])
                x_labels = ['Poor','Rich']
                
                plt.bar(x_axis,y_axis,yerr = y_error,tick_label=x_labels)
                plt.ylabel("Mean harvest time at leaving (ms)")
                plt.xlabel("Mean patch richness")
                plt.title('Richness on leaving time - mouse %s' %subject_IDs[i])
    
    return(av_harvest_time_low_subject,av_harvest_time_med_subject)#av_harvest_time_med_subject