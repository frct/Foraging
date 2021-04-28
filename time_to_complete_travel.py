

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