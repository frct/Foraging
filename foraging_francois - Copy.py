import os
# os.chdir("foraging_decay")

from imp import reload 
import import_pycontrol as ip

# change to location of raw data (.txt files)
behaviour_data_folder = '../../raw_data/behaviour_data/'


# create an experiment object with useful attributes 
experiment = ip.Experiment(behaviour_data_folder)

# attribute examples
experiment.sessions # list of sessions
experiment.subject_IDs # list of mouse IDs

# function examples
experiment.save() # save all sessions as a .pkl so don't have to re-import
experiment.get_sessions(subject_IDs='5', when='all') # get list of specified sessions 

# single session analyses
eg_session = experiment.sessions[0] # assign a single session object to a variable
eg_session.times # print list of events each with an array of times they happened
eg_session.events # print list of events. Each is a tuple with time and name
eg_session.patch_data # prints list of dicts, one for each patch, with key patch info
eg_session.trial_data # prints dict of trials (one trial = one reward) with key trial info

eg_session2 = ip.Session('../../raw_data/behaviour_data/fp01-2019-02-21-112604.txt')