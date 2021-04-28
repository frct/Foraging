import os
import re
import pickle
import numpy as np
from datetime import datetime, date
from collections import namedtuple
#from . import plotting_basic as pl

Event = namedtuple('Event', ['time','name'])

#----------------------------------------------------------------------------------
# Session class
#----------------------------------------------------------------------------------

class Session():
    '''Import data from a pyControl file and represent it as an object with attributes:
      - file_name
      - experiment_name
      - task_name
      - subject_ID
          If argument int_subject_IDs is True, suject_ID is stored as an integer,
          otherwise subject_ID is stored as a string.
      - datetime
          The date and time that the session started stored as a datetime object.
      - datetime_string
          The date and time that the session started stored as a string of format 'YYYY-MM-DD HH:MM:SS'
      - events
          A list of all framework events and state entries in the order they occured. 
          Each entry is a namedtuple with fields 'time' & 'name', such that you can get the 
          name and time of event/state entry x with x.name and x.time respectively/
      - times
          A dictionary with keys that are the names of the framework events and states and 
          corresponding values which are Numpy arrays of all the times (in milliseconds since the
           start of the framework run) at which each event occured.
      - print_lines
          A list of all the lines output by print statements during the framework run, each line starts 
          with the time in milliseconds at which it was printed.
    '''

    def __init__(self, file_path, int_subject_IDs=True):

        # Load lines from file.

        with open(file_path, 'r') as f:
            print('Importing data file: '+os.path.split(file_path)[1])
            all_lines = [line.strip() for line in f.readlines() if line.strip()]

        # Extract and store session information.

        self.file_name = os.path.split(file_path)[1]

        info_lines = [line[2:] for line in all_lines if line[0]=='I']

        self.experiment_name = next(line for line in info_lines if 'Experiment name' in line).split(' : ')[1]
        self.task_name       = next(line for line in info_lines if 'Task name'       in line).split(' : ')[1]
        subject_ID_string    = next(line for line in info_lines if 'Subject ID'      in line).split(' : ')[1]
        datetime_string      = next(line for line in info_lines if 'Start date'      in line).split(' : ')[1]

        if int_subject_IDs: # Convert subject ID string to integer.
            self.subject_ID = int(''.join([i for i in subject_ID_string if i.isdigit()]))
        else:
            self.subject_ID = subject_ID_string

        self.datetime = datetime.strptime(datetime_string, '%Y/%m/%d %H:%M:%S')
        self.datetime_string = self.datetime.strftime('%Y-%m-%d %H:%M:%S')

        # Extract and store session data.

        state_IDs = eval(next(line for line in all_lines if line[0]=='S')[2:])
        event_IDs = eval(next(line for line in all_lines if line[0]=='E')[2:])

        ID2name = {v: k for k, v in {**state_IDs, **event_IDs}.items()}

        data_lines = [line[2:].split(' ') for line in all_lines if line[0]=='D']

        self.events = [Event(int(dl[0]), ID2name[int(dl[1])]) for dl in data_lines]
        self.times  = {event_name: np.array([ev.time for ev in self.events if ev.name == event_name]) for event_name in ID2name.values()}

        print_lines = [line[2:].split(' ',1) for line in all_lines if line[0]=='P'] 
        
        self.block_times = [line for line in print_lines if 'TT:' in line[1]]

        # Make trial data dictionary
        
        trial_lines = [line[1] for line in print_lines if 'P:' in line[1]] # Lines with trial data.
        block_lines = [line[1] for line in print_lines if 'TT:' in line[1]]
        trial_lines_no_travel = [line[1] for line in print_lines if 'PB#' in line[1]]
        remaining_patches_lines = [line[1] for line in print_lines if 'remaining' in line[1]]

        trial_lines = [tl.replace(",","") for tl in trial_lines] #remove comma in scripts where there's a typo
        trial_lines = [line for line in trial_lines if 'RP#:0' not in line] # if animal completes travel but does not CONSUME reward in patch, then completes travel again - remove second instance 
      #  trial_lines_corrected = [line for i,line in enumerate(trial_lines[:-1]) if 'RP#:0' not in trial_lines[i+1]] # if animal completes travel, engages in the patch, but does not receive reward then travels again - remove first instance 
      #  trial_lines_corrected.append(trial_lines[-1])
        
        travel_lines_with_repeats = [line for line in trial_lines if '-1' in line]
        self.travel_lines_with_repeats = travel_lines_with_repeats
    #    trial_lines = trial_lines_corrected
            

# sometimes the animal completes travel, DOES engage in the patch, but does not harvest for long enough to actually get a reward in the patch, then disengages making 
        # travel available again...

        if len(trial_lines) > 0:
            patch             = np.array([int(_str_get(tl,'P:', ' ')) for tl in trial_lines]) # Which patch reward was obtained in, 0, 1, (-1 for travel).
            patch_nos         = np.array([int(_str_get(tl,'P#:', ' ')) for tl in trial_lines]) # Patch numbers
            forage_time       = np.array([int(_str_get(tl,'T:', ' ')) for tl in trial_lines]) # Specific foraging time for reward, specific travel time required for travel
            ave_time          = np.array([int(_str_get(tl,'AFT:',' ')) for tl in trial_lines]) # Average of distribution foraging time was drawn from for reward, giving up time of the previous patch for travel.
            patch_richness    = np.array([float(_str_get(tl,'IFT:',  )) for tl in trial_lines]) # Initial average of distribution - indication of patch richness 
            block_travel      = np.array([int(_str_get(bl,'TT:'  )) for bl in block_lines]) # Travels for each of the blocks
            patch_in_block    = np.array([int(_str_get(tlnt,'PB#:',' ')) for tlnt in trial_lines_no_travel]) # Patch number in block      
            
            # correcting patch_in_block if an animal completes travel but does not engage in the patch - this will be off by 1...
            patch_in_block = np.array([patch_in_block[n] for n in range(len(patch_in_block[:-1])) if patch_in_block[n] != patch_in_block[n+1]]+[patch_in_block[-1]]) # remove repeats from list
            patch_in_block_diffs = [patch_in_block[j+1] - patch_in_block[j] for j, patch_no in enumerate(patch_in_block[:-1])]
            new_block_idx = [k+1 for k, no in enumerate(patch_in_block_diffs) if no < 0] # get indices of negative numbers
            patch_in_block_split = np.split(patch_in_block,new_block_idx) # split patch numbers into blocks
            patch_in_block_split = [np.insert(block,0,0) for block in patch_in_block_split] # add a zero to the beginning of each block so that first patch can be included in comparisons

            # correct any arrays that have skipped patches
            for block in patch_in_block_split:
                for idx, patch_no in enumerate(block[1:]):
                    if block[idx+1] - block[idx] > 1:
                        block[idx+1] = block[idx]+1
                        
            # remove zeros
            patch_in_block_split = [patches[1:] for patches in patch_in_block_split]
                        
            # put arrays back together again
            patch_in_block = np.concatenate(patch_in_block_split)
            
            # similar correction but for patch numbers 
            # find index of when a patch number only occurs once and lower it and all consecutive patch numbers by 1
#            no_occurrences = np.unique(patch_nos,return_counts=True)[1]
#            numbers = np.unique(patch_nos)
#            single_occurrence = [numbers[i] for i,occurrence in enumerate(no_occurrences) if occurrence == 1]
#            if single_occurrence:
#                for occurrence in single_occurrence:
#                    idx_single_occurrence = [m for m, number in enumerate(patch_nos) if number == occurrence]
#                    for idx, patch_number in enumerate(patch_nos):
#                        for occurrence in idx_single_occurrence:
#                            if idx >= occurrence:
#                                patch_nos[idx] = patch_nos[idx]-1

            # May be redundant now...
            for idx, patch_number in enumerate(patch_nos[1:]):
                if patch_nos[idx+1] - patch_nos[idx] > 1:
                    patch_nos[idx+1:] = patch_nos[idx+1:]-1
            
            self.trial_data = {'patch': patch, 'forage_time': forage_time, 'ave_time':ave_time, 'patch_richness': patch_richness, 'travel_block': block_travel, 'patch_in_block':patch_in_block}

            self.n_rewards = int(_str_get(trial_lines[-1],'R#:', ' '))
            self.n_patches = int(patch_nos[-1])
            
            n_blocks = 0
            for no in patch_in_block:
                if no == 1:
                    n_blocks+=1
            
            self.n_blocks  = n_blocks
            
            self.patch_in_block = patch_in_block

            # Because of animals disengaging from the task and then travelling again sometimes travel repeats. 
            # Find where this happens and remove from patch, forage_time, ave_time, patch_richness
            
            travels = np.where(patch==-1)[0]  #indices of travels 
            travel_diffs = [0] + [travel - travels[i-1] for i, travel in enumerate(travels)][1:] 
            travel_diffs_indices = [i for i, diffs in enumerate(travel_diffs) if diffs != 1] # indices to be kept - where the difference isn't 1
            travels_to_be_removed = np.delete(travels,travel_diffs_indices)
            patch = np.delete(patch,travels_to_be_removed)
            travels = np.where(patch==-1)[0]
            
            forage_time    = np.delete(forage_time,travels_to_be_removed)
            ave_time       = np.delete(ave_time,travels_to_be_removed)
            patch_richness = np.delete(patch_richness,travels_to_be_removed)

            # Make patch data.
            
            if patch[-1] == -1: # last trial data line is a travel.
                patch_starts   = np.hstack([0,travels[:-1]+1]) 
                patch_ends     = travels
                give_up_times  = ave_time[travels] #when animals gave up harvesting and decided to travel instead
                travel_times   = forage_time[travels] #travel time required
                patch_richness = patch_richness[travels]
                patch_in_block = patch_in_block
            else: # Last trial data line is a reward.
                patch_starts = np.hstack([0,travels+1])
                patch_ends   = np.hstack([travels, len(patch)])
                give_up_times = np.hstack([ave_time[travels],np.nan])
                travel_times = np.hstack([forage_time[travels],np.nan])
                patch_richness = np.hstack([patch_richness[travels],np.nan])
                patch_in_block = np.hstack([patch_in_block])
            
            # to assign block number to patches, multiply number of patches in block by block number
            patches_per_block = [block[-1] for block in patch_in_block_split]
            block_numbers = [[i+1]*patches_per_block[i] for i, block in enumerate(patches_per_block)]
            block_numbers = [patch for block in block_numbers for patch in block] #flatten

            self.patch_data = [{'patch':patch[s],'block_number':block_numbers[i],'forage_time':forage_time[s:e], 
                                'ave_time':ave_time[s:e], 'number': i+1,
                                'give_up_time':give_up_times[i], 'travel_time':travel_times[i],
                                'patch_richness':patch_richness[i],'patch_in_block':patch_in_block[i]}
                                for i,(s,e) in enumerate(zip(patch_starts, patch_ends))]

        else:
            self.trial_data, self.patch_data = (None, None)

    def plot(self, fig_no=1):
        pl.session_plot(self)



def _str_get(s,start='^',end='$'):
    '''Return chunk of string s between specified start and end sequence.'''
    return re.search(start+'(.*?)'+end, s).group(1)

#----------------------------------------------------------------------------------
# Experiment class
#----------------------------------------------------------------------------------

class Experiment():
    def __init__(self, folder_path, int_subject_IDs=True, rebuild_sessions=False,import_sessions=True,prev_sessions=None):
        '''
        Set import_sessions to false if you have already imported the data
        previously and have it saved as a variable name (for example, when
        importing behaviour and photometry data together). In this case,
        instead set the folder_path variable to be named as the sessions
        variable. 
        '''

        self.folder_name = os.path.split(folder_path)[1]
        self.path = folder_path

        # Import sessions.

        if import_sessions:
            self.sessions = []
            if not rebuild_sessions:
                try: # Load sessions from saved sessions.pkl file.
                    with open(os.path.join(self.path, 'sessions.pkl'),'rb') as sessions_file:
                        self.sessions = pickle.load(sessions_file)
                    print('Saved sessions loaded from: sessions.pkl')
                except IOError:
                   pass

            old_files = [session.file_name for session in self.sessions]
            files = os.listdir(self.path)
            new_files = [f for f in files if f[-4:] == '.txt' and f not in old_files]
    
            if len(new_files) > 0:
                print('Loading new data files..')
                for file_name in new_files:
                    try:
                        self.sessions.append(Session(os.path.join(self.path, file_name), int_subject_IDs))
                    except Exception as error_message:
                        print('Unable to import file: ' + file_name)
                        print(error_message)
                        
        else:
            self.sessions = prev_sessions

        # Assign session numbers.

        self.subject_IDs = list(set([s.subject_ID for s in self.sessions]))
        self.n_subjects = len(self.subject_IDs)

        self.sessions.sort(key = lambda s:s.datetime_string + str(s.subject_ID))
        
        self.sessions_per_subject = {}
        for subject_ID in self.subject_IDs:
            subject_sessions = self.get_sessions(subject_ID)
            for i, session in enumerate(subject_sessions):
                session.number = i+1
            self.sessions_per_subject[subject_ID] = subject_sessions[-1].number

    def save(self):
        '''Save all sessions as .pkl file.'''
        with open(os.path.join(self.path, 'sessions.pkl'),'wb') as sessions_file:
            pickle.dump(self.sessions, sessions_file)
        
    def get_sessions(self, subject_IDs='all', when='all'):
        '''Return list of sessions which match specified subject ID and time range.
        '''
        if subject_IDs == 'all':
            subject_IDs = self.subject_IDs
        if not isinstance(subject_IDs, list):
            subject_IDs = [subject_IDs]

        if when == 'all': # Select all sessions.
            when_func = lambda session: True

        elif isinstance(when, int):
            if when < 0: # Select most recent 'when' sessions.
                when_func = lambda session: (session.number > 
                    self.sessions_per_subject[session.subject_ID] + when)
            else: 
                when_func = lambda session: session.number == when

        elif type(when) in (str, datetime, date): # Select specified date.
            when_func = lambda session: session.datetime.date() == _toDate(when)

        elif ... in when: # Select a range..

            if len(when) == 3:  # Start and end points defined.
                assert type(when[0]) == type(when[2]), 'Start and end of time range must be same type.'
                if type(when[0]) == int: # .. range of session numbers.
                    when_func = lambda session: when[0] <= session.number <= when[2]
                else: # .. range of dates.
                    when_func = lambda session: _toDate(when[0]) <= session.datetime.date() <= _toDate(when[2])
            
            elif when.index(...) == 0: # End point only defined.
                if type(when[1]) == int: # .. range of session numbers.
                    when_func = lambda session: session.number <= when[1]
                else: # .. range of dates.
                    when_func = lambda session: session.datetime.date() <= _toDate(when[1])

            else: # Start point only defined.
                if type(when[0]) == int: # .. range of session numbers.
                    when_func = lambda session: when[0] <= session.number
                else: # .. range of dates.
                    when_func = lambda session: _toDate(when[0]) <= session.datetime.date()
            
        else: # Select specified..
            assert all([type(when[0]) == type(w) for w in when]), "All elements of 'when' must be same type."
            if type(when[0]) == int: # .. session numbers.
                when_func = lambda session: session.number in when
            else: # .. dates.
                dates = [_toDate(d) for d in when]
                when_func = lambda session: session.datetime.date() in dates

        valid_sessions = [s for s in self.sessions if s.subject_ID in subject_IDs and when_func(s)]

        return valid_sessions

    def plot(self, subject_IDs='all', when='all', fig_no=1):
        pl.experiment_plot(self, subject_IDs=subject_IDs, when=when, fig_no=fig_no)    


def _toDate(d): # Convert input to datetime.date object.
    if type(d) is str:
        try:
            return datetime.strptime(d, '%Y-%m-%d').date()
        except ValueError:
            print('Unable to convert string to date, format must be YYYY-MM-DD.')
            raise ValueError
    elif type(d) is datetime:
        return d.date()
    elif type(d) is date:
        return d
    else:
        raise ValueError
        
