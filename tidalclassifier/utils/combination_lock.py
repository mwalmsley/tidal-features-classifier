import numpy as np

class CombinationLock:
    def __init__(self, input_data):

        self.input_data = input_data # expects data to be dict of {column name: vals of that column}


        self.state = [] # list of vals of current val of that column}
        for k in self.input_data:
            self.state.append(self.input_data[k][0]) # initialise with first value

        self.indices = [] # list of vals where val is index of current state
        for k in self.input_data:
            self.indices.append(0)

        self.max_index = [] # list of vals where val is max index (len-1) of that column
        for k in self.input_data:
            self.max_index.append(len(self.input_data[k])-1)

        self.keys = [] # list of keys
        for k in self.input_data:
            self.keys.append(k)


        self.key_n = len(self.keys)

        # print(len(self.max_index))
        # print(len(self.keys))
        # print('keys', self.key_n)

    def permuteCombination(self):

        # update indices once

        # if active index value is less than maximum, increment
        if self.indices[self.active_index] < self.max_index[self.active_index]:
            self.indices[self.active_index] += 1

        # if it is the maximum, increment next column and set all others to 0
        elif self.indices[self.active_index] == self.max_index[self.active_index]:
            self.indices[self.active_index] = 0 # reset active column
            next_col = self.active_index+1
            while next_col < len(self.indices):
                if self.indices[next_col] < self.max_index[next_col]:
                    self.indices[next_col] += 1 # increment next column up by one
                    for lower_index in range(next_col):  # reset all lower columns
                        self.indices[lower_index] = 0
                    break
                else: next_col += 1


            self.active_index = 0 # begin incrementing first column again


        # update state from indices
        for key_index in range(len(self.state)):
            key = self.keys[key_index]
            val_index = self.indices[key_index]
            self.state[key_index] = self.input_data[key][val_index]

        print(self.indices)

    def permuteAllCombinations(self):
        self.active_index = 0
        allCombinations = [self.readCombination()] # read initial state before permuting
        maxed_check = [self.indices[key] >= self.max_index[key] for key in range(self.key_n)]
        while np.sum(maxed_check) < self.key_n: # until all columns hit max values, except final
            self.permuteCombination()
            combination = self.readCombination()
            allCombinations.append(combination)
            maxed_check = [self.indices[key] >= self.max_index[key] for key in range(self.key_n)]
        return allCombinations

    def readCombination(self):
        # print('read')
        return_state = {} # instruct-like dict of parameter and current value
        for index in range(self.key_n):
            k = self.keys[index]
            v = self.state[index]
            # k, v = dic.items()

            return_state[k] = v
        return return_state
