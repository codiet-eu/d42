import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 

from collections import namedtuple

from codietpgm.dag_gflownet.scores.bde_score import BDeScore

StateCounts = namedtuple('StateCounts', ['key', 'counts'])


class DBNBDeScore(BDeScore):
    """BDe score for DBN 
    """
    def __init__(self,
                 data,
                 prior, 
                 num_vars, 
                 num_time_slices, 
                 equivalent_sample_size=1.):
        
        super().__init__(data, prior, equivalent_sample_size)

        self.num_vars = num_vars
        self.num_time_slices = num_time_slices

    def state_counts(self, target, indices, indices_after=None):

        # Source: pgmpy.estimators.BaseEstimator.state_counts()

        all_indices = indices if (indices_after is None) else indices_after

        parents = [self.column_names[index] for index in all_indices]
        variable = self.column_names[target]
        
        data = self.data[self._interventions != target]

        #if variable is in prior network 
        if target < self.num_vars :  

            data = data[[variable] + parents].dropna()
            state_count_data = (data.groupby([variable] + parents, observed = False).size().unstack(parents))

        #transition network
        else: 
                
            data_orig = data[[variable] + parents].dropna()
            state_count_data_orig = (data_orig.groupby([variable]+ parents, observed = False).size().unstack(parents)) 
            state_count_data = state_count_data_orig

            #loop over all parents and vars in other slices
            for i in range(self.num_vars, self.num_variables-self.num_vars, self.num_vars):
                var_inotherslices= self.column_names[target+i]
                parents_inotherslices = [self.column_names[index+i] for index in all_indices]
                data_otherslices = data[[var_inotherslices] + parents_inotherslices].dropna()
                state_count_data_otherslices = (data_otherslices.groupby([var_inotherslices]+ parents_inotherslices, observed = False).size().unstack(parents_inotherslices)) 
                # summing counts over all time slices
                state_count_data = state_count_data+state_count_data_otherslices
                        

        if not isinstance(state_count_data.columns, pd.MultiIndex):
            state_count_data.columns = pd.MultiIndex.from_arrays(
                [state_count_data.columns]
            )

        parent_states = [self.state_names[parent] for parent in parents]
        columns_index = pd.MultiIndex.from_product(parent_states, names=parents)

        state_counts_after = StateCounts(
            key=(target, tuple(all_indices)),
            counts=(state_count_data
                .reindex(index=self.state_names[variable], columns=columns_index)
                .fillna(0))
        )

        if indices_after is not None:
            subset_parents = [self.column_names[index] for index in indices]
            if subset_parents:
                data = (state_counts_after.counts
                    .groupby(axis=1, level=subset_parents)
                    .sum())
            else:
                data = state_counts_after.counts.sum(axis=1).to_frame()

            state_counts_before = StateCounts(
                key=(target, tuple(indices)),
                counts=data
            )
        else:
            state_counts_before = None

        return (state_counts_before, state_counts_after)

