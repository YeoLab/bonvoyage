# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.decomposition import NMF

class VoyageSpace(object):

    n_components = 2
    _binsize = 0.1

    included_label = '~1'
    excluded_label = '~0'

    def binify(self, data):
        return super(SplicingData, self).binify(data, self.bins)

    def nmf(self, data, n_components=2, **kwargs):
        reducer = NMF(n_components=n_components, **kwargs)
        reduced = pd.DataFrame(reducer.fit_transform(self.binify(data).T),
                               index=data.columns)
        return reduced

    def binned_nmf_reduced(self, sample_ids=None, feature_ids=None,
                           data=None):
        if data is None:
            data = self._subset(self.data, sample_ids, feature_ids,
                                require_min_samples=False)
        binned = self.binify(data)
        reduced = self.nmf.transform(binned.T)
        return reduced

    def _is_nmf_space_x_axis_excluded(self, groupby):
        nmf_space_positions = self.nmf_space_positions(groupby)
    
        # Get the correct included/excluded labeling for the x and y axes
        event, phenotype = nmf_space_positions.pc_1.argmax()
        top_pc1_samples = self.data.groupby(groupby).groups[
            phenotype]
    
        data = self._subset(self.data, sample_ids=top_pc1_samples)
        binned = self.binify(data)
        return bool(binned[event][0])
    
    
    def _nmf_space_xlabel(self, groupby):
        if self._is_nmf_space_x_axis_excluded(groupby):
            return self.excluded_label
        else:
            return self.included_label
    
    
    def _nmf_space_ylabel(self, groupby):
        if self._is_nmf_space_x_axis_excluded(groupby):
            return self.included_label
        else:
            return self.excluded_label