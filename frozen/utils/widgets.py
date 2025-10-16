import os
import pickle
import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as parquet
from functools import lru_cache


class FileReader:
    """
    A utility class for reading DataFrame data based on file extensions
    Supported formats: .csv, .pkl, .pickle, .feather, .parquet, .h5, .hdf5
    """
    
    # Class-level format mapping
    _FORMAT_MAPPING = {
        '.csv': '_read_csv',
        '.pkl': '_read_pickle',
        '.pickle': '_read_pickle',
        '.feather': '_read_feather',
        '.parquet': '_read_parquet',
        '.h5': '_read_hdf',
        '.hdf5': '_read_hdf'
    }
    
    def __init__(self):
        # Instance-level cache
        self._cache = {}
    
    def __call__(self, file_path, **kwargs):
        """
        Make the instance callable to directly read files
        
        Parameters:
        ----------
        file_path: file path
        **kwargs: additional parameters passed to the specific reading method
        
        Returns:
        -------
        DataFrame object
        """
        return self.read_file(file_path, **kwargs)
    
    @classmethod
    def read(cls, file_path, **kwargs):
        """
        Class method to read files without instantiation
        
        Parameters:
        ----------
        file_path: file path
        **kwargs: additional parameters passed to the specific reading method
        
        Returns:
        -------
        DataFrame object
        """
        return cls().read_file(file_path, **kwargs)
    
    @classmethod
    def get_supported_formats(cls):
        """Get the list of supported file formats"""
        return list(cls._FORMAT_MAPPING.keys())
    
    @classmethod
    def is_format_supported(cls, file_path):
        """Check if the file format is supported"""
        _, ext = os.path.splitext(file_path)
        return ext.lower() in cls._FORMAT_MAPPING
    
    def read_file(self, file_path, use_cache=False, **kwargs):
        """
        Read DataFrame based on file extension
        
        Parameters:
        ----------
        file_path: file path
        use_cache: whether to use caching
        **kwargs: additional parameters passed to the specific reading method
        
        Returns:
        -------
        DataFrame object
        """
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        # Check cache
        if use_cache and file_path in self._cache:
            return self._cache[file_path].copy()
        
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Check if the format is supported
        if ext not in self._FORMAT_MAPPING:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: {list(self._FORMAT_MAPPING.keys())}")
        
        # Get the reading method name
        method_name = self._FORMAT_MAPPING[ext]
        
        # Call the corresponding reading method
        try:
            # Use getattr to dynamically get the method
            method = getattr(self, method_name)
            result = method(file_path, **kwargs)
            
            # Cache the result
            if use_cache:
                self._cache[file_path] = result.copy()
                
            return result
        except Exception as e:
            raise Exception(f"Failed to read file: {file_path}, error: {str(e)}")
    
    def _read_csv(self, file_path, **kwargs):
        """Read CSV file"""
        # Set default parameters
        default_kwargs = {'encoding': 'utf-8'}
        default_kwargs.update(kwargs)
        return pd.read_csv(file_path, **default_kwargs)
    
    def _read_pickle(self, file_path, **kwargs):
        """Read Pickle file"""
        return pd.read_pickle(file_path, **kwargs)
    
    def _read_feather(self, file_path, **kwargs):
        """Read Feather file"""
        return feather.read_feather(file_path, **kwargs)
    
    def _read_parquet(self, file_path, **kwargs):
        """Read Parquet file"""
        return parquet.read_table(file_path, **kwargs).to_pandas()
    
    def _read_hdf(self, file_path, **kwargs):
        """Read HDF5 file"""
        # If key is not specified, try to auto-detect
        if 'key' not in kwargs:
            with pd.HDFStore(file_path, 'r') as store:
                keys = store.keys()
                if len(keys) == 0:
                    raise ValueError("No datasets found in HDF5 file")
                # Use the first key
                kwargs['key'] = keys[0]
        
        return pd.read_hdf(file_path, **kwargs)
    
    def clear_cache(self):
        """Clear cache"""
        self._cache.clear()


# import datetime
# import pandas as pd

# class DividendAdjustment:
#     '''
#     The customized diviend adjustment module that modifies the 
#     original instrument price in response to dividend events.

#     This module includes three corresponding methods:
    
#     - `pre_div_adjustment`: Anchor to the end date.
    
#     - `post_div_adjustment`: Anchor to the start date.
    
#     - `div_adjustment`: Combine the two methods above.

#     Mutual Parameters:
#     ------------------
#     price: pd.DataFrame
#         The original instrument price data.

#     dividend: pd.DataFrame
#         The dividend event data.
#     '''

#     def pre_div_adjustment(self, price, dividend):

#         data = pd.merge(price, dividend, how='outer', left_on='trade_date', right_on='ex_date')
#         data.index = pd.to_datetime(data['trade_date'])
#         data.drop(['trade_date'], axis=1, inplace=True)

#         info = data[data['ex_date'].notna()]
#         data['pre_adj_factor'] = 1
#         for ex_date in info.index:  # dividend.ex_date
#             pre_close = (data.loc[ex_date, 'close'] - data.loc[ex_date, 'cash_div']) / (1 + data.loc[ex_date, 'stk_div'])
#             adj_factor = pre_close / data.loc[ex_date, 'close']
#             data.loc[:ex_date - datetime.timedelta(days=1), 'pre_adj_factor'] = data.loc[:ex_date - datetime.timedelta(days=1), 'pre_adj_factor'] * adj_factor
#         data['pre_adj_close'] = data['close'] * data['pre_adj_factor']

#         # data['qfq'] = data['close']
#         # for ex_date in info.index:
#         #     data.loc[:ex_date - datetime.timedelta(days=1), 'qfq'] = (data.loc[:ex_date - datetime.timedelta(days=1), 'qfq'] - data.loc[ex_date, 'cash_div']) / (1 + data.loc[ex_date, 'stk_div'])

#         return data


#     def post_div_adjustment(self, price, dividend):
#         '''The same as `pre_div_adjustment` method.'''

#         data = pd.merge(price, dividend, how='outer', left_on='trade_date', right_on='ex_date')
#         data.index = pd.to_datetime(data['trade_date'])
#         data.drop(['trade_date'], axis=1, inplace=True)

#         info = data[data['ex_date'].notna()]
#         data['post_adj_factor'] = 1
#         for ex_date in info.index:
#             pre_close = (data.loc[ex_date, 'close'] - data.loc[ex_date, 'cash_div']) / (1 + data.loc[ex_date, 'stk_div'])
#             adj_factor = data.loc[ex_date, 'close'] / pre_close
#             # pre_close = data.loc[ex_date, 'close'] * (1 + data.loc[ex_date, 'stk_div']) + data.loc[ex_date, 'cash_div']
#             # adj_factor = pre_close / data.loc[ex_date, 'close']
#             data.loc[ex_date + datetime.timedelta(days=1):, 'post_adj_factor'] = data.loc[ex_date + datetime.timedelta(days=1):, 'post_adj_factor'] * adj_factor
#         data['post_adj_close'] = data['close'] * data['post_adj_factor']

#         return data


#     def div_adjustment(self, price, dividend):

#         qfq = dvd_adj.pre_div_adjustment(price, dividend)
#         hfq = dvd_adj.post_div_adjustment(price, dividend)
#         combine = pd.merge(hfq[['close', 'post_adj_close']], qfq['pre_adj_close'], how='outer', left_on='trade_date', right_on = 'trade_date')

#         return combine

# dvd_adj = DividendAdjustment()
