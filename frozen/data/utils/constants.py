import pandas as pd
from datetime import datetime, timedelta

ONE_DAY = timedelta(days=1)
DEFAULT_START = "20050101"
DEFAULT_END = "20231231"
DEFAULT_START_TIME = "20230101 09:30:00"
DEFAULT_END_TIME = "20231231 15:00:00"
TODAY = datetime.today().strftime("%Y%m%d")
MAX_TIMESTAMP = pd.Timestamp.max
MIN_TIMESTAMP = pd.Timestamp.min
