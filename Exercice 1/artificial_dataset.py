import sys
import numpy as np
import pandas as pd

def main():
    data = {
      'int': np.random.randint(0, 9999, 300),
      'float': np.random.normal(7.8, 2, 300),
      'mean': np.random.normal(2.5, 1, 300),
      'zero_correlation': np.random.normal(50, 9, 300),
    }

    data['positive_correlation'] = data['int'] + data['float']
    data['negative_correlation'] = data['int'] - data['float']
    df = pd.DataFrame(data=data)
   
    df.to_csv('artificial_dataset.csv')

if __name__ == "__main__":
    sys.exit(main())
