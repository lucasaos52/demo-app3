import pandas as pd

dz = pd.DataFrame({"a": [1, 2, 34]})

dz.index = ['aa','bb','cc']

print(dz.index)

print("convertendo pra json o teste")