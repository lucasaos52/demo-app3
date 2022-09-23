import pandas as pd
import json

df = pd.DataFrame([{"invoiceDate":"18/08/2019","amount":1140.87}])
print (df)

import json
with open('data.json', 'w') as f:
    json.dump(df.to_dict(orient='records'), f)






dz = pd.DataFrame({"a": [1, 2, 34]})

dz.index = ['aa','bb','cc']

print(dz)

print("os idnices")

print(dz.index)

print("as colunasss")
print(dz.columns)

print("os valores da coluna a")
print(dz["a"].tolist())

print("convertendo pra json o teste")

print(dz.to_json())


print("dicionarizado")

print(dz.to_dict())

print("o hso")
