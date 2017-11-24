import pandas as pd

df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})

print df, '\n'

"""print df["A"][2], '\n'

#this is the only way to set value without getting warning 
df.iat[2, 0] = "b"

print df["A"][2]
print df.iat[2, 0], '\n'

print df, '\n'

# print whole row
print df.loc [df.A == "b"], '\n'
#or
print df.iloc[[1]]

# print whole column
print df.A

#print selected column value only
print df.A [df.A == "b"], '\n'

#print selected value only
print df.A[0]"""
print type(df.values)
print df.values

arr = df.values

df2=pd.DataFrame(arr) 
print type (df2)
print df2

