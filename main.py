import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import os

# load csv
df = pd.read_csv('homeprices.csv')

# create linear regression obj
reg = linear_model.LinearRegression()
# fit linear regression
reg.fit(df[['area']], df.price)
# coefficient
print(reg.coef_)
# intercept
print(reg.intercept_)
# predict price for house of 3300 ft
print(reg.predict([[3300]]))
# check result is equal to: y = coeff * x + inter
print(reg.coef_*3300+reg.intercept_)

# read new dataset with areas
d = pd.read_csv('areas.csv')
# predict prices
p = reg.predict(d)
# add col prices to d
d['prices'] = p
# write to csv without index
d.to_csv('prediction.csv', index=False)

# plot lin reg
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')
plt.show()

# DF2. per capita income

# load csv
df = pd.read_csv('income.csv')
# change column name
df = df.rename(columns={"per capita income (US$)": "PCI"})

# create linear regression obj
reg = linear_model.LinearRegression()
# fit linear regression
reg.fit(df[['year']], df.PCI)
# predict PCI in 2020
reg.predict([[2020]])

# plot distribution
plt.scatter(df.year, df.PCI, color='blue', marker='o')
plt.plot(df.year, reg.predict(df[['year']]), color='red', ls='dashed')
plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.show()
