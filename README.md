# IS590PR Final Project
# Topic

Monte Carlo simulation on profit prediction based on price discrimination

Price discrimination is a selling strategy that charges customers different prices 
for the same products.  While adopting price discrimination can boost sales, 
the decline of price will also reduce the profit per product. What I want to 
explore is whether the revenue created by the discrimination could cancel out 
the loss brought by the decline of profit per product.

# Hypothesis

A well-designed price discrimination strategy could increase a retailer's profit.

# Assumptions
1.Assumptions about Retailers

1.1 Every retailer sell the same products with the same cost

1.2 Every retailer must select a price strategy from the following 3 types: 

No price discrimination: sell everything with original price

Price Discrimination based on the accumulated purchase amount: set a threshold 
amount of receiving discount

Random Price Discrimination: set a upper limit for the discount and proportion
of customers, then randomly select customers to offer random discount 
no more than the limit

2.Assumptions about Products

2.1 The standard price of all products follows the normal distribution

2.2 The probability of one product being viewed is dependent on its popularity

2.3 The popularity of all products also follows the normal distribution

2.4 The profit rate of all products are same

3.Assumptions about Customers

3.1 The daily number of customers visiting any store follows the
   triangular distribution

3.2 The number of products every customer view follows the
   triangular distribution

3.3 Customers' preference is evaluated from 2 aspects: whether they prefer first
go to the store where they have most consumption amount and whether they prefer
compare product's price in all stores

3.4 Customers' willingness to pay will increase as the price declines. Price sensitivity
varies among customers and the relationship among discount, purchase willingness
and sensitivity follow the formula: P=0.15*exp(1+discount*sensitivity/10)

![image](https://raw.githubusercontent.com/lcy511/final_projects/master/images/price_sensitivity.png)

# Simulation
During a given period, simulate the profit of each retailer,based on the
given customers and products, and repeat the simulation for many times.

I simulated 3 scenarios.

1. Scenario 1

All retailers cuoduct no price discrimination strategy

![image](https://raw.githubusercontent.com/lcy511/final_projects/master/images/Scenario1/describe.png)

Result 1

![image](https://raw.githubusercontent.com/lcy511/final_projects/master/images/Scenario1/summary_100times.png)

![image](https://raw.githubusercontent.com/lcy511/final_projects/master/images/Scenario1/result.png)

2. Scenario 2

Retailer 2 and retailer 3 conduct price discrimination strategy

![image](https://raw.githubusercontent.com/lcy511/final_projects/master/images/Scenario2/describe.png)

Result 2

![image](https://raw.githubusercontent.com/lcy511/final_projects/master/images/Scenario2/summary_100times.png)

![image](https://raw.githubusercontent.com/lcy511/final_projects/master/images/Scenario2/result.png)

3. Scenario 3

Retailer 2 pulls up the threshold amount of conducting price discrimination strategy

![image](https://raw.githubusercontent.com/lcy511/final_projects/master/images/Scenario3/describe.png)

Result 3

![image](https://raw.githubusercontent.com/lcy511/final_projects/master/images/Scenario3/summary_1000times.png)

![image](https://raw.githubusercontent.com/lcy511/final_projects/master/images/Scenario3/result.png)

# Conclusion
Price discrimination could increase a retailer's profit, but whether it works depends on some factors and opponent's strategy is important one of them.











 





 


     
   






