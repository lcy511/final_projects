from collections import Counter, defaultdict
from random import choice, random, sample
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

#retailer1: no price discrination
#retailer2: price discrination based on accumulative consumption amount
#retailer3: random price discrimation

def purchase_will(original_price, discount_price:np.array, sensitivity:int) -> np.array:
    """

    :param original_price:
    :param discount_price:
    :param sensitivity:
    :return:

    >>> p1=Products(50,20,0.5,0.2,10000)
    >>> price = p1.price
    >>> a = purchase_will(price, 0.8*price,2)
    >>> b = purchase_will(price, 0.8*price,5)
    >>> c = purchase_will(price, 0.7*price,2)
    >>> np.all(a<b)
    True
    >>> np.all(a<c)
    True
    """
    discount = (original_price-discount_price)/original_price
    #probability = math.exp(sensitivity/100)*(1+discount)*0.3
    probability = np.exp(1+sensitivity/10 *discount) / 2 * 0.3
    return probability

class Customers:
    total_amount = 0
    count = 0
    all_customer = []
    def __init__(self, amount, sensitivity, preference, name=None, record=None):
        Customers.total_amount += amount
        Customers.count += 1
        Customers.all_customer.append(self)
        if name is None:
            self.name = 'Customer group{:}'.format(Customers.count)
        else:
            self.name = name
        self.amount = amount
        self.sensitivity = int(sensitivity)
        self.original_preference = preference
        self.get_preference(preference)
        self.get_record(record)
        #self.record = defaultdict(float) # record purchase amount

    def reset(self):
        self.get_preference(self.original_preference)


    def get_preference(self,preference:tuple): #be same in a span
        """
        Based on the probability of customers' preference,redord the perference of each single customer,
        including if one prefers a familiar retailer and if one always compares the price of the product that he decide
        to buy with the price of same product in other retailers
        :param preference: the first element represents the probability of always visiting the familiar retailer at first,
                           the secend element represents the probability of making an comparison among different retailers
        :return:
        """

        self.eachcustomer = defaultdict(dict)
        if preference[0]<0 or preference[0]>1 or preference[1]<0 or preference[1]>1:
            raise ValueError("The probability must be set of number between 0 and 1.")

        for i in range(self.amount):
            random_1 = random()
            random_2 = random()
            ID = '{:}-{:}'.format(self.name, i)
            self.eachcustomer[ID]['purchase_amount']=defaultdict(float)
            if random_1 <= preference[0]:
                self.eachcustomer[ID]['record_first'] = 1
            else:
                self.eachcustomer[ID]['record_first'] = 0
            if random_2 <= preference[1]:
                self.eachcustomer[ID]['comparison'] = 1
            else:
                self.eachcustomer[ID]['comparison'] = 0

    def get_record(self,d:dict):
        if d is not None:
            for key in d.keys():
                for cus in self.eachcustomer.keys():
                    self.eachcustomer[cus]['purchase_amount'][key]=d[key]

    def customer_shop(self) -> list: #change every day
        visits = int(choice(np.random.triangular(0.05 * self.amount, 0.2 * self.amount, 0.5 * self.amount, 3650)))
        visitors = sample(self.eachcustomer.keys(), visits)
        return visitors

    def store_to_shop(self, customer:str, retailer:list):
        if self.eachcustomer[customer]['record_first'] == 1 and len(self.eachcustomer[customer]['purchase_amount']) > 0:
            most = max(self.eachcustomer[customer]['purchase_amount'],
                        key=self.eachcustomer[customer]['purchase_amount'].get)
            i = 0
            while i<len(retailer):
                if retailer[i].name == most:
                    store = retailer[i]
                    break
                i += 1
                store = choice(retailer)
        else:
            store = choice(retailer)
        return store #Retailers

    def product_visited(self, product:np.array, popularity:np.array) ->np.array:
        amount = len(product)
        n = math.ceil(choice(np.random.triangular(0.0001 * amount, 0.001 * amount,
                                                   0.01 * amount, 3650)))
        p_visited = np.random.choice(product, n, replace=False, p=popularity)  #array
        return p_visited
#####
    def real_price(self, customer:str, price:np.array, store) ->np.array:
        a, p, d, r = store.strategy
        dis_price = price
        if self.eachcustomer[customer]['purchase_amount'][store.name] >= a:
            if random() <= p:
                if r == 0:
                    dis_price = price * (1 - d)
                else:
                    dis_price = price * (1 - np.random.uniform(0, d, len(price)))
        return dis_price

    def compare_retailer(self, customer:str, product:np.array, retailer:list)->np.array:
        best_price = product
        best_store = np.array([choice(retailer)]*len(product))
        for store in retailer:
            current_price = self.real_price(customer,product,store)
            current_store = np.array([store.name]*len(product))
            best_price,best_store = np.where(best_price>current_price, current_price, best_price),\
                                    np.where(best_price>current_price, current_store, best_store)
        return best_price, best_store




    """
    def purchase_decision(self):
        profit = defaultdict()
        visitor = self.customer_shop()
        for cus in visitor:
            store = self.store_to_shop(cus,Retailers.all_store)
            price = self.product_visited()
            store = np.array([store]*len(price))
            dis_price = self.real_price(cus,price,store[0])
            if self.each_customer[cus]['comparison'] == 1:
                store, dis_price = self.compare_retailer(cus,price,Retailers.all_store)
            
            #A, P, D, R = store.strategy          
            #dis_product = product
            #if self.each_customer[cus][store.name] >= A:
                #if random() <= P:
                    #if R == 0:
                        #dis_product = product * (1-D)
                    #else:
                        #dis_product = product * (1-random.uniform(0,D))
             
            probability = purchase_will(price,dis_price,self.sensitivity) #modify later
            random_array = np.random.random_sample(len(price))
            purchase = np.where(random_array<probability,1,0)
            for s in Retailers.all_store:
                self.each_customer[cus][s.name] += sum(dis_price[(purchase==1) * (store==s.name)==True])
                self.record[cus][s.name] += sum(purchase[store==s.name])
                profit[s.name] = sum(dis_price[(purchase==1) * (store==s.name)==True]-
                                     price[(purchase==1) * (store==s.name)==True]*(1-s.profit_rate))
            #self.each_customer[cus][store.name] += sum(dis_product[purchase==1])
            #self.record[cus][store.name] += sum(purchase)
            #profit = sum(dis_product[purchase==1]-product[purchase==1]*(1-store.profit_rate))
            return profit
"""

class Products:
    all_price = np.array([])
    all_popularity = np.array([])
    all_amount = 0
    def __init__(self, price_mean, price_sd, popu_mean, popu_sd, amount):
        self.mean = price_mean
        self.sd = price_sd
        self.amount = amount
        self.get_price()
        self.get_popularity(popu_mean, popu_sd)
        Products.all_price = np.append(Products.all_price,self.price)
        Products.all_popularity = np.append(Products.all_popularity, self.popularity)
        Products.all_popularity = Products.all_popularity/sum(Products.all_popularity)
        Products.all_amount += self.amount


    def get_price(self):
        product = np.zeros(1)
        n = self.amount
        while len(product) < self.amount:
            product = np.random.normal(self.mean, self.sd, n)
            product = product[product>0]
            n += 100
        self.price = np.random.choice(product, self.amount, replace=False)

    def get_popularity(self, mean, sd):
        popularity = np.zeros(1)
        n = self.amount
        while len(popularity) < self.amount:
            popularity = np.random.normal(mean, sd, n)
            popularity = popularity[0 < popularity]
            n += 100
        self.popularity = np.random.choice(popularity, self.amount, replace=False)
        self.popularity = self.popularity/sum(self.popularity)


class Retailers:
    count = 0
    all_store = []
    def __init__(self, name=None, profit_rate=0.4, strategy=None):
        Retailers.count += 1
        Retailers.all_store.append(self)
        # retailer's name
        if name is None:
            self.name = 'Retailer{:}'.format(Retailers.count)
        else:
            self.name = name
        self.get_strategy(strategy)
         # retailer's price discrimination type
        if self.strategy[2] == 0: #no discount
            self.type = 'No price discrimination'
        elif self.strategy[2] > 0 and self.strategy[3]==0:
            self.type = 'Price discrimination based on consumption amount'
        elif self.strategy[2] > 0 and self.strategy[3]==1:
            self.type = 'Random Price discrimination'
        else:
            raise ValueError('Please input a valid strategy')

        self.profit_rate = profit_rate
        self.profit = 0 # reset later
        self.rank=defaultdict(Counter)

    def get_strategy(self, t:tuple):
        if t is None:
            self.strategy = (0,1,0,0)   # amount, population, discount, randomness
        else:
            A,P,D,R = t
            if A<0 or (P<0 or P>1) or (D<0 or D>1) or (R!=0 and R!=1):
                raise ValueError('Please input a valid strategy')
            self.strategy = t

    def reset(self):
        self.profit = 0
        self.rank = defaultdict(Counter)

def purchase_decision(customer:list, retailer:list, product:Products, popularity:np.array):
    """
    Simulate purchase process in one day.Given the parameters, some customers will be selected randomly to shop in the
    randomly selected stores, and they will also view some products that are also selected randomly based on their popularity.
    During the process, whether they would buy the product are dependent on the price and their price sensitivity.
    :param customer:
    :param retailer:
    :param product:
    :param popularity:
    :return:

    >>> c1 = Customers(1000,1,(0.5,0.5))
    >>> c2 = Customers(1000,5,(0.7,0.2))
    >>> a = sum(c1.eachcustomer[cus]['record_first'] for cus in c1.eachcustomer.keys())
    >>> 400 < a < 600
    True
    >>> b = sum(c2.eachcustomer[cus]['comparison'] for cus in c2.eachcustomer.keys())
    >>> 100 < b < 300
    True
    >>> c1.eachcustomer['Customer group1-16']['purchase_amount']['r']
    0.0
    >>> 50<len(c2.customer_shop())<500
    True
    >>> r1=Retailers()
    >>> r2=Retailers(strategy=(0,1,0.2,0))
    >>> r3=Retailers(strategy=(0,0.5,0.25,1))
    >>> p1=Products(50,20,0.5,0.2,10000)
    >>> p2=Products(200,60,0.3,0.1,10000)
    >>> len(Products.all_price) == 20000
    True
    >>> c1.store_to_shop('Customer group1-16',Retailers.all_store).name in ['Retailer1', 'Retailer2', 'Retailer3']
    True
    >>> c = c2.product_visited(Products.all_price, Products.all_popularity)
    >>> 2 < len(c) < 200
    True
    >>> 0 < sum(c) < 200*260
    True
    >>> price1 = c1.real_price('Customer group1-16',p1.price, r1)
    >>> price2 = c1.real_price('Customer group1-16',p1.price, r2)
    >>> price3 = c1.real_price('Customer group1-16',p1.price, r3)
    >>> np.all(price1>price2)
    True
    >>> np.all(price1>=price3)
    True
    >>> best_price, best_store = c1.compare_retailer('Customer group1-16',p1.price, Retailers.all_store)
    >>> np.all(best_store != r1.name)
    True
    >>> outcome = purchase_decision(Customers.all_customer, Retailers.all_store, Products.all_price, Products.all_popularity)
    >>> len(outcome)
    3
    >>> #c2.eachcustomer['Customer group2-16']['purchase_amount']
    >>> #outcome
    >>> #c2.eachcustomer
    """
    # profit = defaultdict(float)
    for group in customer:
        visitor = group.customer_shop()
        for cus in visitor:
            store = group.store_to_shop(cus,retailer)  # class
            price = group.product_visited(product, popularity)
            store_array = np.array([store.name]*len(price))
            dis_price = group.real_price(cus,price,store)  # class
            if group.eachcustomer[cus]['comparison'] == 1:
                dis_price, store_array = group.compare_retailer(cus,price,retailer)
            probability = purchase_will(price,dis_price,group.sensitivity)  # may modify later
            random_array = np.random.random_sample(len(price))
            purchase = np.where(random_array<probability,1,0)
            for s in retailer:
                group.eachcustomer[cus]['purchase_amount'][s.name] += sum(dis_price[(purchase==1) * (store_array==s.name)==True])
                #customer.record[cus][s.name] += sum(purchase[store==s.name])
                s.profit += sum(dis_price[(purchase==1) * (store_array==s.name)==True]-
                                    price[(purchase==1) * (store_array==s.name)==True]*(1-s.profit_rate))
                #profit[s.name] += sum(dis_price[(purchase==1) * (store_array==s.name)==True]-
                                    #price[(purchase==1) * (store_array==s.name)==True]*(1-s.profit_rate))
            #self.each_customer[cus][store.name] += sum(dis_product[purchase==1])
            #self.record[cus][store.name] += sum(purchase)
            #profit = sum(dis_product[purchase==1]-product[purchase==1]*(1-store.profit_rate))
    return [s.profit for s in retailer]

def reset(data:list):
    for obj in data:
        obj.reset()

def simulation(customer:list, retailer:list, price:np.array, popularity:np.array, times:int, span=365, image=False):
    """

    :param customer:
    :param retailer:
    :param price:
    :param popularity:
    :param times:
    :param span:
    :return:
    >>> c1 = Customers(1000,1,(0.5,0.5))
    >>> c2 = Customers(1000,5,(0.7,0.2))
    >>> r1=Retailers()
    >>> r2=Retailers(strategy=(0,1,0.2,0))
    >>> r3=Retailers(strategy=(0,0.5,0.25,1))
    >>> p1=Products(50,20,0.5,0.2,100)
    >>> simulation([c1,c2], [r1,r2,r3], p1.price, p1.popularity,10,10,True) # doctest: +ELLIPSIS
    Retailer4   : No price discrimination
    Earn...
    Rank No.1 for...
    ...
    Retailer6   : Random Price discrimination
    ...
    Rank No.3 for...

    >>> #simulation([c1,c2], [r1,r2,r3], p1.price, p1.popularity,10,10,True)

    """

    record = {}
    rank = {}
    for n in range(times):
        reset(customer)
        reset(retailer)
        profit_record = {}
        for day in range(1,span+1):
            total_profit = purchase_decision(customer, retailer, price, popularity)
            profit_record[day] = total_profit
        record[n+1] = total_profit
        rank[n+1] = list(pd.Series(total_profit).rank(method='first',ascending=False))

        #if image:
            #profit_record = pd.DataFrame(profit_record, index=[s.name for s in retailer]).T
            #profit_record.plot(figsize=(9,6)).set(xlabel='Day', ylabel='Accumulated Profit')
            # reference: https://stackoverflow.com/questions/45376232/how-to-save-image-created-with-pandas-dataframe-plot/45379210
            #plt.savefig("D:/final_projects/plots/time{:}.png".format(n+1))

    record = pd.DataFrame(record,index=[s.name for s in retailer]).T
    rank = pd.DataFrame(rank, index=[s.name for s in retailer]).T

    if image:
        record.plot(figsize=(9,6)).set(xlabel='Simulation Times', ylabel='Profit')
        plt.savefig("D:/final_projects/plots/summary_{:}times.png".format(times))

    total = len(retailer)
    for i in range(total):
        print("{:<12}: {:}\nEarn {:.2f} on average in {:} days".format(retailer[i].name, retailer[i].type, record[retailer[i].name].mean(),span))
        #print(strategy and type)
        for j in range(total):
            ranking_number = Counter(rank[retailer[i].name])[j+1]
            percentage = ranking_number/times
            print("Rank No.{:} for {:^5} times, rate: {:^6.2%}".format(j+1, ranking_number,percentage))

if __name__ == '__main__':
    c1 = Customers(1000, 5, (0.5, 0.5))
    c2 = Customers(500, 3, (0.7, 0.2))
    c3 = Customers(100, 8, (0.3, 0.8))

    r1 = Retailers()
    r2 = Retailers(strategy=(0, 1, 0.2, 0))
    r3 = Retailers(strategy=(0, 0.5, 0.4, 1))
    #r2 = Retailers()
    #r3 = Retailers()
    p1 = Products(50, 20, 5, 1, 100)
    p2 = Products(200, 60, 5, 2, 200)
    simulation(Customers.all_customer, Retailers.all_store, Products.all_price, Products.all_popularity, 100, 100, True)





