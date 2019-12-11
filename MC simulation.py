from collections import Counter, defaultdict
from random import choice, random, sample
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

#retailer1: no price discrination
#retailer2: price discrination based on accumulative consumption amount
#retailer3: random price discrimation

def purchase_will(original_price:np.array, discount_price:np.array, sensitivity:int) -> np.array:
    """
    Calculate people's  willing to pay when they meet a discount
    :param original_price: the standard price without any knid of discount
    :param discount_price: the discount price which must match with the original price
    :param sensitivity: price sensitivity level, the larger the number is, the more sensitive to price customers are
    :return: the probability of making a purchase for each product

    >>> p1=Products(50,20,0.5,0.2,10000)
    >>> price = p1.price
    >>> a = purchase_will(price, 0.8*price,2)
    >>> b = purchase_will(price, 0.8*price,5)
    >>> c = purchase_will(price, 0.7*price,2)
    >>> np.all(a<b)
    True
    >>> np.all(a<c)
    True
    >>> purchase_will(np.array(range(3)), np.array(range(2)),1)
    Traceback (most recent call last):
    ValueError: The array of original price and the array of discount price must have the same dimension.
    """
    if original_price.shape != discount_price.shape:
        raise ValueError("The array of original price and the array of discount price must have the same dimension.")
    discount = (original_price-discount_price)/original_price
    #probability = math.exp(sensitivity/100)*(1+discount)*0.3
    probability = np.exp(1+sensitivity/10 *discount) / 2 * 0.3
    return probability

class Customers:
    """
    A group of customers
    """
    total_amount = 0
    count = 0
    all_customer = []
    def __init__(self, amount:int, sensitivity:int, preference:tuple, name=None, record=None):
        """
        :param amount: how many customers in this group
        :param sensitivity: price sensitivity level
        :param preference: a tuple with 2 numbers between 0 and 1, respectively representing the propotion of customers\
                            who prefer the store they cunsume most and the proportation of customers who prefer to
                            compare the price in all retailers
        :param name: name of the object
        :param record: customers' previous record about accumulated consumption amount in each retailer
        """
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
        self.original_record = record
        self.get_preference(preference)
        self.get_record(record)
        #self.record = defaultdict(float) # record purchase amount

    def reset(self):
        """
        clear all the record in self.eachcustomer

        """
        self.get_preference(self.original_preference)
        self.get_record(self.original_record)


    def get_preference(self,preference:tuple):
        """
        Based on the probability of customers' preference,redord the perference and cunsumption history of each customer

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
        """
        If the customers have previous purchasing history, add them into their current record
        :param d: record about the retailer and consumption amount
        """
        if d is not None:
            for key in d.keys():
                for cus in self.eachcustomer.keys():
                    self.eachcustomer[cus]['purchase_amount'][key]=d[key]

    def customer_shop(self) -> list: #change every day
        """
        random select person to shop in the store based on traingular distribution
        :return: list of customer's ID
        """
        # the size could be larger if the code efficiency is improved
        visits = int(choice(np.random.triangular(0.05 * self.amount, 0.2 * self.amount, 0.5 * self.amount, 365)))
        visitors = sample(self.eachcustomer.keys(), visits)
        return visitors

    def store_to_shop(self, customer:str, retailer:list) :
        """
        Decide which store the customer will shop in
        :param customer: customers selected to shop today
        :param retailer: all the retailers
        :return: an object from Retailers class
        """
        # whether the customer first go to the store with highest consumption record
        if self.eachcustomer[customer]['record_first'] == 1 and len(self.eachcustomer[customer]['purchase_amount']) > 0:
            most = max(self.eachcustomer[customer]['purchase_amount'],
                        key=self.eachcustomer[customer]['purchase_amount'].get)
            i = 0
            # if the selected store is a current retailer
            while i<len(retailer):
                if retailer[i].name == most:
                    store = retailer[i]
                    break
                i += 1
                store = choice(retailer)
        # select a store randomly
        else:
            store = choice(retailer)
        return store # Retailers

    def product_visited(self, product:np.array, popularity:np.array) ->np.array:
        """
        Randomly select products the customer will view based on their popularity
        """
        amount = len(product) # total amount of products
        # amount of products being viewed
        n = math.ceil(choice(np.random.triangular(0.0001 * amount, 0.001 * amount,
                                                   0.01 * amount, 365)))
        p_visited = np.random.choice(product, n, replace=False, p=popularity)  #array
        return p_visited

    def real_price(self, customer:str, price:np.array, store) ->np.array:
        """
        Get the discount price of the given products in the given retailers
        :param customer: list of IDs of customers
        :param price: price of each product
        :param store: class Retailer object
        :return:
        """
        # get price strategy
        a, p, d, r = store.strategy #threshold amount, proportation, discount, randomness
        dis_price = price
        if self.eachcustomer[customer]['purchase_amount'][store.name] >= a:
            if random() <= p:
                if r == 0: # fixed discount rate
                    dis_price = price * (1 - d)
                else: # random discount
                    dis_price = price * (1 - np.random.uniform(0, d, len(price)))
        return dis_price

    def compare_retailer(self, customer:str, product:np.array, retailer:list)->np.array:
        """
        Compare the products' price in every given retailer and choose the cheapest one respectively
        :return: the best price of each product and which retailer it belongs to
        """
        best_price = product
        best_store = np.array([choice(retailer)]*len(product))
        for store in retailer:
            current_price = self.real_price(customer,product,store)
            current_store = np.array([store.name]*len(product))
            best_price,best_store = np.where(best_price>current_price, current_price, best_price),\
                                    np.where(best_price>current_price, current_store, best_store)
        return best_price, best_store


class Products:
    """
    A group of products.
    The standard price of all products follows the normal distribution
    The popularity of all products also follows the normal distribution
    The profit rate of all products are same.
    However, to simplify the simulation, I set a profit rate for each retailer,instead of setting a profit rate for\
    each group of product. If the code efficiency could be improved a lot, I should
    set the products' profit rate and even arrange a distribution pattern for it.
    """
    all_price = np.array([])
    all_popularity = np.array([])
    all_amount = 0
    def __init__(self, price_mean:float, price_sd:float, popu_mean:float, popu_sd:float, amount:int):
        """
        :param price_mean: mean of normal distribution about product
        :param price_sd: standard deviation of normal distribution about product
        :param popu_mean: mean of normal distribution about popularity
        :param popu_sd: standard deviation of normal distribution about popularity
        :param amount: how many kinds of product in this groups
        """
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
        """
        Calculate the price for each kind of product
        """
        product = np.zeros(1)
        n = self.amount
        while len(product) < self.amount:
            product = np.random.normal(self.mean, self.sd, n)
            product = product[product>0]
            n += 100
        self.price = np.random.choice(product, self.amount, replace=False)

    def get_popularity(self, mean, sd):
        """
        Calculate the popularity for each kind of product
        """
        popularity = np.zeros(1)
        n = self.amount
        while len(popularity) < self.amount:
            popularity = np.random.normal(mean, sd, n)
            popularity = popularity[0 < popularity]
            n += 100
        self.popularity = np.random.choice(popularity, self.amount, replace=False)
        self.popularity = self.popularity/sum(self.popularity)


class Retailers:
    """
    Every retailer sell the same products with the same cost
    Every retailer must select a price strategy
    """
    count = 0
    all_store = []
    def __init__(self, name=None, profit_rate=0.4, strategy=None):
        """
        :param name: retailer's name
        :param profit_rate: a fixed profit rate for every product it sell
        :param strategy: a tuple of price discrimination strategy, including threshold amount, the proportion of customers\
                receiving discount, the discount, and the randomness of discount
        """
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
        #self.rank=defaultdict(Counter)

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
        #self.rank = defaultdict(Counter)

def purchase_decision(customer:list, retailer:list, product:np.array, popularity:np.array) -> list:
    """
    Simulate purchase process in one day.Given the parameters, some customers will be selected randomly to shop in the
    randomly selected stores, and they will also view some products that are also selected randomly based on their popularity.
    During the process, whether they would buy the product are dependent on the price and their price sensitivity.
    :param customer: a list of objects in class Customers
    :param retailer: a list of objects in class Retailers
    :param product: an array of priducts' price
    :param popularity: an array of priducts' popularity
    :return: a list containing profit of each retailer

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
    """
    carry out the reset method
    :param data: list containing objects
    """
    for obj in data:
        obj.reset()

def simulation(customer:list, retailer:list, price:np.array, popularity:np.array, times:int, span=365, image=False):
    """
    During a given period, simulate the profit of each retailer, based on the given customers and products
    Repeat the simulation for many times

    :param customer: a list of objects in class Customers
    :param retailer: a list of objects in class Retailers
    :param price: an array of priducts' price
    :param popularity: an array of priducts' popularity
    :param times: the number of simulations
    :param span: how many days you want to simulate in each simulation
    :param image: if save the image about profit curve for each retailer
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

        #if image: #image of every simulation
            #profit_record = pd.DataFrame(profit_record, index=[s.name for s in retailer]).T
            #profit_record.plot(figsize=(9,6)).set(xlabel='Day', ylabel='Accumulated Profit')
            # reference: https://stackoverflow.com/questions/45376232/how-to-save-image-created-with-pandas-dataframe-plot/45379210
            #plt.savefig("D:/final_projects/plots/time{:}.png".format(n+1))

    record = pd.DataFrame(record,index=[s.name for s in retailer]).T
    rank = pd.DataFrame(rank, index=[s.name for s in retailer]).T

    if image: # image of the whole simulation
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
    simulation(Customers.all_customer, Retailers.all_store, Products.all_price, Products.all_popularity, 10, 100, True)





