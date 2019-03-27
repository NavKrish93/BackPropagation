
# coding: utf-8

# In[1]:


import sys
from sympy import symbols, diff,lambdify
import matplotlib.pyplot as plt


# In[2]:


def forward(val,expr,sub):
    ex = expr
    for count,s in enumerate(sub):
        func = lambdify(s,ex)
        ex = func (val[count])
    #print(ex)
    return ex

def loss(Pred,GrdTruth):
    loss = Pred-GrdTruth
    MeanSqError = (loss*loss)/2
    return MeanSqError
  


# In[3]:


def Delta(pred, actual):
    a,x =symbols('a x', real=True)
    y=((x-a)**2)/2
    z= diff(y,x)
    #print(z)
    return z.subs([(x,pred),(a,actual)]) 

def WeightUpdate(val,Output,GrdTruth,LRN,expr,sub):
    delta = Delta(Output,GrdTruth)
    ex = diff(expr,sub[1])
    func = lambdify(sub[0],ex)
    Input = func(val[0])
    #print(Input,delta)
    NW1 = val[1] -(LRN*(delta*Input)) 
    return NW1


# In[4]:


def graphplot(iteration, Error):
    # plotting the points  
    plt.plot(iteration, Error, color='green')
    #plt.plot(iteration, Error, color='green', linestyle='dashed', linewidth = 3,marker='o', markerfacecolor='blue', markersize=12) 

    # setting x and y axis range 
    #plt.ylim(1,8) 
    #plt.xlim(1,8) 

    # naming the x axis 
    plt.xlabel('Iteration - Epoc') 
    # naming the y axis 
    plt.ylabel('Loss - Error') 

    # giving a title to my graph 
    plt.title('Loss Plotting') 

    # function to show the plot 
    plt.show()
    return 0


# In[5]:


def BackPropagation(val,GrdTruth,LRN,expr,sub):
    Output = forward(val,expr,sub)
    Error = loss(Output,GrdTruth)
    Error1 = 0
    ErrorPlot =[]
    i =1
    iteration =[]
    while(Error != Error1):
        val[1] = WeightUpdate(val,Output,GrdTruth,LRN,expr,sub)
        #print(val[1])
        Output = forward(val,expr,sub)
        Error1 = Error
        Error = loss(Output,GrdTruth)
        ErrorPlot.append(Error)
        iteration.append(i)
        i = i+1        
        #print("Output",Output )
        #print("Error",Error )
    graphplot(iteration, ErrorPlot)
    
    #gradiant descent 


# In[6]:


#input
inputdata = 3
w1 = 0.5
grdtruth = 6
lrn =0.1

#                                   input equation 
#
#      # # # # # #                 # # # # # #                    # # # # # #                 
#      #         #                  #         #                   #         #
#      #  Input  #  =========>       # Weight  #        ==        # Output  #
#      #   Ip1   #                    #   W1    #                 #   O1    #
#      # # # # # #                     # # # # # #                # # # # # #
#                             
#                                     
In,W1 =symbols('In W1', real=True) 
expr =  In * W1
sub = (In,W1)
val = [3,0.5]
BeforeTraining = forward(val,expr,sub)
BackPropagation(val,grdtruth,lrn,expr,sub)


# In[7]:


AfterTraining = forward(val,expr,sub)


# In[8]:


print("Expected Result",grdtruth)
print("BeforeTraining",BeforeTraining)
print("AfterTraining",AfterTraining)

