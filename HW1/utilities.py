def sum_nested_list(mylist):
    SumofMylist=0
    for i in mylist:
        for k in i:
            SumofMylist+=k 
    return(SumofMylist)
