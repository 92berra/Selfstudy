
def fibonacci(n):
    global count_a
    global count_b

    if (n==0):
        #print("0")
        count_a += 1
        return 0
    elif (n==1):
        #print("1")
        count_b += 1
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
         
t = int(input())

for i in range(0, t, 1):
    count_a = 0
    count_b = 0
    N = int(input())
    fibonacci(N)
    print(count_a, count_b)