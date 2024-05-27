import math

num = int(input())

for i in range(0, num, 1):

    x1, y1, r1, x2, y2, r2 = map(int, input().split())
    distance = math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))

    if (x1 == x2) and (y1 == y2) and (r1 == r2):
        result = -1
    
    elif ((abs(r1 - r2)) < distance) and (r1 + r2 > distance):
        result = 2
    
    elif (r1 + r2 == distance):
        result = 1
    
    elif (abs(r1 - r2) == distance):
        result = 1

    elif (r1 + r2 < distance):
        result = 0
        
    elif (abs(r1 - r2) > distance):
        result = 0

    print(result)