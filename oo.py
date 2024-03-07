import psutil

# get process id
p = psutil.Process().pid
print("Hello World", p)
while(1):
    pass