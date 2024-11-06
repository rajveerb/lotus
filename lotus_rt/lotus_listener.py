import socket

# create the socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
port = 12345
s.bind(('127.0.0.1', port))
s.listen(5)

print(" --- Log is listening on localhost --- ")

while True:
    # establish a connection with the client
    c, addr = s.accept()
    print("LOG: Got connection from", addr)

    while True:
        text = c.recv(1024).decode()
        if text: print("LOG:", text)
