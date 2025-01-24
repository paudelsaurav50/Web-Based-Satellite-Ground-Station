import sanosatdigipacketparser as ssdp
import socket

HOST = "127.0.0.1"  #Localhost
PORT = 1600  # Port to listen


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (HOST, PORT)
sock.connect(server_address)

while 1:
    message=input("Please Enter the Digipeater Message for SanoSat-1\n") #get input
    data=ssdp.parse_sanosat1_digi_packets(message)
    print(message)
    print('Parsed Bytes:',data)
    sock.sendall(data)
