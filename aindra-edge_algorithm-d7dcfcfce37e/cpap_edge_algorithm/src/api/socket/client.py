import os
import socket
import threading
import subprocess


def recv_q(address, queue):
    """Wraps unix socket and Queue to crate an queue that can receive data"""

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(address)

    sock.listen(1)
    while True:
        connection, client_address = sock.accept()
        data = connection.recv()
        queue.put(data)


def send_q(address, queue):
    """Wraps unix socket and Queue to crate an queue that can send data"""

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(address)

    sock.listen(1)
    while True:
        data = queue.get()
        sock.send(data)


def lock_wrap(lock, lock_file):
    pass


def event_wrap(event, event_file):
    pass


def queue_processor(queue_in, queue_out, process_lock, quit_event, model_path, log_file, progress=None):

    lock_file = '/tmp/cervastra_socket_api.lock'
    event_file = '/tmp/cervastra_socket_api.event'
    send_q_port = '/tmp/cervastra_socket_api.senq'
    recv_q_port = '/tmp/cervastra_socket_api.recvq'

    threads = list()
    threads.append(threading.Thread(target=send_q, args=(send_q_port, queue_in)))
    threads.append(threading.Thread(target=recv_q, args=(recv_q_port, queue_out)))

    threads.append(threading.Thread(target=lock_wrap, args=(process_lock, lock_file)))
    threads.append(threading.Thread(target=event_wrap, args=(quit_event, event_file)))

    for thread in threads:
        thread.start()

    my_path = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(my_path, "server.py")
    server_output = subprocess.check_output(['. /home/hari/Documents/cacx/venv_p3_cacx/bin/activate python3',
                                             server_path, send_q_port, recv_q_port, lock_file, event_file,
                                             model_path, log_file], shell=True)


