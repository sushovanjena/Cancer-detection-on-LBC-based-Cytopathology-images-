"""Starts a server over UNIX socket for Astra"""
import sys
import socket
import threading
import multiprocessing as mp

from edge_algorithm.src.wsi_analyzer import queue_processor


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


if __name__=="__main__":

    # server_path, send_q_port, recv_q_port, \
    # lock_file, event_file, model_path, log_file])

    input_address = sys.argv[1]
    output_address = sys.argv[2]
    lock_file = sys.argv[3]
    event_file = sys.argv[4]
    model_path = sys.argv[5]
    log_file = sys.argv[6]

    queue_in, queue_out = mp.Queue(), mp.Queue()
    proc_lock, quit_event = mp.Lock(), mp.Event()

    threads = list()
    threads.append(threading.Thread(target=send_q, args=(('localhost', input_address), queue_in)))
    threads.append(threading.Thread(target=recv_q, args=(('localhost', output_address), queue_out)))
    threads.append(threading.Thread(target=lock_wrap, args=(lock_file, proc_lock)))
    threads.append(threading.Thread(target=event_wrap, args=(event_file, quit_event)))

    queue_processor(queue_in, queue_out, lock_wrap, event_wrap, None, log_file)





