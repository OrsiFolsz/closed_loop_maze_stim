# This module is used to test the OSC server and client. It is not used in the actual task.
import bonsai_osc

class m25_API():
    def __init__(self):
        # Initialize OSC server
        ip = '127.0.0.1'
        receive_port = 2323
        data_shape = (2, 1)
        self.data_block = bonsai_osc.DataBlock(*data_shape)
        # self.server = bonsai_osc.setup_osc_server(self.ip, self.receive_port, self.data_block)

        try:
            self.server = bonsai_osc.setup_osc_server(ip, receive_port, self.data_block)
            print(f"OSC server started on {ip}:{receive_port}")
        except Exception as e:
            print(f"Failed to start OSC server: {e}")

    def test(self):
        # Print last 20 x-y coordinates received by the server
        for i in range(20):
            self.server.handle_request()
            print(len(self.data_block.data))
            print(self.data_block.x,self.data_block.y)
            # print(self.server)

server = m25_API()
server.test()