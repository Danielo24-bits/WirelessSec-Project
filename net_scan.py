from pyflowmeter.sniffer import create_sniffer

sniffer = create_sniffer(
    server_endpoint='http://127.0.0.1:5000/send_traffic',
    verbose=True,
    sending_interval=1,
    input_interface='lo',
)

sniffer.start()
try:
    sniffer.join()
except KeyboardInterrupt:
    print('Stopping the sniffer')
    sniffer.stop()
finally:
    sniffer.join()