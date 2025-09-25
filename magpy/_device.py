class DeviceManager:
    def __init__(self, device):
        self.device = device

_DEVICE_CONTEXT = DeviceManager('cpu')
