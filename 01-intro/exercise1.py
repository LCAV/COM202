import base64

encrypted_message = b'Vm0weE1HRXlWbFJsYkZaVFlteGthVmx0ZUV0V01WbDNXa2M1VjFadGVGZFNNbEYzVUZFOVBRPT0='
message = base64.b64decode(encrypted_message).decode('utf-8')

print(message)
