import onnx_rust

import logging



logging.getLogger().setLevel(logging.DEBUG)

#grafo = onnx_rust.leggi_file("/home/alberto/Dropbox (Politecnico Di Torino Studenti)/Programmazione di Sistema/onnx-rust/onnxFile/mnist-12/mnist-12.onnx")
#print(grafo)


x= onnx_rust.interferenza(
    "onnxFile/mnist-12/mnist-12.onnx",
    "onnxFile/mnist-12/test_data_set_0/input_0.pb"
)
expected_output = onnx_rust.leggi_file_dati("onnxFile/mnist-12/test_data_set_0/output_0.pb")

print("Result of interference :",x)
print("Expected output :",expected_output)
