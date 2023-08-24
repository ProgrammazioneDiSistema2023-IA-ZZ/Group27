import onnx_rust

import logging
logging.getLogger().setLevel(logging.DEBUG)

#grafo = onnx_rust.leggi_file("/home/alberto/Dropbox (Politecnico Di Torino Studenti)/Programmazione di Sistema/onnx-rust/onnxFile/googlenet-12/googlenet-12.onnx")
#print(grafo)

x= onnx_rust.inference(
    "./onnxFile/googlenet-12/googlenet-12.onnx",
    "./onnxFile/googlenet-12/test_data_set_0/input_0.pb"
)
expected_output = onnx_rust.read_data_file("./onnxFile/googlenet-12/test_data_set_0/output_0.pb")

print("Result of inference :",x)
print("Expected output :",expected_output)
