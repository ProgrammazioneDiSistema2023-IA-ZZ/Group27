import onnx_rust
import logging
from prettytable import PrettyTable

logging.getLogger().setLevel(logging.DEBUG)

x= onnx_rust.inference(
    "./onnxFile/googlenet-12/googlenet-12.onnx",
    "./onnxFile/googlenet-12/test_data_set_0/input_0.pb"
)
expected_output = onnx_rust.read_data_file("./onnxFile/googlenet-12/test_data_set_0/output_0.pb")

table = PrettyTable()

for key in x.keys():
    table.add_column("Expected "+key,x[key][0])
    table.add_column("Result "+key,expected_output[key][0])
print(table)