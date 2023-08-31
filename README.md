# Group27

## Progetto 2.1 - Configurazione ed esecuzione ottimizzata di una rete neurale generica descritta con standard ONNX

### Build & Run (Rust)

```shell
# Build
cargo build --release

# Run
./target/release/onnx-cli <model> --input <input_file> [--output <output_file>]
```
* `model`: Modello in formato .onnx da leggere
* `input_file`: Dati in ingresso al modello in formato .pb
* `output_file`: Eventuali dati di output attesi in formato .pb, stampati insieme all'output effettivo del modello, utile per valutare la correttezza dell'inferenza
### Test (Rust)
Nel file `tests/models.rs` sono presenti i test relativi ai 2 modelli usati per lo sviluppo dell'applicazione:
* `mnist`: Test relativo al modello <a href="https://github.com/onnx/models/tree/main/vision/classification/mnist">MNIST-12</a>
* `googlenet` : Test relativo al modello <a href="https://github.com/onnx/models/tree/main/vision/classification/inception_and_googlenet/googlenet">GoogLeNet-12</a>

Per valutare la correttezza dei modelli si assume che il risultato sia uguale valutandolo fino alla terza cifra significativa
### Build & Run (Python)

```shell
# Build
python/build.sh

# Start virtual environment
source .env/bin/activate

# Run python program
python python/test.py

# Close virtual environment when you want to exit 
deactivate
```

La libreria compilata si trova in `target/wheels`.

### Modelli scelti

<table>
    <thead>
        <tr align="center">
            <td><strong><a href="https://github.com/onnx/models/tree/main/vision/classification/mnist">MNIST-12</a></strong></td>
            <td><strong><a href="https://github.com/onnx/models/tree/main/vision/classification/inception_and_googlenet/googlenet">GoogLeNet-12</a></strong></td>
        </tr>
    </thead>
    <tbody>
        <tr align="center">
            <td width="500px"><img src="img/mnist.svg"></td>
            <td width="500px"><img src="img/googlenet.svg"></td>
        </tr>
    </tbody>
</table>