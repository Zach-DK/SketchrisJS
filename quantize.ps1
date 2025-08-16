@"
import onnx
from onnxconverter_common import float16
m = onnx.load('tetris_classifier.onnx')
m16 = float16.convert_float_to_float16(m, keep_io_types=True)
onnx.save(m16, 'tetris_classifier_fp16.onnx')
print('Saved tetris_classifier_fp16.onnx')
"@ | Set-Content quantize_fp16.py
python quantize_fp16.py