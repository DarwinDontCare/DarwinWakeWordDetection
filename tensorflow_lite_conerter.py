import tensorflow as tf
import os
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb

model_path = input("caminho para o seu modelo: ").replace('"', "").replace(" ", "")
label_path = input("caminho para as labels do seu modelo: ").replace('"', "").replace(" ", "")
output_file = input("caminho onde será criado o arquivo: ").replace('"', "").replace(" ", "")
splited_path = model_path.split("\\")
model_name = splited_path[len(splited_path) - 1].split(".")[0]

if (".pb" in model_path):
  converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
  converter.experimental_new_converter = True
  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
  ]
  tflite_model = converter.convert()
  tflite_file = f"{output_file}\{model_name}.tflite"
  with open(tflite_file, "wb") as f:
    f.write(tflite_model)
  model_path = tflite_file

# Creates model info.
model_meta = _metadata_fb.ModelMetadataT()
model_meta.name = "Wake Word Detection"
model_meta.description = ("Identify the the phrase \"Ei Darwin\"")
model_meta.version = "v1"
model_meta.author = "Darwin don't care"
model_meta.license = ("MIT")

input_meta = _metadata_fb.TensorMetadataT()

# Creates output info.
output_meta = _metadata_fb.TensorMetadataT()


input_meta.name = "audio"
input_meta.description = ("receives a audio mfcc with 1 minute of duration")
input_meta.content = _metadata_fb.ContentT()
input_meta.content.content_properties = _metadata_fb.AudioPropertiesT()
input_meta.content.content_properties.sampleRate = 22050 
input_meta.content.content_properties.channels = 1
input_stats = _metadata_fb.StatsT()
input_stats.max = [400]
input_stats.min = [0]
input_meta.stats = input_stats

# Creates output info.
output_stats = _metadata_fb.StatsT()
output_meta.name = "probability"
output_meta.description = "Probabilities of the 0 and 1 labels respectively."
output_meta.content = _metadata_fb.ContentT()
output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
output_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)
output_stats.max = [1.0]
output_stats.min = [0.0]
output_meta.stats = output_stats

label_file = _metadata_fb.AssociatedFileT()
label_file.name = os.path.basename(label_path)
label_file.description = "Labels for objects that the model can recognize."
label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
output_meta.associatedFiles = [label_file]

subgraph = _metadata_fb.SubGraphMetadataT()
subgraph.inputTensorMetadata = [input_meta]
subgraph.outputTensorMetadata = [output_meta]
model_meta.subgraphMetadata = [subgraph]

b = flatbuffers.Builder(0)
b.Finish(
    model_meta.Pack(b),
    _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
metadata_buf = b.Output()

populator = _metadata.MetadataPopulator.with_model_file(f"{output_file}\{model_name}.tflite")
populator.load_metadata_buffer(metadata_buf)
populator.load_associated_files([label_path])
populator.populate()

displayer = _metadata.MetadataDisplayer.with_model_file(f"{output_file}\{model_name}.tflite")
json_file = displayer.get_metadata_json()

with open(f"metadata\{model_name}_metadata.json", "w") as f:
  f.write(json_file)

print("Conversão bem-sucedida")
