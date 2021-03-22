from pathlib import Path 
import dill

def export_keras(model, output_path:Path):
    d = model.get_params()
    d['filepath'] = model.model_filepath

    with open(output_path, 'wb') as file:
        dill.dump(d, file)


def import_keras(model_cls, output_path:Path, model_path:Path):
    with open(output_path, 'rb') as file:
        d = dill.load(file)
    filepath = model_path / d.pop('filepath')
    model = model_cls(**d)
    model._model_filepath = filepath
    model.load_best_model()
    return model