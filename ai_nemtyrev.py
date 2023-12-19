python
import nemtyrev_ai
import dataset_utils

class AI_NEMTYREV:
    def init(self, model_path):
        self.model = nemtyrev_ai.load_model(model_path)
    
    def process_dataset(self, dataset_path):
        dataset = dataset_utils.load_dataset(dataset_path)
        dataset.shuffle()
        
        for image, label in dataset:
            prediction = self.model.predict(image)
            object_label = prediction["object"]
            object_name = dataset.get_class_name(object_label)
            
            print("Объект:", object_name)
        
    def close_model(self):
        self.model.close()


//пример использования 
python
from ai_nemtyrev import AI_NEMTYREV

model_path = "path_to_model"(путь к модели) 
dataset_path = "path_to_dataset"(путь к дата сету) 

ai = AI_NEMTYREV(model_path)
ai.process_dataset(dataset_path)
ai.close_model()
// //
