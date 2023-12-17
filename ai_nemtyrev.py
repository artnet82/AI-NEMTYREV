python
import llama2
import dataset_utils

class AI_NEMTYREV:
    def init(self, model_path):
        self.model = llama2.load_model(model_path)
    
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